from transforms3d import euler
import numpy as np
import xml.etree.ElementTree as ET
import pybullet
from config import *
import os

class Utils:
    def __init__(self):
        pass

    def find_project_root(self, project_name="hybrid_action_space"):
        path = os.path.abspath(__file__)
        while True:
            if os.path.basename(path) == project_name:
                return path
            new_path = os.path.dirname(path)
            if new_path == path:
                raise RuntimeError(f"Project root '{project_name}' not found.")
            path = new_path
            return path

    # randomly sampled arbitrary contacting points on horizontal surface
    def sample_point_on_top_of_arbitrary(self, object_id, margin=0.01, max_tries=10, seed=None):
        """
        Ray‐casts down through a randomly chosen (x,y) inside the object's AABB
        (shrunk by 'margin') to find a point on its top surface.
        Returns:
          sampled_world_point (3,)
        Raises:
          RuntimeError if no hit after max_tries.
        """
        np.random.seed(seed)
        aabb_min, aabb_max = pybullet.getAABB(object_id)
        amin = np.array(aabb_min)
        amax = np.array(aabb_max)
        # shrink the XY region by margin
        x_min, y_min = amin[0] + margin, amin[1] + margin
        x_max, y_max = amax[0] - margin, amax[1] - margin

        for _ in range(max_tries):
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            # cast from just above the top down past the bottom
            start = [x, y, amax[2] + 0.05]
            end = [x, y, amin[2] - 0.05]
            hits = pybullet.rayTest(start, end)
            # (hitObjectId, hitLinkIndex, hitFraction, hitPosition, hitNormal)
            obj_hit, _, hit_pos, _ = hits[0][0], hits[0][2], hits[0][3], hits[0][4]
            if obj_hit == object_id:
                return np.array(hit_pos)
        raise RuntimeError(f"Could not sample top surface after {max_tries} tries")

    def world_to_local(self, point_world, object_id):
        """Convert a world‐frame point into the object’s local frame."""
        pos, orn = pybullet.getBasePositionAndOrientation(object_id)
        R = np.array(pybullet.getMatrixFromQuaternion(orn)).reshape(3, 3)
        return R.T.dot(point_world - np.array(pos))

    def local_to_world(self, point_local, object_id):
        """Convert a local‐frame point back into world‐coordinates."""
        pos, orn = pybullet.getBasePositionAndOrientation(object_id)
        R = np.array(pybullet.getMatrixFromQuaternion(orn)).reshape(3, 3)
        return R.dot(point_local) + np.array(pos)

    def read_config_from_node(self, root_node, parent_name, child_name, dtype=int):
        # find parent
        parent_node = root_node.find(parent_name)
        if parent_node is None:
            quit("Parent %s not found" % parent_name)

        # get child data
        child_data = parent_node.get(child_name)
        if child_data is None:
            quit("Child %s not found" % child_name)

        config_val = np.array(child_data.split(), dtype=dtype)
        return config_val

    def get_config_root_node(self, config_file_name=None, config_file_data=None):
        # get root
        if config_file_data is None:
            with open(config_file_name) as config_file_content:
                config = ET.parse(config_file_content)
            root_node = config.getroot()
        else:
            root_node = ET.fromstring(config_file_data)

        # get root data
        root_data = root_node.get("name")
        assert isinstance(root_data, str)
        root_name = np.array(root_data.split(), dtype=str)

        return root_node, root_name

    def eulerXYZ_to_quatXYZW(self, rotation):  # pylint: disable=invalid-name
        """Abstraction for converting from a 3-parameter rotation to quaterion.

        This will help us easily switch which rotation parameterization we use.
        Quaternion should be in xyzw order for pybullet.

        Args:
        rotation: a 3-parameter rotation, in xyz order tuple of 3 floats

        Returns:
        quaternion, in xyzw order, tuple of 4 floats
        """
        euler_zxy = (rotation[2], rotation[0], rotation[1])
        quaternion_wxyz = euler.euler2quat(*euler_zxy, axes='szxy')
        q = quaternion_wxyz
        quaternion_xyzw = (q[1], q[2], q[3], q[0])
        return quaternion_xyzw

    def quatXYZW_to_eulerXYZ(self, quaternion_xyzw):  # pylint: disable=invalid-name
        """Abstraction for converting from quaternion to a 3-parameter toation.

        This will help us easily switch which rotation parameterization we use.
        Quaternion should be in xyzw order for pybullet.

        Args:
        quaternion_xyzw: in xyzw order, tuple of 4 floats

        Returns:
        rotation: a 3-parameter rotation, in xyz order, tuple of 3 floats
        """
        q = quaternion_xyzw
        quaternion_wxyz = np.array([q[3], q[0], q[1], q[2]])
        euler_zxy = euler.quat2euler(quaternion_wxyz, axes='szxy')
        euler_xyz = (euler_zxy[1], euler_zxy[2], euler_zxy[0])
        return euler_xyz

    # --------------
    # Camera setting
    # --------------

    def render_image(self,
                     image_size,
                     intrinsics,
                     position,
                     orientation,
                     zrange,
                     noise=False
                     ):
        # OpenGL camera settings.
        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        rotation = pybullet.getMatrixFromQuaternion(orientation)
        rotm = np.float32(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = position + lookdir
        focal_len = intrinsics[0]
        znear, zfar = zrange
        viewm = pybullet.computeViewMatrix(position, lookat, updir)
        fovh = (image_size[0] / 2) / focal_len
        fovh = 180 * np.arctan(fovh) * 2 / np.pi

        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        aspect_ratio = image_size[1] / image_size[0]
        projm = pybullet.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

        # Render with OpenGL camera settings.
        _, _, color, depth, segm = pybullet.getCameraImage(
            width=image_size[1],
            height=image_size[0],
            viewMatrix=viewm,
            projectionMatrix=projm,
            shadow=1,
            flags=pybullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)

        # Get color image.
        color_image_size = (image_size[0], image_size[1], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel
        if noise:
            color = np.int32(color)
            color += np.int32(np.random.normal(0, 3, color.shape))
            color = np.uint8(np.clip(color, 0, 255))

        # Get depth image.
        depth_image_size = (image_size[0], image_size[1])
        zbuffer = np.float32(depth).reshape(depth_image_size)
        depth = (zfar + znear - (2 * zbuffer - 1) * (zfar - znear))
        depth = (2 * znear * zfar) / depth
        if noise:
            depth += np.random.normal(0, 0.003, depth.shape)

        intrinsics = np.float32(intrinsics).reshape(3, 3)
        return color, depth, position, orientation, intrinsics