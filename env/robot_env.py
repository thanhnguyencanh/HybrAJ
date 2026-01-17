import os
import numpy as np
from utils.suction import Suction
import pybullet
import pybullet_data
from reward.individual_reward import Reward
from gym.spaces import Discrete, Box
import gym
from utils.random_env import random_env
from config import *
from utils.utils import Utils
from utils import camera


# @markdown **Gym-style environment class:** this initializes a robot overlooking a workspace with objects.
class ImitationLearning(gym.Env):
    def __init__(self):
        self.dt = 1 / 400  # smaller dt gets better accuracy -> physic update frequency
        # Configure and start PyBullet.
        pybullet.connect(pybullet.DIRECT)  # pybullet.GUI for local GUI.
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
        pybullet.setPhysicsEngineParameter(enableFileCaching=0)
        assets_path = os.path.dirname(os.path.abspath(""))
        pybullet.setAdditionalSearchPath(assets_path)
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        pybullet.setPhysicsEngineParameter(numSolverIterations=100)  # physic engine step per time step
        pybullet.setPhysicsEngineParameter(contactERP=0.9, solverResidualThreshold=0)
        pybullet.setTimeStep(self.dt)

        self.home_joints = (-np.pi / 2, -np.pi / 15, -2 * np.pi / 9, np.pi, 2 * np.pi / 7, 0)  # Initialize angles.
        self.ee_link_id = 6  # Link ID of UR5 end effector.
        self.suction = None
        self.action_space = Box(low=-1, high=1, shape=(6,), dtype=np.float32)  # rad/s
        self.discrete_action = Discrete(2)  # {0, 1}
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(37,), dtype=np.float32)
        self.agent_cams = camera.Camera.CONFIG
        self.state = None
        self.robot = "uf850"
        self.action_type = None
        self.max_episode = None
        self.utils = Utils()  # if moving into reset function -> remove ../ in config path
        self.curriculum_learning = None
        self.noise = True
        self.frame = 5
        self.episode = 0
        self.max_steps = None
        self.virtualize = False
        self.eval = False
        self.workspace_bound = np.array([
            BOUNDS[0][-1] - BOUNDS[0][0],
            BOUNDS[1][-1] - BOUNDS[1][0],
            BOUNDS[2][-1] - BOUNDS[2][0]
        ])
        self._read_specs_from_config(os.path.abspath(
            os.path.join(self.utils.find_project_root("HybrAJ"), f'../asset/params/{self.robot}_suction.xml')))

    def seed(self, seed):
        print('set seed')
        self.np_random, _ = gym.utils.seeding.np_random(seed)

    def _load(self, config):
        ''' Generate object in env'''
        self.config = config  # Required objects
        self.obj_ids = []
        self.obj_name_to_id = {}  # Object id
        pybullet.resetSimulation(pybullet.RESET_USE_DEFORMABLE_WORLD)
        pybullet.setGravity(0, 0, -9.8)
        # Temporarily disable rendering to load URDFs faster.
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

        # Add robot.
        self.plane = pybullet.loadURDF("plane.urdf", [0, 0, -0.001])
        self.robot_id = pybullet.loadURDF(
            os.path.abspath(os.path.join(self.utils.find_project_root("HybrAJ"), f"../asset/{self.robot}/{self.robot}.urdf"))
            , [0, 0, 0], useFixedBase=True, flags=pybullet.URDF_USE_MATERIAL_COLORS_FROM_MTL)
        self.ghost_id = pybullet.loadURDF(
            os.path.abspath(os.path.join(self.utils.find_project_root("HybrAJ"), f"../asset/{self.robot}/{self.robot}.urdf"))
            , [0, 0, -10])  # For forward kinematics.
        self.joint_ids = [pybullet.getJointInfo(self.robot_id, i) for i in range(pybullet.getNumJoints(self.robot_id))]
        self.joint_ids = [j[0] for j in self.joint_ids if j[2] == pybullet.JOINT_REVOLUTE]

        # Move robot to home configuration.
        for i in range(len(self.joint_ids)):
            pybullet.resetJointState(self.robot_id, self.joint_ids[i], self.home_joints[i])
        # Add workspace.
        plane_shape = pybullet.createCollisionShape(pybullet.GEOM_BOX, halfExtents=[0.45, 0.35, 0.003])
        plane_visual = pybullet.createVisualShape(pybullet.GEOM_BOX, halfExtents=[0.45, 0.35, 0.003])
        plane_id = pybullet.createMultiBody(0, plane_shape, plane_visual, basePosition=[0, -0.5, 0])
        pybullet.changeVisualShape(plane_id, -1, rgbaColor=[0.2, 0.2, 0.2, 1.0])

        # Load objects according to config.
        obj_names = list(self.config['pick']) + list(self.config['place'])
        obj_xyz = np.zeros((0, 3))
        for obj_name in obj_names:
            if obj_name not in list(fixed_destination.keys()):
                # Get random position 15cm+ from other objects.
                while True:
                    rand_x = self.np_random.uniform(BOUNDS[0, 0], BOUNDS[0, 1])
                    rand_y = self.np_random.uniform(BOUNDS[1, 0], BOUNDS[1, 1])
                    rand_xyz = np.float32([rand_x, rand_y, 0.03]).reshape(1, 3)
                    if len(obj_xyz) == 0:
                        obj_xyz = np.concatenate((obj_xyz, rand_xyz), axis=0)
                        break
                    else:
                        nn_dist = np.min(np.linalg.norm(obj_xyz - rand_xyz, axis=1)).squeeze()
                        if nn_dist > 0.13:
                            obj_xyz = np.concatenate((obj_xyz, rand_xyz), axis=0)
                            break

                object_color = COLORS[obj_name.split(' ')[0]]
                object_type = obj_name.split(' ')[1]
                object_position = rand_xyz.squeeze()
                if object_type == 'block':
                    object_shape = pybullet.createCollisionShape(pybullet.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])
                    object_visual = pybullet.createVisualShape(pybullet.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])
                    object_id = pybullet.createMultiBody(0.01, object_shape, object_visual,
                                                         basePosition=object_position)
                elif object_type == 'egg':
                    object_shape = pybullet.createCollisionShape(pybullet.GEOM_SPHERE, 0.03, [1.0, 1.0, 1.4])
                    object_visual = pybullet.createVisualShape(pybullet.GEOM_SPHERE, 0.03, [1.0, 1.0, 1.4])
                    object_id = pybullet.createMultiBody(0.05, object_shape, object_visual,
                                                         basePosition=object_position)
                else:
                    object_position[2] = 0
                    object_id = pybullet.loadURDF(os.path.abspath(
                        os.path.join(self.utils.find_project_root("HybrAJ"),
                                     f"../asset/{object_type}/{object_type}.urdf")), object_position,
                                                  useFixedBase=1 if 'bowl' in object_type else 0)
                pybullet.changeDynamics(object_id, -1,
                                        lateralFriction=1.0,
                                        spinningFriction=0.01,
                                        rollingFriction=0.01,
                                        restitution=0.0,  # No bounciness
                                        linearDamping=0.5,  # Stops sliding
                                        angularDamping=0.5,  # Stops rotation
                                        frictionAnchor=1,  # Experimental: Helps improve friction stability
                                        activationState=pybullet.ACTIVATION_STATE_ENABLE_SLEEPING)  # Allows object to "freeze" when not moving
                # Calculate dimensions
                self.obj_ids.append(object_id)
                # print(self.object_properties)
                pybullet.changeVisualShape(object_id, -1, rgbaColor=object_color)
                self.obj_name_to_id[obj_name] = object_id

        self.suction = Suction(self.robot_id, self.ee_link_id, self.obj_ids)  # Gripper
        self.suction.release()

        # Re-enable rendering.
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
        for _ in range(200):
            pybullet.stepSimulation()
        self._read_state()
        print('Environment reset: done.')

    def reset(self):
        '''
        Recreate environment and observation
        '''
        print(f"Training Action: {self.action_type}")
        self._init_history()
        self.observation = None
        self.cache_front_video = []  # for front view visualization
        self.cache_side_video = []  # for side view visualization
        self.cache_diagonal_video = []  # for diagonal visualization
        self.success_buffer = []
        self.pick_object_id = None
        self.place_object_id = None
        self.sim_step = 0
        self.stable_factor = 0
        self.seed(self.episode)
        config, instruction = random_env(self.action_type, seed=self.episode)
        self._load(config)
        # read joint positions
        self._read_state()
        self.last_commanded_position = self.state[:6]
        object1 = config["pick"][self.np_random.randint(0, len(config["pick"]))]
        object2 = config["place"][self.np_random.randint(0, len(config["place"]))]
        # providing instruction
        instruction = instruction.format(object1, object2) if instruction.count("{}") == 2 else instruction.format(
            object1)
        self.info = [action_state[instruction.split(" ")[0]], object1, object2]
        # id assignment
        self.pick_object_id = self.obj_name_to_id[self.info[1]]
        if self.info[2] not in fixed_destination.keys():
            self.place_object_id = self.obj_name_to_id[self.info[2]]
        if self.info[2] in fixed_destination.keys():
            self.place = np.array(fixed_destination[self.info[2]])
        else:
            self.place = self._read_info(self.place_object_id)[:3]  # ori shape: [3,]

        self.target = self._coor(self._read_info(self.pick_object_id)[:3], self.place,
                                 self.info[0])  # target determination

        self.ee_obj = np.float32(pybullet.getLinkState(self.suction.body, 0)[0]) - self._read_info(self.pick_object_id)[
                                                                                   :3]  # ee and obj relative distance
        self.obj_target = self._read_info(self.pick_object_id)[7:10] - self.target  # obj and target relative distance

        self.task_state_vector = np.concatenate((self.ee_obj / self.workspace_bound,
                                                 self._read_info(self.pick_object_id)[3:7],
                                                 self.obj_target / self.workspace_bound,
                                                 self._read_info(self.pick_object_id)[:3] / self.workspace_bound,
                                                 self.target / self.workspace_bound,
                                                 ))  # shape: [16,]

        self.task_state_vector += self._random_noise(self.task_state_vector, 0.005)  # randomize

        self.reward_func = Reward(dt=self.dt * self.frame, env=self)
        self.observation = np.concatenate((self.state,  # joint position + joint velocity
                                           np.zeros(2),  # suction state, contact state
                                           self.task_state_vector,
                                           np.zeros(7),  # previous action
                                           ))
        self._save_data(np.zeros(7), "action")  # t-2 and t-1 action
        self._save_data(np.zeros(7), "action")  # t-2 and t-1 action
        self.episode += 1

        return self.observation

    def _do_simulation(self):
        if self.virtualize:
            for _ in range(self.frame):
                self.step_sim_and_render()
        else:
            for _ in range(self.frame):
                pybullet.stepSimulation()

    def step(self, action):
        """
        robot execution function
        rvaluate actions through its subsequent sub-actions
        """
        continuous_action, discrete_action = action[:-1], action[-1]
        self._read_state()
        self._save_data(np.float32(pybullet.getLinkState(self.suction.body, 0)[0]), "ee_history")  # shape: [3,])
        self._save_data(self._read_info(self.pick_object_id), "obj_history")
        self._save_data(np.concatenate((continuous_action, discrete_action)),
                        "action")  # save previous predicted actions and angles

        arm_joint_ctrl = np.clip(continuous_action, -1.0, 1.0)
        # Compute single target position from converted velocities
        ctrl_feasible = self._velocity_to_position_control(arm_joint_ctrl)

        self._save_data(ctrl_feasible, "joint")
        # Apply position control once
        self._servoj(ctrl_feasible)
        self._do_simulation()
        self._read_state()  # Final state after control
        self.last_commanded_position = self.state[:6]

        self.suction.activate() if bool(int(discrete_action[0])) else self.suction.release()  # suction de/activate
        self._do_simulation()

        # Get current data
        self._save_data([bool(discrete_action[0]), self._read_contact(self.info[0])],
                        "contact_history")  # suction signal, contacting signal
        self._save_data(np.float32(pybullet.getLinkState(self.suction.body, 0)[0]), "ee_history")  # ee position
        self._save_data(self._read_info(self.pick_object_id), "obj_history")  # shape: [3,] store new movement position

        if self.place_object_id is not None:
            self.history["target_obj"].append(
                self._read_info(self.obj_name_to_id[self.info[2]])
            )  # shape: [3,] store new movement position

        self.ee_obj = np.float32(pybullet.getLinkState(self.suction.body, 0)[0]) - self._read_info(self.pick_object_id)[
                                                                                   :3]
        self.obj_target = self._read_info(self.pick_object_id)[7:10] - self.target

        # Compute reward
        reward, done, success = self.reward_func.get_reward()

        self.task_state_vector = np.concatenate((self.ee_obj / self.workspace_bound,
                                                 self._read_info(self.pick_object_id)[3:7],
                                                 self.obj_target / self.workspace_bound,
                                                 self._read_info(self.pick_object_id)[:3] / self.workspace_bound,
                                                 self.target / self.workspace_bound,
                                                 ))
        self.task_state_vector += self._random_noise(self.task_state_vector, 0.005)  # randomize
        # Update next state
        self.observation = np.concatenate((self.state,
                                           [discrete_action[0],
                                            self._read_contact(self.info[0])],
                                           self.task_state_vector,
                                           self.history["action"][-1],
                                           ))  # 37D dimensional vector
        self.success_buffer.append(success)
        info = {'log': self.success_buffer}  # task successful logger
        if (self.action_type == 3 or self.action_type == 0) and self.eval:
            if success:
                self.stable_factor += 1
                if self.stable_factor == 2:
                    self.suction.release()
                    self._do_simulation()
                    done = True
                    self._do_simulation()
            else:
                self.stable_factor = 0

        return self.observation, reward, done, info

    def _save_data(self, data, attr: str):
        self.history[attr].append(data)

    def _velocity_to_position_control(self, velocity_action: np.ndarray) -> np.ndarray:
        """
        Converts a policy's velocity action into a safe, bounded position command.
        Args:
            velocity_action (np.ndarray): The raw velocity command from the policy (e.g., in range [-1, 1]).
        Returns:
            safe_position_command (np.ndarray): The final joint position command sent to the actuators.
        """
        clipped_velocity = np.clip(
            velocity_action, self.robot_vel_bound[:6, 0], self.robot_vel_bound[:6, 1]
        )
        # (Here we assume the last element is the suction command and doesn't affect position)
        target_position = self.last_commanded_position + clipped_velocity * 0.03  # max change will be 1.7 deg

        safe_position_command = np.clip(
            target_position, self.robot_pos_bound[:6, 0], self.robot_pos_bound[:6, 1]
        )
        # self.last_commanded_position = safe_position_command

        return safe_position_command

    def _coor(self, pick_xyz, place_xyz, action):
        """
        Compute target coordinates for touch, pick, move, place, and push actions.
        Args:
            pick_xyz (np.ndarray): pick position
            place_xyz (np.ndarray): place position
            action (str): action type ('0', '1', '2', or other)
        Returns:
            np.ndarray: target coordinates
        """
        hover_xyz = pick_xyz.copy() + np.float32([0, 0, 0.15])
        place_xyz_above = place_xyz.copy() + np.float32([0, 0, 0.15])
        place_xyz = place_xyz + np.float32([0, 0, 0.015])

        if action == '0':
            return pick_xyz
        elif action == '1':
            return hover_xyz
        elif action == '2':
            return place_xyz_above
        else:
            return place_xyz

    def _random_noise(self, shape, noise):
        return np.concatenate([
            self.np_random.uniform(-noise, noise, size=shape.shape[0])
        ])

    def _read_state(self):
        '''Update joint states information'''
        jointStates = pybullet.getJointStates(self.robot_id,
                                              self.joint_ids)  # (position, velocity, reaction_forces, applied_joint_motor_torque)

        jointPoses = np.array([x[0] for x in jointStates])
        jointVelocity = np.array([x[1] for x in jointStates])
        # jointTorque = [x[3] for x in jointStates]
        jointPoses += (
            self.np_random.uniform(low=-0.005, high=0.005, size=jointPoses.shape)
        )
        jointVelocity += (
            self.np_random.uniform(low=-0.05, high=0.05, size=jointVelocity.shape)
        )
        self.state = np.hstack((np.array(jointPoses).copy(), np.array(jointVelocity).copy()))

    def _read_info(self, obj_id):
        '''Update tracked object information'''
        pos, orn = pybullet.getBasePositionAndOrientation(obj_id)
        aabb_min, aabb_max = pybullet.getAABB(obj_id)
        center_coor_up = [(a + b) / 2 for a, b in zip(aabb_min, aabb_max)]
        center_coor_down = [(a + b) / 2 for a, b in zip(aabb_min, aabb_max)]

        center_coor_up[2] = aabb_max[2]  # target of end-effector
        center_coor_down[2] = aabb_min[2]  # target of moving towards desire destination

        return np.concatenate((
            center_coor_up,
            orn,
            center_coor_down
        ))  # shape: [10,]

    def _servoj(self, joints):
        """Move to target joint positions with position control.
        torque = Kp.(pt-pc) - Kd.q
        Directly move to desire angles"""
        pybullet.setJointMotorControlArray(
            bodyIndex=self.robot_id,
            jointIndices=self.joint_ids,
            controlMode=pybullet.POSITION_CONTROL,
            targetPositions=joints,
            positionGains=[1] * 6)

    def _read_contact(self, action: str):
        """ Check if the end effector contacts correct object
        Args: action: action type
        Returns: bool
        """
        target_obj_id = self.obj_name_to_id[self.info[1]]
        points = pybullet.getContactPoints(bodyA=self.suction.body, linkIndexA=0)
        if points:
            for point in points:
                obj_id = point[2]
                if obj_id == target_obj_id:
                    return True
            return False
        return False

    def _read_specs_from_config(self, robot_configs: str):
        """Read the specs of the UR5 robot joints from the config xml file.
                - pos_bound: position limits of each joint.
                - vel_bound: velocity limits of each joint.
                - pos_noise_amp: scaling factor of the random noise applied in each observation of the robot joint positions.
                - vel_noise_amp: scaling factor of the random noise applied in each observation of the robot joint velocities.
            Args:
                robot_configs (str): path to 'ur5_suction.xml'
            """
        root, root_name = self.utils.get_config_root_node(config_file_name=robot_configs)
        robot_name = root_name[0]
        print(robot_name)
        self.robot_pos_bound = np.zeros([self.action_space.shape[0], 2], dtype=float)
        self.robot_vel_bound = np.zeros([self.action_space.shape[0], 2], dtype=float)
        self.robot_pos_noise_amp = np.zeros(self.action_space.shape[0], dtype=float)
        self.robot_vel_noise_amp = np.zeros(self.action_space.shape[0], dtype=float)

        for i in range(self.action_space.shape[0]):
            self.robot_pos_bound[i] = self.utils.read_config_from_node(
                root, "qpos" + str(i), "pos_bound", float
            )
            self.robot_vel_bound[i] = self.utils.read_config_from_node(
                root, "qpos" + str(i), "vel_bound", float
            )
            self.robot_pos_noise_amp[i] = self.utils.read_config_from_node(
                root, "qpos" + str(i), "pos_noise_amp", float
            )[0]
            self.robot_vel_noise_amp[i] = self.utils.read_config_from_node(
                root, "qpos" + str(i), "vel_noise_amp", float
            )[0]

    def step_sim_and_render(self):
        pybullet.stepSimulation()
        self.sim_step += 1
        if self.sim_step % 5 == 0:
            self.cache_front_video.append(self.render(mode='rgb_array', view=0))  # front view
            self.cache_side_video.append(self.render(mode='rgb_array', view=1))  # side view
            self.cache_diagonal_video.append(self.render(mode='rgb_array', view=2))  # diagonal view

    def _init_history(self):
        self.history = {
            "obj_history": [],
            "ee_history": [],
            "action": [],
            "contact_history": [],
            "target_obj": [],
            "joint": [],
        }

    def render(self, mode='rgb_array', view=0):
        if mode != 'rgb_array':
            raise NotImplementedError('Only rgb_array implemented')
        color, _, _, _, _ = self.utils.render_image(**self.agent_cams[view])
        return color

