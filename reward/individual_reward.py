import numpy as np
from config import *
import pybullet
import os

class Reward:
    def __init__(self,
                 dt=None,
                 env=None
                 ):
        """
        dt: simulation time step
        env: object of training environment
        """
        self.dt = dt
        self.step = 0  # for terminating
        self.env = env
        self._read_specs_from_config(os.path.abspath(os.path.join(self.env.utils.find_project_root("drl_transporterNet"), '../asset/params/reward.xml')))
        self.first_success_step = 0
        self.remain = 15
        self.scale_factor = 10

    def _read_specs_from_config(self, robot_configs: str):
        """Read the specs of the reward params from the config xml file.
                - threshold0: success distance threshold.
                - threshold0: object distance threshold.
                - sigma0: scaling factor of the reward 0
                - sigma1: scaling factor of the reward 1
                - sigma2: scaling factor of the reward 2
                - sigma3: scaling factor of the reward 3
            """
        root, root_name = self.env.utils.get_config_root_node(config_file_name=robot_configs)
        robot_name = root_name[0]

        self.dist_threshold = self.env.utils.read_config_from_node(root, "r" + self.env.info[0], "threshold0", float)[0]
        self.obj_threshold = self.env.utils.read_config_from_node(root, "r" + self.env.info[0], "threshold1", float)[0]
        sigma = []
        for i in range(4):
            value = self.env.utils.read_config_from_node(
                root, "r" + self.env.info[0], f"sigma{i}", float)[0]
            sigma.append(value)

        self.sigma0, self.sigma1, self.sigma2, self.sigma3 = sigma

    def _check_collisions(self):
        """
        Check for collisions involving the robot:
            - self-collision
            - ground collision
            - collision with objects (optional)
        Returns:
            collision_penalty (float): total penalty from collisions
        """
        collision_penalty = 0.0
        pick_object_collision = False
        place_object_collision = False
        self_collision = False
        ground_collision = False

        #  self-collision check (between different links only)
        # (contactFlags, bodyUniqueIdA, bodyUniqueIdB, linkIndexA, linkIndexB, ...)
        for contact in pybullet.getContactPoints(self.env.robot_id, self.env.robot_id):
            if contact[3] != contact[4]:  # Ignore same-link contacts
                collision_penalty += 3.0
                self_collision = True
                break  # early exit if collision already detected

        #  suction-collision check (between different links only)
        for contact in pybullet.getContactPoints(self.env.suction.body, self.env.robot_id):
            if contact[3] != contact[4]:  # Ignore same-link contacts
                collision_penalty += 3.0
                self_collision = True
                break  # early exit if collision already detected

        # Ground collision check
        if pybullet.getContactPoints(self.env.robot_id, self.env.plane) \
                or pybullet.getContactPoints(self.env.suction.body, self.env.plane):
            ground_collision = True
            collision_penalty += 3.0

        # Object 1 collision checking (robot only except suction body)
        if self.env.pick_object_id is not None:
            if pybullet.getContactPoints(self.env.robot_id, self.env.pick_object_id):
                pick_object_collision = True
                collision_penalty += 1.5

        for object_id in [obj for obj in self.env.obj_ids if obj != self.env.pick_object_id]:
            # Object 2 as target object collision checking
            if pybullet.getContactPoints(self.env.robot_id, object_id):
                place_object_collision = True
                collision_penalty += 1.5

            if pybullet.getContactPoints(self.env.suction.body, object_id):
                place_object_collision = True
                collision_penalty += 1.5

        return collision_penalty, self_collision, ground_collision, pick_object_collision, \
               place_object_collision

    def _outside_workspace(self, coords):
        """
        Calculate workspace violation penalty based on distance outside bounds.
        Args:
            coords: List or tuple of [x, y] coordinates
        Returns:
            float: Penalty value (0 to -1.0)
        """
        x, y, z = coords
        bounds = BOUNDS.copy()
        ws_x, ws_y = 0.0, 0.0

        # X-axis violation
        if x > bounds[0][1]:
            ws_x = abs(x - bounds[0][1]) / 0.05
        elif x < bounds[0][0]:
            ws_x = abs(x - bounds[0][0]) / 0.05

        # Y-axis violation
        if y > bounds[1][1]:
            ws_y = abs(y - bounds[1][1]) / 0.05
        elif y < bounds[1][0]:
            ws_y = abs(y - bounds[1][0]) / 0.05

        total_violation = ws_x + ws_y
        return total_violation if total_violation > 0 else 0.0, True if total_violation > 0 else False

    def _check_angle_constraints(self, joints):
        """
        Check if any joint exceeds its angle limits.
        Args:
            joints: 6 joint angle values
        Returns:
            float: Penalty for exceeding joint limits (negative or 0)
            violation (bool): True if any joint limit is violated
        """
        total_penalty = 0.0
        violation = False

        for joint_idx, (lower_limit, upper_limit) in enumerate(JOINT_LIMITS):
            # joint = pybullet.getJointState(self.robot_id, joint_idx)
            current_angle = joints[joint_idx]  # Current joint angle (radians)
            # Check if angle is outside bounds
            if current_angle < lower_limit:
                total_penalty += (lower_limit - current_angle)
                violation = True
            elif current_angle > upper_limit:
                total_penalty += (current_angle - upper_limit)
                violation = True

        return abs(total_penalty), violation

    # velocity constrain
    def _check_suction_orientation(self):
        """Force to move suction with parallel - yaw pose"""
        ee_orientation = pybullet.getLinkState(self.env.robot_id, self.env.ee_link_id)[1]
        ee_euler = pybullet.getEulerFromQuaternion(ee_orientation)  # (roll, pitch, yaw)
        roll, pitch = ee_euler[0], ee_euler[1]
        roll_target = np.pi  # Vertical fingers
        pitch_target = 0.0
        roll_error = abs(roll - roll_target)
        pitch_error = abs(pitch - pitch_target)
        return roll_error + pitch_error

    def _align_velocity(self, current_obj_pos, previous_obj_pos, target):
        """Encourage moving towards goal
            Args:
                current_obj_pos: current object position
                previous_obj_pos: previous object position
                target: target destination
            Returns:
                float: reward/ penalty
        """
        # Velocity vector
        vel = (current_obj_pos - previous_obj_pos) / self.dt

        # Direction vectors (unit vectors)
        vel_norm = np.linalg.norm(vel)
        goal_dir = target - current_obj_pos
        goal_dir_norm = np.linalg.norm(goal_dir)

        if vel_norm <= 0.01 or goal_dir_norm <= 0.01:
            return 0.0  # no movement or undefined direction

        vel_unit = vel / vel_norm
        goal_unit = goal_dir / goal_dir_norm

        alignment = np.dot(vel_unit, goal_unit)  # cos(theta) ∈ [-1, 1], expectation: 1

        return alignment

    def _check_inclination(self):
        """check object's pitch and roll inclination"""
        orientation = self.buf['obj_history'][-1][3:7]  # Adjust index based on your buffer structure
        euler = pybullet.getEulerFromQuaternion(orientation)
        pitch, roll = euler[1], euler[0]  # Scalars
        max_incl = np.deg2rad(8)  # 10 degrees
        incl = max(0, abs(pitch) - max_incl) + max(0, abs(roll) - max_incl)  # Corrected penalty

        return abs(incl)

    def _is_subgoal_achieved(self):
        """Check if the current sub-goal is achieved based on distance and contact state.
        Returns:
            bool: True if current task is successful else False
            """
        current_ee = self.buf['ee_history'][-1]  # Current end-effector state
        current_obj_up = self.buf['obj_history'][-1][:3]  # Current object position
        current_obj_down = self.buf['obj_history'][-1][7:10]  # Current object position
        contact_condition = False
        contact = self.buf['contact_history']
        obj_po = np.linalg.norm(current_obj_down - self.env.target)  # Tracking object position
        cur_dist = np.linalg.norm(current_ee - current_obj_up)

        # Sub-goals with contact conditions (e.g., suction activation)
        if self.env.info[0] in ['1', '2', '3']:  # For pick/move/place: suction should be active
            contact_condition = contact[-1][0] and contact[-1][1]
        elif self.env.info[0] in ['0']:  # For touch/push: check success contact only
            contact_condition = contact[-1][1]  # Check success contact only

        if self.env.info[0] in ['0']:
            # return contact_condition
            return cur_dist < self.dist_threshold or contact_condition
        elif self.env.info[0] in ['1', '2', '3']:
            return contact_condition and obj_po < self.obj_threshold
        else:
            return False

    def get_reward(self):
        """Compute reward and done flag for the current step.
        Returns:
            float: reward/ penalty
            bool: termination
            bool: task achieved
        """
        self.buf = self.env.history
        self.step += 1
        reward_2 = 0.0
        reward_3 = 0.0

        # Check if episode exceeds max steps
        if self.step == self.env.max_steps:
            return -2.0/self.scale_factor, True, False  # penalty if still remaining sub-action

        # Retrieve data
        contact = self.buf['contact_history']
        prev_obj_up = self.buf['obj_history'][-2][:3]  # previous object position (top coordination)
        current_obj_up = self.buf['obj_history'][-1][:3]  # Current object position (top coordination)
        prev_obj_down = self.buf['obj_history'][-2][7:10]  # previous object position (bottom coordination)
        current_obj_down = self.buf['obj_history'][-1][7:10]  # Current object position (bottom coordination)
        target_obj = self.buf['target_obj']  # Target object position
        current_ee = self.buf['ee_history'][-1]  # Current suction head position
        prev_ee = self.buf['ee_history'][-2]  # Previous suction head position
        prev_prev_act = self.buf['action'][-3][:-1]  # previous of previous predicted action
        prev_act = self.buf['action'][-2][:-1]  # previous predicted action
        current_act = self.buf['action'][-1][:-1]  # current predicted action
        current_joint = self.buf['joint'][-1]
        # Reward
        obj_po = np.linalg.norm(current_obj_down - self.env.target)  # Tracking ee-object position
        cur_dist = np.linalg.norm(current_ee - current_obj_up)   # Tracking object-target position
        cur_height = np.linalg.norm(current_ee[-1] - current_obj_up[-1])   # Tracking ee-object height position
        action_diff = np.linalg.norm(current_act - prev_act)**2  # Deviation of joints
        jerk = np.linalg.norm((current_act - prev_act) - (prev_act - prev_prev_act))**2  # Deviation of joints
        suc = bool(contact[-1][0] and contact[-1][1])  # Suction and contact state
        toward_goal = self._align_velocity(current_obj_up, prev_obj_up, self.env.target)  # instant velocity
        # toward_object = self._align_velocity(current_ee, prev_ee, current_obj_up)

        # Constraint
        joint_penalty, joint_violation = self._check_angle_constraints(current_joint)  # penalize joint collisions
        # outs, _ = self._outside_workspace(current_ee, termination=False)  # penalize moving outside workspace
        obj_outs1, outside1 = self._outside_workspace(current_obj_up)  # penalize tossing outside workspace
        obj_outs2, outside2 = (self._outside_workspace(self.buf['target_obj'][-1][:3])  # penalize outside workspace
                               if self.buf['target_obj'] else (0.0, False))
        incl = self._check_inclination()  # object inclination
        collision_penalty, collision_self, collision_ground, _, _ = self._check_collisions()  # collision
        ee_orientation_error = self._check_suction_orientation()  # penalize inclination suction
        # Compute total
        constraint = collision_penalty + joint_penalty + obj_outs1 + obj_outs2 + 0.32 * ee_orientation_error + incl + 0.05 * jerk

        # Phase 1: Pre-contact
        reward_1 = (1.3 if self.env.episode < self.env.curriculum_learning else 1.3/4) * np.exp(-cur_dist / self.sigma0)
        reward_1 += 0.3 * np.exp(-ee_orientation_error / 0.1)
        # reward_1 += 0.35 if cur_dist <= 0.02 else 0
        reward_1 += 0.15 * np.exp(-cur_height / 0.02)  # robot moves its end effector close to the object along z axis
        reward_1 += 0.1 * contact[-1][0]
        # reward_1 += 0.3 * np.exp((toward_object / self.sigma2) - 1) if toward_object > 0 else 0.0
        # Phase 2: Post-contact
        if suc:
            # If suction is activated, robot maintains object on its suction.
            reward_2 = 2.0 * np.exp(-obj_po / self.sigma1)
            reward_2 += 3.0 * np.exp(-obj_po / self.obj_threshold + 0.015)
            reward_3 = 0.80 * np.exp((toward_goal / self.sigma2) - 1) if toward_goal > 0 else 0.0

        total_reward = reward_1 + reward_2 + reward_3

        print(
            f'\ndistance: {cur_dist} - '
            f'object: {obj_po} - '
            f'action: {action_diff} - '
            f'suction: {suc} - '
            f'suction pose {ee_orientation_error}'
        )
        print(
            f'\nrew distance: {reward_1} - '
            f'rew object: {reward_2} - '
            f'rew suction: {suc} - '
            f'rew towards {reward_3} - '
            f'constrain {constraint} - '
            f'collision {collision_self or collision_ground} - '
            f'outside {outside1 or outside2} - '
            f'joint violation {joint_violation}'
        )
        total_reward = (total_reward - constraint)/self.scale_factor

        if self._is_subgoal_achieved():
            self.first_success_step += 1
            total_reward = (reward_1 + 3 * 3 - constraint)/self.scale_factor
            return total_reward, True if self.first_success_step == self.remain else False, True
            # keep running even after success to get more useful experiences
        self.first_success_step = 0

        return total_reward, outside1 or outside2 or joint_violation or collision_self or collision_ground, False
