import numpy as np
import pdb
from gym import utils
from . import mujoco_env
from . import geom_utils

class BaseAntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    # Initialize Mujoco environment
    def __init__(self, xml_file='my_ant.xml'):
        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)
        utils.EzPickle.__init__(self)

    # Forward step
    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    # State - 29 dimensional
    # This is possibly the only documentation that exists for mujoco ant. You're welcome
    # s[0:3]: (x, y, z)
    # s[3:7]: quaternion of the rotation of the ant from perspective of x+ being the front of the ant
    # s[7:9], s[9:11], s[11:13], s[13:15]: joint angles for front left leg, back left leg, back right leg and front right leg respectively
    # First variable is side to side angle (hip), second is up down angle (knee). In radians, but circles around
    # s[15:18]: (d_x, d_y, d_z)
    # s[18:21]: (d_roll, d_pitch, d_yaw)
    # s[21:28]: angle velocities - same order as before
    def get_state_by_name(self, name, s=None):
        # Get state (if not passed in)
        if s is None:
            s = self.state_vector()
        
        # Switch on name
        if name == 'xyz':
            val = s[0:3]
        elif name == 'x':
            val = s[0]
        elif name == 'y':
            val = s[1]
        elif name == 'z':
            val = s[2]
        elif name == 'quart':
            val = s[3:7]
        elif name in ['rpy', 'roll', 'pitch', 'yaw']:
            quart = s[3:7]
            roll, pitch, yaw = geom_utils.quaternion_to_euler_angle(quart)
            if name == 'roll':
                val = roll
            elif name == 'pitch':
                val = pitch
            elif name == 'yaw':
                val = yaw
            elif name == 'rpy':
                return np.array([roll, pitch, yaw])
        elif name == 'joint_angles':
            val = s[7:15]
        elif name == 'front_left_joints':
            val = s[7:9]
        elif name == 'front_left_hip':
            val = s[7]
        elif name == 'front_left_knee':
            val = s[8]
        elif name == 'back_left_joints':
            val = s[9:11]
        elif name == 'back_left_hip':
            val = s[9]
        elif name == 'back_left_knee':
            val = s[10]
        elif name == 'back_right_joints':
            val = s[11:13]
        elif name == 'back_right_hip':
            val = s[11]
        elif name == 'back_right_knee':
            val = s[12]
        elif name == 'front_right_joints':
            val = s[13:15]
        elif name == 'front_right_hip':
            val = s[13]
        elif name == 'front_right_knee':
            val = s[14]
        elif name == 'xyz_vel':
            val = s[15:18]
        elif name == 'x_vel':
            val = s[15]
        elif name == 'y_vel':
            val = s[16]
        elif name == 'z_vel':
            val = s[17]
        elif name == 'rpy_vel':
            val = s[18:21]
        elif name == 'roll_vel':
            val = s[18]
        elif name == 'pitch_vel':
            val = s[19]
        elif name == 'yaw_vel':
            val = s[20]
        elif name == 'joint_angle_vel':
            val = s[21:]
        elif name == 'front_left_joint_vel':
            val = s[21:23]
        elif name == 'front_left_hip_vel':
            val = s[21]
        elif name == 'front_left_knee_vel':
            val = s[22]
        elif name == 'back_left_joint_vel':
            val = s[23:25]
        elif name == 'back_left_hip_vel':
            val = s[23]
        elif name == 'back_left_knee_vel':
            val = s[24]
        elif name == 'back_right_joint_vel':
            val = s[25:27]
        elif name == 'back_right_hip_vel':
            val = s[25]
        elif name == 'back_right_knee_vel':
            val = s[26]
        elif name == 'front_right_joint_vel':
            val = s[27:29]
        elif name == 'front_right_hip_vel':
            val = s[27]
        elif name == 'front_right_knee_vel':
            val = s[28]
        else:
            raise Error("Not a recognized state")
        return val

    # We consider only roll, pitch and joint angles, and their velocities as "internal" state
    # We consider the rest as "external"
    # We convert away from quaternions to do this
    def get_intern_extern_state(self):
        # Extract different states
        s = self.state_vector()
        xyz = self.get_state_by_name('xyz', s)
        rpy = self.get_state_by_name('rpy', s)
        joint_angles = self.get_state_by_name('joint_angles', s)
        d_xyz = self.get_state_by_name('xyz_vel', s)
        d_rpy = self.get_state_by_name('rpy_vel', s)
        d_joint = self.get_state_by_name('joint_angle_vel', s)

        # Seperate out yaw
        roll = rpy[0]
        pitch = rpy[1]
        yaw = rpy[2]
        d_roll = d_rpy[0]
        d_pitch = d_rpy[1]
        d_yaw = d_rpy[2]

        # Set internal/external states
        s_internal = np.concatenate([[roll, pitch], joint_angles, [d_roll, d_pitch], d_joint])
        s_external = np.concatenate([xyz, [yaw], d_xyz, [d_yaw]])
        assert(s_internal.shape[0] == 20)
        assert(s_external.shape[0] == 8)

        return s_internal, s_external

    def _get_obs(self):
        raise NotImplementedError

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

class BaseAntLowGearEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    # Initialize Mujoco environment
    def __init__(self, xml_file='my_ant.xml'):
        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)
        utils.EzPickle.__init__(self)

    # Forward step
    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a/5).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    # State - 29 dimensional
    # This is possibly the only documentation that exists for mujoco ant. You're welcome
    # s[0:3]: (x, y, z)
    # s[3:7]: quaternion of the rotation of the ant from perspective of x+ being the front of the ant
    # s[7:9], s[9:11], s[11:13], s[13:15]: joint angles for front left leg, back left leg, back right leg and front right leg respectively
    # First variable is side to side angle (hip), second is up down angle (knee). In radians, but circles around
    # s[15:18]: (d_x, d_y, d_z)
    # s[18:21]: (d_roll, d_pitch, d_yaw)
    # s[21:28]: angle velocities - same order as before
    def get_state_by_name(self, name, s=None):
        # Get state (if not passed in)
        if s is None:
            s = self.state_vector()
        
        # Switch on name
        if name == 'xyz':
            val = s[0:3]
        elif name == 'x':
            val = s[0]
        elif name == 'y':
            val = s[1]
        elif name == 'z':
            val = s[2]
        elif name == 'quart':
            val = s[3:7]
        elif name in ['rpy', 'roll', 'pitch', 'yaw']:
            quart = s[3:7]
            roll, pitch, yaw = geom_utils.quaternion_to_euler_angle(quart)
            if name == 'roll':
                val = roll
            elif name == 'pitch':
                val = pitch
            elif name == 'yaw':
                val = yaw
            elif name == 'rpy':
                return np.array([roll, pitch, yaw])
        elif name == 'joint_angles':
            val = s[7:15]
        elif name == 'front_left_joints':
            val = s[7:9]
        elif name == 'front_left_hip':
            val = s[7]
        elif name == 'front_left_knee':
            val = s[8]
        elif name == 'back_left_joints':
            val = s[9:11]
        elif name == 'back_left_hip':
            val = s[9]
        elif name == 'back_left_knee':
            val = s[10]
        elif name == 'back_right_joints':
            val = s[11:13]
        elif name == 'back_right_hip':
            val = s[11]
        elif name == 'back_right_knee':
            val = s[12]
        elif name == 'front_right_joints':
            val = s[13:15]
        elif name == 'front_right_hip':
            val = s[13]
        elif name == 'front_right_knee':
            val = s[14]
        elif name == 'xyz_vel':
            val = s[15:18]
        elif name == 'x_vel':
            val = s[15]
        elif name == 'y_vel':
            val = s[16]
        elif name == 'z_vel':
            val = s[17]
        elif name == 'rpy_vel':
            val = s[18:21]
        elif name == 'roll_vel':
            val = s[18]
        elif name == 'pitch_vel':
            val = s[19]
        elif name == 'yaw_vel':
            val = s[20]
        elif name == 'joint_angle_vel':
            val = s[21:]
        elif name == 'front_left_joint_vel':
            val = s[21:23]
        elif name == 'front_left_hip_vel':
            val = s[21]
        elif name == 'front_left_knee_vel':
            val = s[22]
        elif name == 'back_left_joint_vel':
            val = s[23:25]
        elif name == 'back_left_hip_vel':
            val = s[23]
        elif name == 'back_left_knee_vel':
            val = s[24]
        elif name == 'back_right_joint_vel':
            val = s[25:27]
        elif name == 'back_right_hip_vel':
            val = s[25]
        elif name == 'back_right_knee_vel':
            val = s[26]
        elif name == 'front_right_joint_vel':
            val = s[27:29]
        elif name == 'front_right_hip_vel':
            val = s[27]
        elif name == 'front_right_knee_vel':
            val = s[28]
        else:
            raise Error("Not a recognized state")
        return val

    # We consider only roll, pitch and joint angles, and their velocities as "internal" state
    # We consider the rest as "external"
    # We convert away from quaternions to do this
    def get_intern_extern_state(self):
        # Extract different states
        s = self.state_vector()
        xyz = self.get_state_by_name('xyz', s)
        rpy = self.get_state_by_name('rpy', s)
        joint_angles = self.get_state_by_name('joint_angles', s)
        d_xyz = self.get_state_by_name('xyz_vel', s)
        d_rpy = self.get_state_by_name('rpy_vel', s)
        d_joint = self.get_state_by_name('joint_angle_vel', s)

        # Seperate out yaw
        roll = rpy[0]
        pitch = rpy[1]
        yaw = rpy[2]
        d_roll = d_rpy[0]
        d_pitch = d_rpy[1]
        d_yaw = d_rpy[2]

        # Set internal/external states
        s_internal = np.concatenate([[roll, pitch], joint_angles, [d_roll, d_pitch], d_joint])
        s_external = np.concatenate([xyz, [yaw], d_xyz, [d_yaw]])
        assert(s_internal.shape[0] == 20)
        assert(s_external.shape[0] == 8)

        return s_internal, s_external

    def _get_obs(self):
        raise NotImplementedError

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

