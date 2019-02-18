import numpy as np
import numpy as np
from gym import utils
import pdb
import math
from . import mujoco_env
from . import geom_utils

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))

class BaseSimpleHumanoidEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    # Initialize Mujoco environment
    def __init__(self, xml_file='simple_humanoid.xml'):
        mujoco_env.MujocoEnv.__init__(self, xml_file, 1)
        utils.EzPickle.__init__(self)

    # Forward step
    def step(self, action):
        pos_before = mass_center(self.model, self.sim)
        self.do_simulation(action, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)
        alive_bonus = 0.2
        data = self.sim.data
        lin_vel_cost = 0.25 * (pos_after - pos_before) / self.model.opt.timestep
        lb = -100
        ub = 100
        scaling = (ub - lb) * 0.5
        quad_ctrl_cost = .5 * 1e-3 * np.sum(
                np.square(action / scaling))
        quad_impact_cost = .5 * 1e-5 * np.sum(
            np.square(np.clip(data.cfrc_ext, -1, 1)))
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos
        done = bool((qpos[2] < 0.8) or (qpos[2] > 2.0))
        return self._get_obs(), reward, done, dict(reward_linvel=lin_vel_cost, reward_quadctrl=-quad_ctrl_cost, reward_alive=alive_bonus, reward_impact=-quad_impact_cost)

    # Get states by name
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
            val = s[7:17]
        elif name == 'xyz_vel':
            val = s[17:20]
        elif name == 'x_vel':
            val = s[17]
        elif name == 'y_vel':
            val = s[18]
        elif name == 'z_vel':
            val = s[19]
        elif name == 'rpy_vel':
            val = s[20:23]
        elif name == 'roll_vel':
            val = s[20]
        elif name == 'pitch_vel':
            val = s[21]
        elif name == 'yaw_vel':
            val = s[22]
        elif name == 'joint_angle_vel':
            val = s[23:] 
        return val   

    # We remove the first 5 values from state which should correspond to global orientation and position
    # https://github.com/openai/gym/wiki/Humanoid-V1 
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
        s_internal = np.concatenate([[roll, pitch], joint_angles, [d_roll, d_pitch]])
        s_external = np.concatenate([xyz, [yaw], d_xyz, [d_yaw]])
        #s_internal = np.concatenate([s, np.clip(self.sim.data.cfrc_ext, -1, 1).flat, self.get_body_com("torso").flat] )

        #assert(s_internal.shape[0] == 20)
        assert(s_external.shape[0] == 8)
        
        return s_internal, s_external

    def _get_obs(self):
        raise NotImplementedError

    def get_body_com(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.sim.data.subtree_com[idx]

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20
