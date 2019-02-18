import numpy as np
import numpy as np
from gym import utils
from . import mujoco_env
from . import geom_utils

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))

class BaseProprioceptiveHumanoidEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    # Initialize Mujoco environment
    def __init__(self, xml_file='humanoid.xml'):
        # Set start values for registration
        self.start_yaw = float('inf')
        self.start_z = float('inf')
        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)
        utils.EzPickle.__init__(self)

    # Forward step
    def step(self, a):
        pos_before = mass_center(self.model, self.sim)
        self.do_simulation(a, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)
        alive_bonus = 5.0
        data = self.sim.data
        lin_vel_cost = 0.25 * (pos_after - pos_before) / self.model.opt.timestep
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        return self._get_obs(), reward, done, dict(reward_linvel=lin_vel_cost, reward_quadctrl=-quad_ctrl_cost, reward_alive=alive_bonus, reward_impact=-quad_impact_cost)

    # Get states by name
    def get_state_by_name(self, name, s=None):
        # Get state (if not passed in)
        if s is None:
            s = self.state_vector()
       
        # Replace with mass center
        s[0:3] = mass_center(self.model, self.sim)
 
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
            val = s[7:24]
        elif name == 'xyz_vel':
            val = s[24:27]
        elif name == 'x_vel':
            val = s[24]
        elif name == 'y_vel':
            val = s[25]
        elif name == 'z_vel':
            val = s[26]
        elif name == 'rpy_vel':
            val = s[27:30]
        elif name == 'roll_vel':
            val = s[27]
        elif name == 'pitch_vel':
            val = s[28]
        elif name == 'yaw_vel':
            val = s[29]
        elif name == 'joint_angle_vel':
            val = s[30:] 
        return val   

    # We remove the first 5 values from state which should correspond to global orientation and position
    # https://github.com/openai/gym/wiki/Humanoid-V1 
    def get_intern_extern_state(self):
        # Extract different states
        s = self.state_vector()
        z = self.get_state_by_name('z', s)
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
        # Internal keeps track of integral z and yaw (subtract out the initial value)
        pro_yaw = geom_utils.convert_to_egocentric(self.start_yaw, yaw)
        pro_z = z - self.start_z
        s_internal = np.concatenate([[pro_z, roll, pitch, pro_yaw], joint_angles, d_xyz, [d_roll, d_pitch, d_yaw], d_joint])
        s_external = np.concatenate([xyz, [yaw]])
        
        return s_internal, s_external

    def _get_obs(self):
        raise NotImplementedError

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )

        # Get the initial z and yaw, and keep track of to get integral values
        self.start_yaw = self.get_state_by_name('yaw')
        self.start_z = self.get_state_by_name('z')

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20
