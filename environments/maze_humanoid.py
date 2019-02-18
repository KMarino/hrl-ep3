# Kenneth Marino
# Lines taken from https://github.com/haarnoja/sac
from . import proprioceptive_humanoid_env
import numpy as np
import random
import pdb

# Hierarchy humanoid for the proprioceptive environment
# Obs is split into two
# Used for hierarchical methods
class HierarchyProprioceptiveHumanoidEnv(proprioceptive_humanoid_env.BaseProprioceptiveHumanoidEnv):
    # Initialize environment
    def __init__(self, xml_file='humanoid.xml'):
        super(HierarchyProprioceptiveHumanoidEnv, self).__init__(xml_file)

    # Get internal/propriaceptive state
    def _get_pro_obs(self):
        s_pro, _ = self.get_intern_extern_state()
        return s_pro

    # Get external state
    def _get_ext_obs(self):
        _, s_ext = self.get_intern_extern_state()
        return s_ext

    # Returns internal state as first part and external state concatted with the cfrc as external
    def _get_obs(self):
        return np.concatenate([self._get_pro_obs(), self._get_ext_obs()])

    # Return masks telling which are internal and external states
    def _get_pro_ext_mask(self):
        s_internal = self._get_pro_obs()
        s_external = self._get_ext_obs()
        pro_mask = np.concatenate([np.ones(s_internal.shape, dtype='bool'), np.zeros(s_external.shape, dtype='bool')])
        ext_mask = np.concatenate([np.zeros(s_internal.shape, dtype='bool'), np.ones(s_external.shape, dtype='bool')])
        return pro_mask, ext_mask

# Proprioceptive Humanoid maze environment
class ProprioceptiveHumanoidMazeEnv(HierarchyProprioceptiveHumanoidEnv):
    # Init
    def __init__(self, maze_opt):
        xml_file = maze_opt['xml_file']        
        self.possible_goal_positions = maze_opt['possible_goal_positions']

        # Define these early, but only to let mujoco_env get correct sizes
        self.goal_ind = random.randint(0, len(self.possible_goal_positions)-1)
        self.goal_position = self.possible_goal_positions[self.goal_ind]
        self.goal_radius = 0
        self.goal_reward = 0
        self.velocity_reward_weight = 0
        self.use_ctrl_cost = False
        self.use_contact_cost = False
        self.use_survive_reward = False
        self.use_negative_goals = False
        self.negative_goal_weight = 0

        super(ProprioceptiveHumanoidMazeEnv, self).__init__(xml_file)

    # Init options
    def option_init(self, maze_opt):
        # Extract relevant values
        self.goal_radius = maze_opt['goal_radius']
        self.goal_reward = maze_opt['goal_reward']
        self.velocity_reward_weight = maze_opt['velocity_reward_weight']
        self.use_ctrl_cost = maze_opt['use_ctrl_cost']
        self.use_contact_cost = maze_opt['use_contact_cost']
        self.use_survive_reward = maze_opt['use_survive_reward']
        self.use_negative_goals = maze_opt['use_negative_goals']
        self.negative_goal_weight = maze_opt['negative_goal_reward']
         
    # Get goal info
    def _get_goal_obs(self):
        reward_obs = np.zeros(len(self.possible_goal_positions))
        reward_obs[self.goal_ind] = 1
        return reward_obs

    # Get internal/propriaceptive state
    def _get_pro_obs(self):
        s_pro = super(ProprioceptiveHumanoidMazeEnv, self)._get_pro_obs()
        return s_pro

    # Get external state
    # Includes goal obs at the high level
    def _get_ext_obs(self): 
        s_ext = super(ProprioceptiveHumanoidMazeEnv, self)._get_ext_obs()
        goal_obs = self._get_goal_obs()
        s_ext = np.concatenate([s_ext, goal_obs])
        return s_ext
 
    # Get obs mask for internal
    def _get_pro_obs_mask(self):
        s_int = super(ProprioceptiveHumanoidMazeEnv, self)._get_pro_obs()
        s_int_mask = np.zeros(s_int.shape)
        return s_int_mask

    # Get obs mask for external
    def _get_ext_obs_mask(self):
        s_ext = super(ProprioceptiveHumanoidMazeEnv, self)._get_ext_obs()
        goal_obs = self._get_goal_obs()
        s_ext_mask = np.concatenate([np.zeros(s_ext.shape), np.ones(goal_obs.shape)])
        return s_ext_mask

    # Observation includes the goal obs at the high level
    def _get_obs(self):
        # Get internal and external obs
        s_int = self._get_pro_obs()
        s_ext = self._get_ext_obs()
        return np.concatenate([s_int, s_ext])

    # Get observation mask (1 at values we shouldn't be doing running averaging and such)
    def _get_obs_mask(self):
        s_int_mask = self._get_pro_obs_mask()
        s_ext_mask = self._get_ext_obs_mask()
        return np.concatenate([s_int_mask, s_ext_mask])

    # Reset - change the goal location from among possible locations
    def reset(self):
        # Randomly choose from possible goal loactions
        self.goal_ind = random.randint(0, len(self.possible_goal_positions)-1)
        self.goal_position = self.possible_goal_positions[self.goal_ind]

        # Call super reset method
        return super(ProprioceptiveHumanoidMazeEnv, self).reset()

    # Step
    def step(self, action):
        # Call super step function
        obs, reward, done, info = super(ProprioceptiveHumanoidMazeEnv, self).step(action)

        # Get xy position and distance to goal
        xy_position = self.get_body_com('torso')[:2]
        goal_distance = np.linalg.norm(xy_position - self.goal_position)
        goal_distance_radius = max(0, goal_distance - self.goal_radius)
        goal_reached = goal_distance < self.goal_radius
        goal_reward = int(goal_reached) * self.goal_reward
        if self.use_negative_goals:
            for neg_goal_ind in range(0, len(self.possible_goal_positions)):
                if self.goal_ind != neg_goal_ind:
                    # Check if near other goals, not the correct one
                    neg_goal_position = self.possible_goal_positions[neg_goal_ind]
                    neg_goal_distance = np.linalg.norm(xy_position - neg_goal_position)
                    neg_goal_reached = neg_goal_distance < self.goal_radius
            
                    # Give negative reward for reaching the negative goal
                    goal_reward = int(neg_goal_reached) * self.negative_goal_reward

                    # Negative goal is an absorbing state as well
                    goal_reached |= neg_goal_reached
       
        # Get velocity weight
        if self.velocity_reward_weight > 0:
            xy_velocities = self.data.qvel[:2]
            # rewards for speed on xy-plane (no matter which direction)
            velocity_reward = self.velocity_reward_weight * np.linalg.norm(xy_velocities)
        else:
            velocity_reward = 0
    
        # Rename some costs
        info['reward_survive'] = info['reward_alive']
        info['reward_ctrl'] = info['reward_quadctrl']
        info['reward_contact'] = info['reward_impact']
        info['reward_forward'] = info['reward_linvel']

        # Get control cost 
        if self.use_ctrl_cost:
            ctrl_reward = info['reward_ctrl']
        else:
            ctrl_reward = 0

        # Get contact cost
        if self.use_contact_cost:
            contact_reward = info['reward_contact']
        else:
            contact_reward = 0

        # Get survive reward
        if self.use_survive_reward:
            survive_reward = info['reward_survive']
        else:
            survive_reward = 0        

        # Compute final reward
        reward = goal_reward + velocity_reward + survive_reward + ctrl_reward + contact_reward
        
        # Compute whether we're done (dead or at goal)
        done = done or goal_reached

        # Update info
        info['goal_distance'] = goal_distance
        info['goal_distance_radius'] = goal_distance_radius
        info['goal_position'] = self.goal_position
        info['reward_velocity'] = velocity_reward
        info['reward_goal'] = goal_reward
        
        # Return
        return obs, reward, done, info

# Smaller cross humanoid maze environment
class ProprioceptiveHumanoidSmallCrossMazeEnv(ProprioceptiveHumanoidMazeEnv):
    # Init
    def __init__(self):
        maze_opt = {}

        # Add Cross maze goals
        maze_opt['possible_goal_positions'] = [[3, 3], [6, 0], [3, -3]]
 
        # Add cross maze xml
        maze_opt['xml_file'] = 'small_cross_maze_humanoid.xml' 

        # Set radius to 1
        maze_opt['goal_radius'] = 2
        
        # Call super class
        super(ProprioceptiveHumanoidSmallCrossMazeEnv, self).__init__(maze_opt)

    def option_init(self, maze_opt):
        super(ProprioceptiveHumanoidSmallCrossMazeEnv, self).option_init(maze_opt)  


