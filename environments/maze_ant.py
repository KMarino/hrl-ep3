# Kenneth Marino
# Lines taken from https://github.com/haarnoja/sac
from . import ant_env
import numpy as np
import random
import pdb

# Obs is split into two - internal and external
# Used for hierarchical methods
class HierarchyAntEnv(ant_env.BaseAntEnv):
    # Initialize environment
    def __init__(self, xml_file='my_ant.xml'):
        super(HierarchyAntEnv, self).__init__(xml_file)

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

# Obs is split into two - internal and external
# Used for hierarchical methods
class HierarchyAntLowGearEnv(ant_env.BaseAntLowGearEnv):
    # Initialize environment
    def __init__(self):
        super(HierarchyAntLowGearEnv, self).__init__(xml_file='ant_custom_gear.xml')

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

# Ant maze environment
class AntMazeEnv(HierarchyAntEnv):
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

        super(AntMazeEnv, self).__init__(xml_file)

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
        s_pro = super(AntMazeEnv, self)._get_pro_obs()
        return s_pro

    # Get external state
    # Includes goal obs at the high level
    def _get_ext_obs(self): 
        s_ext = super(AntMazeEnv, self)._get_ext_obs()
        goal_obs = self._get_goal_obs()
        s_ext = np.concatenate([s_ext, goal_obs])
        return s_ext
 
    # Get obs mask for internal
    def _get_pro_obs_mask(self):
        s_int = super(AntMazeEnv, self)._get_pro_obs()
        s_int_mask = np.zeros(s_int.shape)
        return s_int_mask

    # Get obs mask for external
    def _get_ext_obs_mask(self):
        s_ext = super(AntMazeEnv, self)._get_ext_obs()
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
        return super(AntMazeEnv, self).reset()

    # Step
    def step(self, action):
        # Call super step function
        obs, reward, done, info = super(AntMazeEnv, self).step(action)

        # Get xy position and distance to goal
        xy_position = self.get_body_com('torso')[:2]
        goal_distance = np.linalg.norm(xy_position - self.goal_position)
        goal_distance_radius = max(0, goal_distance - self.goal_radius)
        goal_reached = goal_distance < self.goal_radius
        goal_reward = int(goal_reached) * self.goal_reward

        # Use other possible goals as negative rewards
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

# Ant navigate environment
class AntNavigateEnv(HierarchyAntEnv):
    # Init
    def __init__(self):
        # Init values
        self.goals = [(2, 2), (2, -2), (-2, -2), (-2, 2), (2, 2)] 
        self.goal_radius = 0.5
        self.reward_waypoint = 1

        # Init things for registration
        self.goal_ind = 0
        self.goal_position = self.goals[self.goal_ind]
        self.goal_reward = 0
        super(AntNavigateEnv, self).__init__() #'ant_custom_gear.xml')
        
    # Get goal info
    def _get_goal_obs(self):
        reward_obs = np.zeros(len(self.goals))
        reward_obs[self.goal_ind] = 1
        return reward_obs
    
    # Dummy
    def option_init(self, opt):
        pass

    # Get internal/propriaceptive state
    def _get_pro_obs(self):
        s_pro = super(AntNavigateEnv, self)._get_pro_obs()
        return s_pro

    # Get external state
    # Includes goal obs at the high level
    def _get_ext_obs(self): 
        s_ext = super(AntNavigateEnv, self)._get_ext_obs()
        goal_obs = self._get_goal_obs()
        s_ext = np.concatenate([s_ext, goal_obs])
        return s_ext
 
    # Get obs mask for internal
    def _get_pro_obs_mask(self):
        s_int = super(AntNavigateEnv, self)._get_pro_obs()
        s_int_mask = np.zeros(s_int.shape)
        return s_int_mask

    # Get obs mask for external
    def _get_ext_obs_mask(self):
        s_ext = super(AntNavigateEnv, self)._get_ext_obs()
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
        self.goal_ind = 0
        self.goal_position = self.goals[self.goal_ind]  

        # Call super reset method
        return super(AntNavigateEnv, self).reset()

    # Step
    def step(self, action):
        # Call super step function
        obs, reward, done, info = super(AntNavigateEnv, self).step(action)

        # Get xy position and distance to goal
        xy_position = self.get_body_com('torso')[:2]
        goal_distance = np.linalg.norm(xy_position - self.goal_position)
        goal_distance_radius = max(0, goal_distance - self.goal_radius)
        goal_reached = goal_distance < self.goal_radius
        reward = int(goal_reached) * self.reward_waypoint 

        # Update goal_ind
        done = False
        if goal_reached:
            self.goal_ind += 1
            if self.goal_ind == len(self.goals):
                done = True
            else:
                self.goal_position = self.goals[self.goal_ind] 

        # Update info
        info['goal_distance'] = goal_distance
        info['goal_distance_radius'] = goal_distance_radius
        info['goal_position'] = self.goal_position
        info['reward_velocity'] = 0
        info['reward_goal'] = reward
        
        # Return
        return obs, reward, done, info

# Ant navigate environment
class AntNavigateLowGearEnv(HierarchyAntLowGearEnv):
    # Init
    def __init__(self):
        # Init values
        self.goals = [(2, 2), (2, -2), (-2, -2), (-2, 2), (2, 2)] 
        self.goal_radius = 0.5
        self.reward_waypoint = 1

        # Init things for registration
        self.goal_ind = 0
        self.goal_position = self.goals[self.goal_ind]
        self.goal_reward = 0
        super(AntNavigateLowGearEnv, self).__init__() #'ant_custom_gear.xml')
        
    # Get goal info
    def _get_goal_obs(self):
        reward_obs = np.zeros(len(self.goals))
        reward_obs[self.goal_ind] = 1
        return reward_obs
    
    # Dummy
    def option_init(self, opt):
        pass

    # Get internal/propriaceptive state
    def _get_pro_obs(self):
        s_pro = super(AntNavigateLowGearEnv, self)._get_pro_obs()
        return s_pro

    # Get external state
    # Includes goal obs at the high level
    def _get_ext_obs(self): 
        s_ext = super(AntNavigateLowGearEnv, self)._get_ext_obs()
        goal_obs = self._get_goal_obs()
        s_ext = np.concatenate([s_ext, goal_obs])
        return s_ext
 
    # Get obs mask for internal
    def _get_pro_obs_mask(self):
        s_int = super(AntNavigateLowGearEnv, self)._get_pro_obs()
        s_int_mask = np.zeros(s_int.shape)
        return s_int_mask

    # Get obs mask for external
    def _get_ext_obs_mask(self):
        s_ext = super(AntNavigateLowGearEnv, self)._get_ext_obs()
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
        self.goal_ind = 0
        self.goal_position = self.goals[self.goal_ind]  

        # Call super reset method
        return super(AntNavigateLowGearEnv, self).reset()

    # Step
    def step(self, action):
        # Call super step function
        obs, reward, done, info = super(AntNavigateLowGearEnv, self).step(action)

        # Get xy position and distance to goal
        xy_position = self.get_body_com('torso')[:2]
        goal_distance = np.linalg.norm(xy_position - self.goal_position)
        goal_distance_radius = max(0, goal_distance - self.goal_radius)
        goal_reached = goal_distance < self.goal_radius
        reward = int(goal_reached) * self.reward_waypoint 

        # Update goal_ind
        done = False
        if goal_reached:
            self.goal_ind += 1
            if self.goal_ind == len(self.goals):
                done = True
            else:
                self.goal_position = self.goals[self.goal_ind] 

        # Update info
        info['goal_distance'] = goal_distance
        info['goal_distance_radius'] = goal_distance_radius
        info['goal_position'] = self.goal_position
        info['reward_velocity'] = 0
        info['reward_goal'] = reward
        
        # Return
        return obs, reward, done, info

# Cross ant maze environment
class AntCrossMazeEnv(AntMazeEnv):
    # Init
    def __init__(self):
        maze_opt = {}

        # Add Cross maze goals
        maze_opt['possible_goal_positions'] = [[6, 6], [12, 0], [6, -6]]
 
        # Add cross maze xml
        maze_opt['xml_file'] = 'cross_maze_ant.xml' 
        
        # Call super class
        super(AntCrossMazeEnv, self).__init__(maze_opt)

    def option_init(self, maze_opt):
        super(AntCrossMazeEnv, self).option_init(maze_opt)  

# Cross key maze ant environment
class AntCrossKeyMazeEnv(AntKeyMazeEnv):
    # Init
    def __init__(self):
        maze_opt = {}

        # Add Cross maze goals
        maze_opt['possible_goal_positions'] = [[6, 6], [12, 0], [6, -6]]
 
        # Add cross maze xml
        maze_opt['xml_file'] = 'cross_maze_ant.xml' 
        
        # Call super class
        super(AntCrossKeyMazeEnv, self).__init__(maze_opt)

    def option_init(self, maze_opt):
        super(AntCrossKeyMazeEnv, self).option_init(maze_opt)  

# Skull ant maze environment
class AntSkullMazeEnv(AntMazeEnv):
    # Init
    def __init__(self):
        maze_opt = {}

        # Add Cross maze goals
        maze_opt['possible_goal_positions'] = [[-8, 4], [8, 4], [-4, 8], [4, 8]]
 
        # Add cross maze xml
        maze_opt['xml_file'] = 'skull_maze_ant.xml' 
        
        # Call super class
        super(AntSkullMazeEnv, self).__init__(maze_opt)

    def option_init(self, maze_opt):
        super(AntSkullMazeEnv, self).option_init(maze_opt)  

# Debug hierarchical environment
# Actually same as cross maze env, but we only have goal straight ahead
# Should be dead simple to learn
class DebugAntMazeEnv(AntMazeEnv):
    # Init
    def __init__(self):
        maze_opt = {}

        # Add Cross maze goals
        maze_opt['possible_goal_positions'] = [[12, 0]]
 
        # Add cross maze xml
        maze_opt['xml_file'] = 'cross_maze_ant.xml' 
        
        # Call super class
        super(DebugAntMazeEnv, self).__init__(maze_opt)

    def option_init(self, maze_opt):
        super(DebugAntMazeEnv, self).option_init(maze_opt)  


# Debug hierarchical environment
# Actually same as cross maze env, but we only have goal straight ahead
# Should be dead simple to learn
class DebugAntMazeLeftEnv(AntMazeEnv):
    # Init
    def __init__(self):
        maze_opt = {}

        # Add Cross maze goals
        maze_opt['possible_goal_positions'] = [[6, 6]]
 
        # Add cross maze xml
        maze_opt['xml_file'] = 'cross_maze_ant.xml' 
        
        # Call super class
        super(DebugAntMazeLeftEnv, self).__init__(maze_opt)

    def option_init(self, maze_opt):
        super(DebugAntMazeLeftEnv, self).option_init(maze_opt)  


# Debug hierarchical environment
# Actually same as cross maze env, but we only have goal straight ahead
# Should be dead simple to learn
class DebugAntMazeRightEnv(AntMazeEnv):
    # Init
    def __init__(self):
        maze_opt = {}

        # Add Cross maze goals
        maze_opt['possible_goal_positions'] = [[6, -6]]
 
        # Add cross maze xml
        maze_opt['xml_file'] = 'cross_maze_ant.xml' 
        
        # Call super class
        super(DebugAntMazeRightEnv, self).__init__(maze_opt)

    def option_init(self, maze_opt):
        super(DebugAntMazeRightEnv, self).option_init(maze_opt)  
