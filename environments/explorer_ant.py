from . import ant_env
import numpy as np

# Only contains the internal state of the ant in the observation
class LowlevelAntEnv(ant_env.BaseAntEnv):
    # Initialize environment
    def __init__(self):
        super(LowlevelAntEnv, self).__init__()
         
    def _get_obs(self):
        s_internal, _ = self.get_intern_extern_state()
        return s_internal

# Only contains the internal state of the ant in the observation
class LowlevelAntLowGearEnv(ant_env.BaseAntLowGearEnv):
    # Initialize environment
    def __init__(self):
        super(LowlevelAntLowGearEnv, self).__init__(xml_file='ant_custom_gear.xml')
         
    def _get_obs(self):
        s_internal, _ = self.get_intern_extern_state()
        return s_internal

