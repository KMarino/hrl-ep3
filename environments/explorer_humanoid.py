from . import proprioceptive_humanoid_env
import numpy as np

# All obs but xy but yaw and z use integrals
class LowlevelProprioceptiveHumanoidEnv(proprioceptive_humanoid_env.BaseProprioceptiveHumanoidEnv):
    # Initialize environment
    def __init__(self):
        super(LowlevelProprioceptiveHumanoidEnv, self).__init__()
         
    def _get_obs(self):
        s_internal, _ = self.get_intern_extern_state()
        return s_internal
