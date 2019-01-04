from baxter_gym.envs.baxter_mjc_env import *
from gym import spaces

class BaxterContinuousClothEnv(BaxterMJCEnv):
    def __init__(self):
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(8,))
        cloth_info = {'width': 5, 'length': 3, 'spacing': 0.1, 'radius': 0.01}
        cloth = get_deformable_cloth(cloth_info['width'], 
                                     cloth_info['length'], 
                                     cloth_info['spacing'], 
                                     cloth_info['radius'],
                                     (0.5, -0.2, 0.0))
        obs_include = set(['end_effector', 'overhead_image', 'cloth_points'])
        super(BaxterContinuousClothEnv, self).__init__(mode='end_effector_pos', 
                                                       items=[cloth], 
                                                       cloth_info=cloth_info, 
                                                       obs_include=obs_include,
                                                       view=True)

    def reset():
        self.randomize_cloth()
        super(BaxterContinuousClothEnv, self).reset()



class BaxterDiscreteClothEnv(BaxterMJCEnv):
    def __init__(self):
        self.action_space = spaces.Discrete(16)
        cloth_info = {'width': 5, 'length': 3, 'spacing': 0.1, 'radius': 0.01}
        cloth = get_deformable_cloth(cloth_info['width'], 
                                     cloth_info['length'], 
                                     cloth_info['spacing'], 
                                     cloth_info['radius'],
                                     (0.5, -0.2, 0.0))
        obs_include = set(['end_effector', 'overhead_image', 'cloth_points'])
        super(BaxterDiscreteClothEnv, self).__init__(mode='discrete_pos', 
                                                     items=[cloth], 
                                                     cloth_info=cloth_info, 
                                                     obs_include=obs_include, 
                                                     view=True)

    def reset():
        self.randomize_cloth()
        super(BaxterDiscreteClothEnv, self).reset()
