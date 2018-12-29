from baxter_gym.envs.baxter_mjc_env import *


class BaxterContinuousClothEnv(BaxterMJCEnv):
    def __init__(self):
        cloth_info = {'width': 5, 'length': 3, 'spacing': 0.1, 'radius': 0.01}
        cloth = get_deformable_cloth(cloth_info['width'], 
                                     cloth_info['length'], 
                                     cloth_info['spacing'], 
                                     cloth_info['radius'],
                                     (0.5, -0.2, 0.0))
        super(BaxterContinuousClothEnv, self).__init__(mode='end_effector_pos', items=[cloth], cloth_info=cloth_info, view=True)


class BaxterDiscreteClothEnv(BaxterMJCEnv):
    def __init__(self):
        cloth_info = {'width': 5, 'length': 3, 'spacing': 0.1, 'radius': 0.01}
        cloth = get_deformable_cloth(cloth_info['width'], 
                                     cloth_info['length'], 
                                     cloth_info['spacing'], 
                                     cloth_info['radius'],
                                     (0.5, -0.2, 0.0))
        super(BaxterDiscreteClothEnv, self).__init__(mode='discrete_pos', items=[cloth], cloth_info=cloth_info, view=True)
