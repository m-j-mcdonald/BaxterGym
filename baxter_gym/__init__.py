from gym.envs.registration import register

register(
    id='baxter_cloth_fold-v0',
    entry_point='baxter_gym.envs:BaxterContinousClothEnv',
)

register(
    id='baxter_discrete_cloth_fold-v0',
    entry_point='baxter_gym.envs:BaxterDiscreteClothEnv',
)
