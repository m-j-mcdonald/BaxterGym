from gym.envs.registration import register

register(
    id='BaxterContinuousClothFold-v0',
    entry_point='baxter_gym.envs:BaxterContinousClothEnv',
)

register(
    id='BaxterDiscreteClothFold-v0',
    entry_point='baxter_gym.envs:BaxterDiscreteClothEnv',
)
