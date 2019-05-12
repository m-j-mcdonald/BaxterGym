from gym.envs.registration import register

register(
    id='BaxterContinuousClothFold-v0',
    entry_point='baxter_gym.envs:BaxterContinousClothEnv',
    max_episode_steps=250,
    reward_threshold=2500,
)

register(
    id='BaxterDiscreteClothFold-v0',
    entry_point='baxter_gym.envs:BaxterDiscreteClothEnv',
    max_episode_steps=250,
    reward_threshold=2500,
)

register(
    id='BaxterBlockStack-v0',
    entry_point='baxter_gym.envs:BaxterBlockStackEnv',
    max_episode_steps=20,
)

register(
    id='BaxterLeftBlockStack-v0',
    entry_point='baxter_gym.envs:BaxterLeftBlockStackEnv',
    max_episode_steps=20,
)
