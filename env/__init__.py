from gym.envs.registration import register

register(
    id='ImitationLearning-v1',
    entry_point='env.robot_env:ImitationLearning',
    max_episode_steps=300,
)