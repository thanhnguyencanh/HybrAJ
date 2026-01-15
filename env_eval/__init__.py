from gym.envs.registration import register

register(
    id='ImitationLearning-v1',
    entry_point='env_eval.robot_env:ImitationLearning',
    max_episode_steps=300,
)