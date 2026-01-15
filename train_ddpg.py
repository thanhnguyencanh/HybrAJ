import numpy as np
from datetime import datetime
import gym
import env
import argparse
from torch.utils.tensorboard import SummaryWriter
import shutil
import os
import time
from utils.relay_buffer import ReplayBuffer
from models.ddpg import Agent
from utils.normalizer import RunningStatNormalizer
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default="ddpg")  # Policy name
    parser.add_argument("--env_name", default="ImitationLearning-v1")  # environment name
    parser.add_argument("--warmup", default=20000, type=int)
    parser.add_argument("--action", default=0, type=int, required=True)  # training action
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--log_path", default='reward_log', type=str)  # reward log path
    parser.add_argument("--checkpoint", default='ckpt', type=str)  # reward log path
    parser.add_argument("--max_episode", default=1e6, type=float)  # Max episode to run environment for
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic (recommend 128)
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--max_steps", default=100, type=int)  # max steps per episode
    parser.add_argument("--tau", default=0.001, type=float)  # Target network update rate
    args = parser.parse_args()

    file_name = "%s_%s_%s" % (args.policy_name, args.env_name, str(args.seed))
    print("---------------------------------------")
    print("Settings: %s" % (file_name))
    print("---------------------------------------")

    # Initialize saving file
    base = f'{args.policy_name}_{str(args.action)}'
    for path in [args.log_path]:
        path = os.path.join(base, path)
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(path)

    writer = SummaryWriter(os.path.join(base, args.log_path))
    env = gym.make(args.env_name)
    env.unwrapped.max_steps = args.max_steps  # define max steps
    env.unwrapped.max_episode = args.max_episode  # define max steps
    # Set seeds
    normalizer = RunningStatNormalizer(shape=(12,))
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    discrete_action = env.discrete_action.n

    # Initialize policy
    policy = Agent(state_dim=state_dim, action_dim=action_dim, max_action=max_action, p_action_dim=discrete_action,
                   discount=0.99, actor_lr=1e-4, critic_lr=1e-3, tau=args.tau, writer=writer,
                   chkpt_dir=os.path.join(base, args.checkpoint), epsilon=1.0, eps_min=0.05, eps_dec=1e-5)
    replay_buffer = ReplayBuffer()

    best_reward= float('-inf')
    env.unwrapped.action_type = args.action
    env.unwrapped.writer = writer
    env.unwrapped.curriculum_learning = 50000
    curr_episode = 0
    episode_reward = 0
    reward_history = []
    avg_reward_history = []
    success = []
    t0 = time.time()
    '''TRAINING PROCESS'''
    while curr_episode < args.max_episode:
        total_reward = 0
        n_steps = 0
        state = env.reset()
        done = False

        while not done:
            # Use policy with exploration noise
            action = policy.choose_action(state)
            if replay_buffer.ready(batch_size=args.batch_size):
                policy.learn(replay_buffer, args.batch_size)
            next_state, reward, done, info = env.step(action)
            total_reward += reward  # total step reward
            replay_buffer.add((state, next_state, action, reward, done))
            state = next_state
            n_steps += 1

            elapsed = int(time.time() - t0)
            print('------------------------------------------------------')
            print(f"Episode: {curr_episode} Step: {n_steps} "
                  f"Reward: {reward:.5f}  --  Wallclk T: {elapsed // 86400}d {(elapsed % 86400) // 3600}h {(elapsed % 3600) // 60}m {elapsed % 60}s")
            print('------------------------------------------------------')

        reward_history.append(total_reward)
        avg_reward_history.append(total_reward / n_steps)
        success.append(any(info['log']))

        if best_reward <= total_reward / n_steps and success[-1]:
            best_reward = total_reward / n_steps
            policy.save()

        episode_reward += total_reward
        if len(reward_history) >= 100:
            moving_avg = np.mean(reward_history[-100:])
            moving_avg_avg = np.mean(avg_reward_history[-100:])
            writer.add_scalar("Reward/moving_avg_return", moving_avg, curr_episode)
            writer.add_scalar("Reward/moving_avg_of_avg_reward", moving_avg_avg, curr_episode)
        writer.add_scalar('Reward/avg_total_reward_per_episode', total_reward / n_steps, curr_episode)
        writer.add_scalar('Reward/episode reward', episode_reward, curr_episode)
        writer.add_scalar(f"Reward/success rate", sum(success[-100:]) / 100, curr_episode)
        elapsed = int(time.time() - t0)
        print('------------------------------------------------------')
        print(
            f"Episode: {curr_episode} Total reward: {total_reward:.5f}  --  Wallclk T: {elapsed // 86400}d {(elapsed % 86400) // 3600}h {(elapsed % 3600) // 60}m {elapsed % 60}s")
        print('------------------------------------------------------')
        curr_episode += 1
