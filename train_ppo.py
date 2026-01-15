import numpy as np
from datetime import datetime
import env
import gym
import argparse
from torch.utils.tensorboard import SummaryWriter
import shutil
import os
import time
import random
from models.ppo import Agent
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default="ppo")  # Policy name
    parser.add_argument("--env_name", default="ImitationLearning-v1")  # environment name
    parser.add_argument("--action", default=0, type=int, required=True)  # training action
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--log_path", default='reward_log', type=str)  # reward log path
    parser.add_argument("--checkpoint", default='ckpt', type=str)  # checkpoint log path
    parser.add_argument("--max_episode", default=1e5, type=float)  # Max episode to run environment for
    parser.add_argument("--batch_size", default=128, type=int)  # Batch size for both actor and critic (recommend 128)
    parser.add_argument("--n_epochs", default=2, type=int)
    parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
    parser.add_argument('--lambd', type=float, default=0.95, help='GAE Factor')
    parser.add_argument('--policy_clip', type=float, default=0.2, help='PPO Clip rate')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate of actor')
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
    env.unwrapped.max_steps = args.batch_size  # define max steps
    best_reward = float('-inf')
    state_dim = env.observation_space.shape[0]  # 535
    action_dim = env.action_space.shape[0]  # 7 joints
    max_action = float(env.action_space.high[0])
    discrete_action = env.discrete_action.n

    # Initialize policy
    policy = Agent(action_dims=action_dim, input_dims=state_dim, p_action_dim=discrete_action, gamma=args.gamma,
                   alpha=args.lr, gae_lambda=args.lambd, policy_clip=args.policy_clip, batch_size=int(args.batch_size/2),
                   n_epochs=args.n_epochs, chkpt=os.path.join(base, args.checkpoint))
    curr_episode = 0
    env.unwrapped.action_type = args.action
    env.unwrapped.writer = writer
    env.unwrapped.curriculum_learning = 50000
    episode_reward = 0
    reward_history = []
    avg_reward_history = []
    success = []
    t0 = time.time()

    while curr_episode < args.max_episode:
        total_reward = 0
        n_steps = 0
        state = env.reset()
        done = False

        for _ in range(args.batch_size):
            action, logprob, value, logp_c, logp_d = policy.choose_action(state)
            next_state, reward, done, info = env.step([action[:-1], [action[-1]]])
            total_reward += reward
            policy.add(state, action, logp_c, logp_d, value, reward, done)  # store V at rollout time instead of re-compute V in the future
            state = next_state
            elapsed = int(time.time() - t0)
            print('------------------------------------------------------')
            print(f"Episode: {curr_episode} Step: {n_steps} "
                  f"Reward: {reward:.5f}  --  Wallclk T: {elapsed // 86400}d {(elapsed % 86400) // 3600}h {(elapsed % 3600) // 60}m {elapsed % 60}s")
            print('------------------------------------------------------')
            if done and n_steps < args.batch_size-1:
                state = env.reset()
            n_steps += 1

        reward_history.append(total_reward)
        avg_reward_history.append(total_reward / n_steps)
        policy.learn()
        success.append(any(info['log']))

        if best_reward <= total_reward / n_steps and success[-1]:
            best_reward = total_reward / n_steps
            policy.save_models()

        episode_reward += total_reward
        if len(reward_history) >= 100:
            moving_avg = np.mean(reward_history[-100:])
            moving_avg_avg = np.mean(avg_reward_history[-100:])
            writer.add_scalar("Reward/moving_avg_return", moving_avg, curr_episode)
            writer.add_scalar("Reward/moving_avg_of_avg_reward", moving_avg_avg, curr_episode)
        writer.add_scalar('Reward/avg_total_reward_per_episode', total_reward/n_steps, curr_episode)
        writer.add_scalar('Reward/episode reward', episode_reward, curr_episode)
        writer.add_scalar(f"Reward/success rate", sum(success[-100:]) / 100, curr_episode)
        print('------------------------------------------------------')
        print(f"Episode: {curr_episode} Total reward: {total_reward:.5f}  --  Wallclk T: {int(time.time() - t0)} sec")
        print('------------------------------------------------------')
        curr_episode += 1