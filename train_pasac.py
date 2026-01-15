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
from related_works.pasac import Agent
import warnings
warnings.filterwarnings("ignore")

def eval_policy(policy, env, eval_episodes=15):
    success_rate = 0
    for _ in range(eval_episodes):
        state, done = env.reset(), False
        while not done:
            action = policy.choose_action(np.array(state), evaluate=True)
            state, reward, done, info = env.step(action)
            # avg_reward += reward
        success_rate += any(info["log"])

    success_rate /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {success_rate:.3f}")
    print("---------------------------------------")
    return success_rate

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default="pasac")  # Policy name
    parser.add_argument("--env_name", default="ImitationLearning-v1")  # environment name
    parser.add_argument("--action", default=0, type=int, required=True)  # training action
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--log_path", default='reward_log', type=str)  # reward log path
    parser.add_argument("--checkpoint", default='ckpt', type=str)  # reward log path
    parser.add_argument("--max_episode", default=1e5, type=float)  # Max episode to run environment for
    parser.add_argument("--eval_freq", default=5e3, type=int)  # Evaluate frequency
    parser.add_argument("--save_models", action="store_true")  # Whether or not models are saved
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic (recommend 128)
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--max_steps", default=100, type=int)  # max steps per eposide
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    args = parser.parse_args()

    file_name = "%s_%s" % (args.policy_name, args.env_name)
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
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    discrete_action = env.discrete_action.n

    # Initialize policy
    policy = Agent(input_dim=state_dim, action_space=env.action_space, p_action_dim=discrete_action, alpha=0.05, writer=writer,
                 discount=args.discount, tau=args.tau, chkpt_dir=os.path.join(base, args.checkpoint))
    replay_buffer = ReplayBuffer()

    curr_episode = 0
    best_reward = float('-inf')
    updates_per_step = 4
    env.unwrapped.action_type = args.action
    env.unwrapped.writer = writer
    env.unwrapped.curriculum_learning = 50000
    episode_reward = 0
    success = []
    reward_history = []
    avg_reward_history = []
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
                for i in range(updates_per_step):
                    policy.learn(replay_buffer, args.batch_size)

            next_state, reward, done, info = env.step(action)
            total_reward += reward
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

        if best_reward <= total_reward/n_steps and success[-1]:
            best_reward = total_reward/n_steps
            policy.save()
            success_rate = eval_policy(policy, env)
            writer.add_scalar("Eval/success rate", success_rate, curr_episode)

        episode_reward += total_reward
        if len(reward_history) >= 100:
            moving_avg = np.mean(reward_history[-100:])
            moving_avg_avg = np.mean(avg_reward_history[-100:])
            writer.add_scalar("Reward/moving_avg_return", moving_avg, curr_episode)
            writer.add_scalar("Reward/moving_avg_of_avg_reward", moving_avg_avg, curr_episode)
        writer.add_scalar('Reward/avg_total_reward_per_episode', total_reward/n_steps, curr_episode)
        writer.add_scalar('Reward/episode reward', episode_reward, curr_episode)
        writer.add_scalar(f"Reward/success rate", sum(success[-100:]) / 100, curr_episode)
        elapsed = int(time.time() - t0)
        print('------------------------------------------------------')
        print(
            f"Episode: {curr_episode} Total reward: {total_reward:.5f}  --  Wallclk T: {elapsed // 86400}d {(elapsed % 86400) // 3600}h {(elapsed % 3600) // 60}m {elapsed % 60}s")
        print('------------------------------------------------------')
        curr_episode += 1
