import numpy as np
from datetime import datetime
import gym
import env
from moviepy.editor import ImageSequenceClip
import argparse
from torch.utils.tensorboard import SummaryWriter
import shutil
import os
import time
from utils.relay_buffer import ReplayBuffer
from related_works.ts_mp_dqn import Agent
from utils.normalizer import RunningStatNormalizer
import warnings
warnings.filterwarnings("ignore")

def eval_policy(policy, env, eval_episodes=10, video=None, save=False):
    current_datetime = datetime.now()
    timestamp = current_datetime.strftime("%Y%m%d_%H%M%S")
    success_rate = 0
    for i in range(eval_episodes):
        state, done = env.reset(), False
        while not done:
            action = policy.choose_action(state, validation=True)
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
    parser.add_argument("--policy_name", default="tsmpdqn")  # Policy name
    parser.add_argument("--env_name", default="ImitationLearning-v1")  # environment name
    parser.add_argument("--warmup", default=20000, type=int)  # purely random action
    parser.add_argument("--action", default=0, type=int, required=True)  # training action
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--log_path", default='reward_log', type=str)  # reward log path
    parser.add_argument("--checkpoint", default='ckpt', type=str)  # checkpoint log path
    parser.add_argument("--video", default='demo', type=str)  # demo path
    parser.add_argument("--max_episode", default=1e6, type=int)  # Max episode to run environment for
    parser.add_argument("--eval_freq", default=5e3, type=int)  # Evaluate frequency
    parser.add_argument("--expl_noise", default=0.2, type=float)  # Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic (recommend 256)
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--max_steps", default=100, type=int)  # max steps per episode
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    args = parser.parse_args()

    file_name = "%s_%s" % (args.policy_name, args.env_name)
    print("---------------------------------------")
    print("Settings: %s" % (file_name))
    print("---------------------------------------")

    # Initialize saving file
    base = f'{args.policy_name}_{str(args.action)}'
    for path in [args.log_path, args.video]:
        path = os.path.join(base, path)
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(path)

    writer = SummaryWriter(os.path.join(base, args.log_path))
    env = gym.make(args.env_name)
    env.unwrapped.max_steps = args.max_steps  # define max steps
    env.unwrapped.max_episode = args.max_episode  # define max steps
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    discrete_action = env.discrete_action.n
    # Initialize policy
    policy = Agent(state_dim=state_dim, action_dim=action_dim, p_action_dim=discrete_action, max_action=max_action, warmup=args.warmup, lr=1e-4,
                   writer=writer, discount=args.discount, tau=args.tau, policy_noise=args.policy_noise,
                   noise_clip=args.noise_clip, policy_freq=args.policy_freq, chkpt_dir=os.path.join(base, args.checkpoint))
    replay_buffer = ReplayBuffer()

    max_episode_avg_reward = best_reward = float('-inf')
    env.unwrapped.action_type = args.action
    env.unwrapped.writer = writer
    reward_history = []
    avg_reward_history = []
    curr_episode = 0
    env.unwrapped.curriculum_learning = 50000
    episode_reward = 0
    success = []
    t0 = time.time()
    '''TRAINING PROCESS'''
    while curr_episode < args.max_episode:
        total_reward = 0
        n_steps = 0
        state = env.reset()
        done = False

        while not done:
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

        if best_reward <= total_reward/n_steps and success[-1]:
            best_reward = total_reward/n_steps
            policy.save()
            # env.unwrapped.virtualize = True
            # env.unwrapped.eval = True
            success_rate = eval_policy(policy, env, eval_episodes=10, video=args.video, save=env.unwrapped.virtualize)
            writer.add_scalar("Eval/success rate", success_rate, curr_episode)

        env.unwrapped.virtualize = False
        env.unwrapped.eval = False
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
        print(f"Episode: {curr_episode} Total reward: {total_reward:.5f}  --  Wallclk T: {elapsed // 86400}d {(elapsed % 86400) // 3600}h {(elapsed % 3600) // 60}m {elapsed % 60}s")
        print('------------------------------------------------------')
        curr_episode += 1
