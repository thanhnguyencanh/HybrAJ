import numpy as np
import gym
import env
from moviepy.editor import ImageSequenceClip
from datetime import datetime
import argparse
from torch.utils.tensorboard import SummaryWriter
import shutil
import os
import time
from utils.relay_buffer import ReplayBuffer
from models.our_td3_min_2a import Agent
from utils.normalizer import RunningStatNormalizer
import warnings
warnings.filterwarnings("ignore")

def eval_policy(policy, env, normalizer, eval_episodes=10, video=None, save=False):
    current_datetime = datetime.now()
    timestamp = current_datetime.strftime("%Y%m%d_%H%M%S")
    success_rate = 0
    for i in range(eval_episodes):
        state, done = env.reset(), False
        while not done:
            normalized_state = state.copy()
            normalized_state[:12] = normalizer.normalize(state[:12])
            action = policy.choose_action(normalized_state, validation=True)
            state, reward, done, info = env.step(action)
            # avg_reward += reward
        success_rate += any(info["log"])
        is_success = any(info["log"])
        if is_success and save:
            print("...video saving...")
            debug_clip = ImageSequenceClip(env.cache_front_video, fps=15)
            video_path = os.path.join(video, f"demo_ep{i}_view_0_action_{env.unwrapped.action_type}_{timestamp}_state_{is_success}.mp4")
            debug_clip.write_videofile(video_path, fps=15)

            debug_clip = ImageSequenceClip(env.cache_side_video, fps=15)
            video_path = os.path.join(video, f"demo_ep{i}_view_1_action_{env.unwrapped.action_type}_{timestamp}_state_{is_success}.mp4")
            debug_clip.write_videofile(video_path, fps=15)

            debug_clip = ImageSequenceClip(env.cache_diagonal_video, fps=15)
            video_path = os.path.join(video, f"demo_ep{i}_view_2_action_{env.unwrapped.action_type}_{timestamp}_state_{is_success}.mp4")
            debug_clip.write_videofile(video_path, fps=15)
            print(f"Video saved to {video_path}")

    success_rate /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {success_rate:.3f}")
    print("---------------------------------------")

    return success_rate

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default="td3_min_2a")  # Policy name
    parser.add_argument("--env_name", default="ImitationLearning-v1")  # environment name
    parser.add_argument("--warmup", default=20000, type=int)  # How many time steps purely random policy is run for (20000 for all actions)
    parser.add_argument("--action", default=0, type=int, required=True)  # training action
    parser.add_argument("--render", default=False, type=bool, required=True)  # training action
    parser.add_argument("--eval", default=False, type=bool, required=True)  # training action
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
    normalizer = RunningStatNormalizer(shape=(12,))
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    discrete_action = env.discrete_action.n

    # Initialize policy
    policy = Agent(state_dim=state_dim, action_dim=action_dim, p_action_dim=discrete_action, max_action=max_action,
                   warmup=args.warmup, writer=writer, discount=args.discount, tau=args.tau, policy_noise=args.policy_noise,
                   noise_clip=args.noise_clip, policy_freq=args.policy_freq, normalizer=normalizer, chkpt_dir=os.path.join(base, args.checkpoint))
    replay_buffer = ReplayBuffer()

    max_episode_avg_reward = best_reward = max_episode_reward = float('-inf')
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
        normalizer.update(state[:12])

        while not done:
            normalized_state = state.copy()
            normalized_state[:12] = normalizer.normalize(state[:12])
            # Use policy with exploration noise
            action = policy.choose_action(normalized_state, noise_scale=args.expl_noise)
            if replay_buffer.ready(batch_size=args.batch_size):
                policy.learn(replay_buffer, args.batch_size)

            next_state, reward, done, info = env.step(action)
            total_reward += reward  # total step reward
            replay_buffer.add((state, next_state, action, reward, done))

            normalizer.update(next_state[:12])
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
            env.unwrapped.virtualize = False
            env.unwrapped.eval = False
            success_rate = eval_policy(policy, env, normalizer, eval_episodes=15, video=args.video, save=env.unwrapped.virtualize)
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