import numpy as np
import torch
from collections import defaultdict
from moviepy.editor import ImageSequenceClip
from datetime import datetime
import os

def eval_policy(policy, env, normalizer, eval_episodes=15, video=None, save=False, gamma=0.99, writer=None, eval_step=None):
    success_rate = 0
    current_datetime = datetime.now()
    timestamp = current_datetime.strftime("%Y%m%d_%H%M%S")
    all_bias = []
    all_abs_bias = []
    all_sq_bias = []

    for i in range(eval_episodes):
        state, done = env.reset(), False

        traj_states = []
        traj_actions = []
        traj_rewards = []

        while not done:
            normalized_state = state.copy()
            normalized_state[:12] = normalizer.normalize(state[:12])

            action = policy.choose_action(normalized_state, validation=True)
            next_state, reward, done, info = env.step(action)

            traj_states.append(state)
            traj_actions.append(action)
            traj_rewards.append(reward)

            state = next_state

        success_rate += any(info["log"])
        is_success = any(info["log"])

        if is_success and save:  # save video demo
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

        # ---------- Monte Carlo returns ----------
        T = len(traj_rewards)
        returns = np.zeros(T)
        G = 0.0
        for t in reversed(range(T)):
            G = traj_rewards[t] + gamma * G
            returns[t] = G

        # ---------- Q-value estimation ----------
        q_sum = defaultdict(float)
        q_count = defaultdict(int)

        for t in range(T):
            s = traj_states[t].copy()
            s[:12] = normalizer.normalize(s[:12])
            a = traj_actions[t]

            s_t = torch.FloatTensor(s).unsqueeze(0).to(policy.device)
            a_t = torch.FloatTensor(a).unsqueeze(0).to(policy.device)

            with torch.no_grad():
                q1, q2 = policy.critic(s_t, a_t)
                q = 0.5 * (q1 + q2)

            key = (t,)  # time index key (safe for continuous actions)
            q_sum[key] += q.item()
            q_count[key] += 1

        # ---------- bias computation ----------
        for t in range(T):
            q_avg = q_sum[(t,)] / q_count[(t,)]
            bias = returns[t] - q_avg

            all_bias.append(bias)
            all_abs_bias.append(abs(bias))
            all_sq_bias.append(bias ** 2)

        writer.add_scalar("Eval/mean_bias", bias, eval_step)
        writer.add_scalar("Eval/mean_abs_bias", abs(bias), eval_step)
        writer.add_scalar("Eval/rmse_bias", bias ** 2, eval_step)
        eval_step += 1
    success_rate /= eval_episodes

    mean_bias = np.mean(all_bias)  # if > 0 -> underestimates else
    mean_abs_bias = np.mean(all_abs_bias)
    rmse_bias = np.sqrt(np.mean(all_sq_bias))

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes")
    print(f"Success rate: {success_rate:.3f}")
    print(f"Critic bias (mean): {mean_bias:.4f}")
    print(f"Critic bias (MAE): {mean_abs_bias:.4f}")
    print(f"Critic bias (RMSE): {rmse_bias:.4f}")
    print("---------------------------------------")

    return success_rate, mean_bias, mean_abs_bias, rmse_bias
