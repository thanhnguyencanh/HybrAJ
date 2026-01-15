import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import os
import shutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def path_init(file_path):
    dir_path = os.path.dirname(file_path)
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.constant_(m.bias, 0.0)


class Actor(nn.Module):
    def __init__(self, state_dim, p_action_dim=4, action_dim=7, max_action=1, lr=1e-4,
                 name='actor', chkpt_dir='ckpt/modified_td3'):
        super(Actor, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_td3')

        self.l1 = nn.Linear(state_dim, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512 + 32, action_dim)  # Adjusted for embedding
        self.p_action = nn.Linear(512, p_action_dim)
        self.p_embed = nn.Embedding(p_action_dim, 32)
        self.max_action = max_action
        self.optimizer = Adam(self.parameters(), lr=lr)
        self.apply(weights_init)
        path_init(self.checkpoint_file)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        p_action_logits = self.p_action(x)
        p_action_id = torch.argmax(p_action_logits, dim=-1, keepdim=True)
        p_embed = self.p_embed(p_action_id).squeeze(1)
        joint_input = torch.cat([x, p_embed], dim=-1)
        joint_action = self.max_action * torch.tanh(self.l3(joint_input))
        return joint_action, p_action_logits

    def sample_discrete(self, p_action_logits, epsilon=0.2):
        p_action_probs = F.softmax(p_action_logits, dim=-1)
        batch_size = p_action_logits.shape[0]
        if np.random.random() < epsilon:
            return torch.randint(0, p_action_logits.shape[-1], (batch_size, 1), device=device)
        return torch.argmax(p_action_probs, dim=-1, keepdim=True)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim=7, p_action_dim=5, lr=1e-4,
                 name='critic', chkpt_dir='ckpt/modified_td3'):
        super(Critic, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_td3')

        self.p_action_embed = nn.Embedding(p_action_dim, 32)
        input_dim = state_dim + action_dim + 32
        self.l1 = nn.Linear(input_dim, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, 1)

        self.l4 = nn.Linear(input_dim, 512)
        self.l5 = nn.Linear(512, 512)
        self.l6 = nn.Linear(512, 1)

        self.l7 = nn.Linear(input_dim, 512)
        self.l8 = nn.Linear(512, 512)
        self.l9 = nn.Linear(512, 1)

        self.optimizer = Adam(self.parameters(), lr=lr)
        self.apply(weights_init)
        path_init(self.checkpoint_file)

    def forward(self, x, u, p_action_id):
        p_action_id = p_action_id.view(-1)
        p_action_embed = self.p_action_embed(p_action_id)
        xu = torch.cat([x, u, p_action_embed], -1)
        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)

        x3 = F.relu(self.l7(xu))
        x3 = F.relu(self.l8(x3))
        x3 = self.l9(x3)

        return x1, x2, x3

    def Q1(self, x, u, p_action_id):
        p_action_embed = self.p_action_embed(p_action_id).squeeze(1)
        xu = torch.cat([x, u, p_action_embed], -1)
        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class Agent(object):
    def __init__(self, state_dim, action_dim, p_action_dim, max_action=1, lr=1e-4, epsilon=1.0, eps_min=0.05,
                 eps_dec=1e-5, discount=0.99, tau=0.001,
                 policy_noise=0.2, noise_clip=0.2, policy_freq=4, warmup=20000, writer=None, episode=0):
        self.actor = Actor(state_dim, action_dim, p_action_dim, max_action, lr=lr).to(device)
        self.actor_target = Actor(state_dim, action_dim, p_action_dim, max_action, lr=lr).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = Critic(state_dim=state_dim, action_dim=action_dim, p_action_dim=p_action_dim, lr=lr).to(device)
        self.critic_target = Critic(state_dim=state_dim, action_dim=action_dim, p_action_dim=p_action_dim, lr=lr).to(
            device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.p_action_dim = p_action_dim
        self.warmup = warmup
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.max_action = max_action
        self.action_dim = action_dim
        self.update_step = 0
        self.writer = writer
        self.eps = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.scale_rate = episode + 1000000

    def select_action(self, state, noise_scale=0.5, validation=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        x = F.relu(self.actor.l1(state))
        x = F.relu(self.actor.l2(x))
        p_action_logits = self.actor.p_action(x)
        if validation:
            p_action_id = torch.argmax(p_action_logits, dim=-1, keepdim=True)
        else:
            p_action_id = self.actor.sample_discrete(p_action_logits,
                                                     epsilon=self.eps if self.update_step < self.warmup else 0.0)
        p_embed = self.actor.p_embed(p_action_id).squeeze(1)
        joint_input = torch.cat([x, p_embed], dim=-1)
        joint_action = self.actor.max_action * torch.tanh(self.actor.l3(joint_input))

        if not validation:
            current_noise_scale = noise_scale * max(0.01, 1 - self.update_step / 700000)
            if self.update_step < self.warmup:
                joint_action = torch.tensor(np.random.normal(scale=current_noise_scale, size=(self.action_dim,))).to(
                    device)
            else:
                joint_action = joint_action + torch.tensor(np.random.normal(scale=current_noise_scale),
                                                           dtype=torch.float).to(device)
                joint_action = torch.clamp(joint_action, -self.max_action, self.max_action)
        self.decrement_epsilon()
        return joint_action.cpu().data.numpy().flatten(), p_action_id.cpu().data.numpy().flatten()

    def decrement_epsilon(self):
        self.eps = max(self.eps_min, self.eps * self.eps_dec)

    def learn(self, replay_buffer, batch_size=128):
        ids, states, next_states, actions, rewards, dones = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(states).to(device)
        next_state = torch.FloatTensor(next_states).to(device)
        action, p_action = actions
        action = torch.FloatTensor(action).to(device)
        p_action = torch.LongTensor(p_action).to(device)
        done = torch.FloatTensor(1 - dones).to(device)
        reward = torch.FloatTensor(rewards).to(device)

        next_action, next_p_action_logits = self.actor_target(next_state)
        next_p_action_id = self.actor_target.sample_discrete(next_p_action_logits, epsilon=0.0)
        noise = torch.randn_like(action) * self.policy_noise
        noise = noise.clamp(-self.noise_clip, self.noise_clip)
        next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

        target_Q1, target_Q2, target_Q3 = self.critic_target(next_state, next_action, next_p_action_id)
        target_Qs = torch.stack([target_Q1, target_Q2, target_Q3], dim=1)
        median_target_Q, _ = torch.median(target_Qs, dim=1)
        target_Q = reward + (done * self.discount * median_target_Q).detach()

        current_Q1, current_Q2, current_Q3 = self.critic(state, action, p_action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) + F.mse_loss(current_Q3,
                                                                                                       target_Q)
        self.writer.add_scalar("Loss/critic", critic_loss.item(), self.update_step)

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        if self.update_step % self.policy_freq == 0:
            x = F.relu(self.actor.l2(F.relu(self.actor.l1(state))))
            p_action_logits = self.actor.p_action(x)
            p_action_probs = F.softmax(p_action_logits, dim=-1)

            q_values = []
            for p_a in range(self.p_action_dim):
                p_a_id = torch.full((batch_size,), p_a, device=device)
                p_embed = self.actor.p_embed(p_a_id)
                joint_input = torch.cat([x, p_embed], dim=-1)
                joint_action_p_a = self.actor.max_action * torch.tanh(self.actor.l3(joint_input))
                q1, q2, q3 = self.critic(state, joint_action_p_a, p_a_id)
                qs = torch.stack([q1, q2, q3], dim=1)
                median_q, _ = torch.median(qs, dim=1)
                weighted_q = median_q * p_action_probs[:, p_a]
                q_values.append(weighted_q)
            actor_loss = -torch.stack(q_values, dim=1).sum(dim=1).mean()
            self.writer.add_scalar("Loss/actor", actor_loss.item(), self.update_step)

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.writer.add_scalar("Q/Q1_mean", current_Q1.mean().item(), self.update_step)
        self.writer.add_scalar("Q/Q2_mean", current_Q2.mean().item(), self.update_step)
        self.writer.add_scalar("Q/Q3_mean", current_Q3.mean().item(), self.update_step)
        self.update_step += 1

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), f'{directory}/{filename}_actor.pth')
        torch.save(self.critic.state_dict(), f'{directory}/{filename}_critic.pth')

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load(f'{directory}/{filename}_actor.pth'))
        self.critic.load_state_dict(torch.load(f'{directory}/{filename}_critic.pth'))