import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import shutil
from config import *
import os
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
    def __init__(self, state_dim, action_dim=6, max_action=1, lr=1e-4,
                 name='actor', chkpt_dir='ckpt/td3'):
        super(Actor, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_td3')

        self.l1 = nn.Linear(state_dim, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, action_dim)
        self.max_action = max_action
        self.optimizer = Adam(self.parameters(), lr=lr)
        self.apply(weights_init)
        path_init(self.checkpoint_file)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        joint_action = self.max_action * torch.tanh(self.l3(x))
        return joint_action

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim=7, lr=1e-4,
                 name='critic', chkpt_dir='ckpt/td3'):
        super(Critic, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_td3')

        input_dim = state_dim + action_dim
        self.l1 = nn.Linear(input_dim, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, 1)

        self.l4 = nn.Linear(input_dim, 512)
        self.l5 = nn.Linear(512, 512)
        self.l6 = nn.Linear(512, 1)

        self.optimizer = Adam(self.parameters(), lr=lr)
        self.apply(weights_init)
        path_init(self.checkpoint_file)

    def forward(self, x, u):
        xu = torch.cat([x, u], -1)
        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)

        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], -1)
        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class Agent(object):
    def __init__(self, state_dim, action_dim, max_action=1, lr=1e-4, warmup=20000, writer=None,
                 discount=0.99, tau=0.001, policy_noise=0.2, noise_clip=0.2, policy_freq=2, normalizer=None,
                 chkpt_dir=None):
        self.actor = Actor(state_dim, action_dim, max_action, lr=lr, chkpt_dir=chkpt_dir).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action, lr=lr, chkpt_dir=chkpt_dir).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = Critic(state_dim, action_dim, lr=lr, chkpt_dir=chkpt_dir).to(device)
        self.critic_target = Critic(state_dim, action_dim, lr=lr, chkpt_dir=chkpt_dir).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
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
        self.seed = 0
        self.normalizer = normalizer

    def choose_action(self, state, noise_scale=0.5, validation=False):
        rng = np.random.default_rng(self.seed)  # Local RNG
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        joint_action = self.actor(state)
        if not validation:
            if self.update_step < self.warmup:
                # joint_action = torch.tensor(np.random.normal(scale=current_noise_scale, size=(self.action_dim,)), dtype=torch.float).to(device)
                joint_action = torch.tensor(rng.uniform(-self.max_action, self.max_action, size=(self.action_dim,)),
                                            dtype=torch.float, device=device)
            else:
                current_noise_scale = noise_scale * max(0.0, 1 - self.update_step / 500000)
                joint_action = joint_action + torch.normal(0, current_noise_scale, size=joint_action.shape, device=device)
            joint_action = torch.clamp(joint_action, -self.max_action, self.max_action)
        self.seed += 1
        return joint_action.cpu().data.numpy().flatten()

    def learn(self, replay_buffer, batch_size=256):
        states, next_states, actions, rewards, dones = replay_buffer.sample(batch_size)
        states[:, :12] = self.normalizer.normalize(states[:, :12])
        next_states[:, :12] = self.normalizer.normalize(next_states[:, :12])
        state = torch.tensor(states, dtype=torch.float32, device=device)
        next_state = torch.tensor(next_states, dtype=torch.float32, device=device)

        action = torch.FloatTensor(actions).to(device)
        done = torch.FloatTensor(1 - dones).to(device)
        reward = torch.FloatTensor(rewards).to(device)

        next_action = self.actor_target(next_state)
        noise = torch.randn_like(action) * self.policy_noise
        noise = noise.clamp(-self.noise_clip, self.noise_clip)
        next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

        target_Q1, target_Q2 = self.critic_target(next_state, next_action)
        target_Qs = torch.stack([target_Q1, target_Q2], dim=1)
        min_target_Q, _ = torch.min(target_Qs, dim=1)
        target_Q = reward + (done * self.discount * min_target_Q).detach()

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        # writer.add_scalar('Loss/Critic', critic_loss.item(), self.update_step)
        # self.writer.add_scalar(
        #     "Loss/critic", critic_loss.item(), self.update_step
        # )

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        if self.update_step % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            # self.writer.add_scalar(
            #     "Loss/actor", actor_loss.item(), self.update_step
            # )

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # self.writer.add_scalar(
        #     "Q/Q1_mean", current_Q1.mean().item(), self.update_step
        # )
        # self.writer.add_scalar(
        #     "Q/Q2_mean", current_Q2.mean().item(), self.update_step
        # )
        self.update_step += 1

    def save(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()