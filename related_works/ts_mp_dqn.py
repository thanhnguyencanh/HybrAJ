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
    def __init__(self, state_dim, p_action_dim=2, action_dim=6, max_action=1, lr=1e-4,
                 name='actor', chkpt_dir='ckpt/our_td3_min_1a'):
        super(Actor, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_td3')

        self.l1 = nn.Linear(state_dim, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3_1 = nn.Linear(512, action_dim)
        self.l3_2 = nn.Linear(512, action_dim)
        # self.p_action = nn.Linear(512, p_action_dim)
        self.max_action = max_action
        self.optimizer = Adam(self.parameters(), lr=lr)
        self.apply(weights_init)
        path_init(self.checkpoint_file)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        # Discrete action prediction
        # p_action_logits = self.p_action(x)
        # p_action_id = torch.argmax(p_action_logits, dim=-1)  # Deterministic choice

        joint_action1 = self.max_action * torch.tanh(self.l3_1(x))
        joint_action2 = self.max_action * torch.tanh(self.l3_2(x))
        return joint_action1, joint_action2

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim=6, p_action_dim=2, lr=1e-4,
                 name='critic', chkpt_dir='ckpt/our_td3_min_1a'):
        super(Critic, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_td3')

        input_dim = state_dim + action_dim + p_action_dim
        self.l1 = nn.Linear(input_dim, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, 1)

        self.l4 = nn.Linear(input_dim, 512)
        self.l5 = nn.Linear(512, 512)
        self.l6 = nn.Linear(512, 1)

        self.optimizer = Adam(self.parameters(), lr=lr)
        self.apply(weights_init)
        path_init(self.checkpoint_file)

    def forward(self, x, u, p_action_id):
        if p_action_id.dim() == 1:
            p_action_id = p_action_id.unsqueeze(-1)  # (B, 1)
        xu = torch.cat([x, u, p_action_id], -1)
        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        return self.l3(x1)

    def main(self, x, u, p_action_id):
        if p_action_id.dim() == 1:
            p_action_id = p_action_id.unsqueeze(-1)  # (B, 1)
        xu = torch.cat([x, u, p_action_id], -1)
        main = F.relu(self.l4(xu))
        main = F.relu(self.l5(main))
        return self.l6(main)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class Agent(object):
    def __init__(self, state_dim, action_dim, p_action_dim, max_action=1, lr=1e-4, epsilon=1.0, eps_min=0.05, eps_dec=1e-5,
                 discount=0.99, tau=0.001, policy_noise=0.2, noise_clip=0.2, policy_freq=2, warmup=20000, writer=None, chkpt_dir=None):
        self.actor = Actor(state_dim, action_dim=action_dim, p_action_dim=p_action_dim, max_action=max_action, lr=lr, chkpt_dir=chkpt_dir).to(device)
        self.actor_target = Actor(state_dim, action_dim=action_dim, p_action_dim=p_action_dim, max_action=max_action, lr=lr, chkpt_dir=chkpt_dir).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = Critic(state_dim=state_dim, action_dim=action_dim, p_action_dim=p_action_dim, lr=lr, chkpt_dir=chkpt_dir).to(device)
        self.critic_target = Critic(state_dim=state_dim, action_dim=action_dim, p_action_dim=p_action_dim, lr=lr, chkpt_dir=chkpt_dir).to(device)
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
        self.seed = 0

    def choose_action(self, state):
        rng = np.random.default_rng(self.seed)  # Local RNG
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        joint_action = self.actor(state)

        q_values = []
        for k in range(self.p_action_dim):
            k_id = torch.full((1,), k, device=device, dtype=torch.long)
            p_action_onehot = F.one_hot(k_id, self.p_action_dim).float()
            q_k = self.critic.main(state, joint_action[k], p_action_onehot)  # torch.Size([1, 1])
            q_values.append(q_k)
        q_values = torch.stack(q_values, dim=1)  # [1, K, 1]
        # argmax over discrete actions
        k = q_values.argmax(dim=1).item()
        x_k = joint_action[k-1]

        if self.update_step < self.warmup:
            x_k = torch.tensor(rng.uniform(-self.max_action, self.max_action, size=(self.action_dim,)),
                               dtype=torch.float, device=device)
        x_k = torch.clamp(x_k, -self.max_action, self.max_action)
        self.seed += 1
        return x_k.cpu().data.numpy().flatten(), np.array([k])

    def learn(self, replay_buffer, batch_size=128):
        states, next_states, actions, rewards, dones = replay_buffer.sample(batch_size)
        state = torch.tensor(states, dtype=torch.float32, device=device)
        next_state = torch.tensor(next_states, dtype=torch.float32, device=device)

        action, p_action = actions[:, :-1], actions[:, -1]
        action = torch.FloatTensor(action).to(device)
        p_action = torch.LongTensor(p_action).to(device)
        done = torch.FloatTensor(1 - dones).to(device)
        reward = torch.FloatTensor(rewards).to(device)

        with torch.no_grad():
            # Next action = noise + actor target of next state
            next_action = self.actor_target(next_state)
            # next_p_action_id = self.actor_target.sample_discrete(next_p_action_logits,
            #                                                      epsilon=0.0)  # Deterministic for target
            noise = torch.randn_like(action) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)  # Joints
            next_action = [(action + noise).clamp(-self.max_action, self.max_action) for action in list(next_action)]

        # Transform to onehot to flow the gradient
        p_action_onehot = F.one_hot(p_action, self.p_action_dim).float()
        # Compute the target Q-value
        q1_list, q2_list = [], []

        for k in range(self.p_action_dim):
            k_id = torch.full((batch_size,), k, device=device, dtype=torch.long)
            onehot = F.one_hot(k_id, self.p_action_dim).float()
            q1 = self.critic_target(next_state, next_action[k], onehot)
            q2 = self.critic_target.main(next_state, next_action[k], onehot)
            q1_list.append(q1)
            q2_list.append(q2)
        # q1_stack = torch.stack(q1_list, dim=1)
        # q2_stack = torch.stack(q2_list, dim=1)
        # min_q = torch.min(q1_stack, q2_stack)  # min over critics
        # target_Q = reward + done * self.discount * min_q.max(dim=1)[0].detach()
        q1_stack = torch.stack(q1_list, dim=1)  # [B, K, 1]
        q2_stack = torch.stack(q2_list, dim=1)  # [B, K, 1]
        max_q1, _ = q1_stack.max(dim=1)  # [B,1]  <- got overestimation bias
        max_q2, _ = q2_stack.max(dim=1)  # [B,1]
        min_max_q = torch.min(max_q1, max_q2)  # [B,1]
        target_Q = reward + done * self.discount * min_max_q.detach()

        current_Q1 = self.critic(state, action, p_action_onehot)
        main = self.critic.main(state, action, p_action_onehot)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(main, target_Q)
        # Compute bias and variance
        self.writer.add_scalar("Critic/critic_loss", critic_loss.item(), self.update_step)

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        if self.update_step % self.policy_freq == 0:
            joint_action = self.actor(state)

            q_values = []
            for p_a in range(self.p_action_dim):
                p_a_id = torch.tensor(p_a, device=device).expand(batch_size)
                p_action_onehot = F.one_hot(p_a_id, self.p_action_dim).float()
                main = self.critic.main(state, joint_action[p_a], p_action_onehot)  # continuous branch come through Q,
                q_values.append(main)
            actor_loss = -torch.stack(q_values, dim=1).sum(dim=1).mean()

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()
            self.writer.add_scalar("Actor/actor_loss", actor_loss.item(), self.update_step)

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        self.update_step += 1

    def save(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()