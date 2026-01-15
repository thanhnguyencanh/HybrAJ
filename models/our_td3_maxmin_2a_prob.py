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
        self.l3 = nn.Linear(512, action_dim)

        self.p_action = nn.Linear(512, p_action_dim)
        self.max_action = max_action
        self.optimizer = Adam(self.parameters(), lr=lr)
        self.apply(weights_init)
        path_init(self.checkpoint_file)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        # Discrete action prediction
        p_action_logits = self.p_action(x)
        # p_action_id = torch.argmax(p_action_logits, dim=-1)  # Deterministic choice
        # p_action_onehot = F.one_hot(p_action_id, p_action_logits.size(-1)).float().to(device)
        joint_action = self.max_action * torch.tanh(self.l3(x))
        return joint_action, p_action_logits

    def sample_discrete(self, p_action_logits, epsilon=0.2):
        p_action_probs = F.softmax(p_action_logits, dim=-1)
        batch_size = p_action_logits.shape[0]
        if np.random.random() < epsilon:
            return torch.randint(0, p_action_logits.shape[-1], (batch_size, 1), device=device)
        return torch.argmax(p_action_probs, dim=-1)

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
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)

        return x1, x2

    def Q1(self, x, u, p_action_id):
        if p_action_id.dim() == 1:
            p_action_id = p_action_id.unsqueeze(-1)  # (B, 1)
        xu = torch.cat([x, u, p_action_id], -1)
        # xu = torch.cat([x, u], -1)
        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class Agent(object):
    def __init__(self, state_dim, action_dim, p_action_dim, max_action=1, lr=1e-4, epsilon=1.0, eps_min=0.05, eps_dec=1e-5,
                 discount=0.99, tau=0.001, policy_noise=0.2, noise_clip=0.2, policy_freq=2, warmup=20000, writer=None,
                 normalizer=None, chkpt_dir=None):
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
        self.normalizer = normalizer

    def choose_action(self, state, noise_scale=0.5, validation=False):
        rng = np.random.default_rng(self.seed)  # Local RNG
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        joint_action, p_action_logits = self.actor(state)
        p_action_id = self.actor.sample_discrete(p_action_logits,
                                                 epsilon=self.eps if self.update_step < self.warmup else 0.0)  # Small epsilon post-warmup
        if not validation:
            if self.update_step < self.warmup:
                # joint_action = torch.tensor(np.random.normal(scale=current_noise_scale, size=(self.action_dim,)), dtype=torch.float).to(device)
                joint_action = torch.tensor(rng.uniform(-self.max_action, self.max_action, size=(self.action_dim,)),
                                            dtype=torch.float, device=device)
            else:
                current_noise_scale = noise_scale * max(0.0, 1 - self.update_step / 500000)
                joint_action = joint_action + torch.normal(0, current_noise_scale, size=joint_action.shape, device=device)
            joint_action = torch.clamp(joint_action, -self.max_action, self.max_action)
        self.decrement_epsilon()
        self.seed += 1
        return joint_action.cpu().data.numpy().flatten(), p_action_id.cpu().data.numpy().flatten()

    def decrement_epsilon(self):
        # Which one is the best choice for discrete learning
        self.eps = max(self.eps_min, self.eps - self.eps_dec)  # Additive decay (eps_min=0.05, eps=1.0, eps_decay=0.0001)
        # self.eps = max(self.eps_min, self.eps*self.eps_dec)  # Multiplicative decay (eps_min=0.01, eps=1.0, eps_decay=0.995)

    def learn(self, replay_buffer, batch_size=128):
        states, next_states, actions, rewards, dones = replay_buffer.sample(batch_size)
        states[:, :12] = self.normalizer.normalize(states[:, :12])
        next_states[:, :12] = self.normalizer.normalize(next_states[:, :12])
        state = torch.tensor(states, dtype=torch.float32, device=device)
        next_state = torch.tensor(next_states, dtype=torch.float32, device=device)

        action, p_action = actions[:, :-1], actions[:, -1]
        action = torch.FloatTensor(action).to(device)
        p_action = torch.LongTensor(p_action).to(device)
        done = torch.FloatTensor(1 - dones).to(device)
        reward = torch.FloatTensor(rewards).to(device)

        with torch.no_grad():
            # Next action = noise + actor target of next state
            next_action, next_p_action_logits = self.actor_target(next_state)
            next_p_action_probs = F.softmax(next_p_action_logits, dim=-1)
            # next_p_action_id = self.actor_target.sample_discrete(next_p_action_logits,
            #                                                      epsilon=0.0)  # Deterministic for target
            noise = torch.randn_like(action) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)  # Joints
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

        # Transform to onehot to flow the gradient
        p_action_onehot = F.one_hot(p_action, self.p_action_dim).float()
        target_Qs = []
        for p_a in range(self.p_action_dim):
            p_a_id = torch.full((batch_size,), p_a, device=device)
            p_a_onehot = F.one_hot(p_a_id, self.p_action_dim).float()
            q1, q2 = self.critic_target(next_state, next_action, p_a_onehot)
            q = torch.min(q1, q2)
            target_Qs.append(q * next_p_action_probs[:, p_a])

        next_Q = 0.1 * torch.min(target_Qs[0], target_Qs[1]) + (1 - 0.1) * torch.max(target_Qs[0], target_Qs[1])
        target_Q = reward + done * self.discount * next_Q
        current_Q1, current_Q2 = self.critic(state, action, p_action_onehot)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.writer.add_scalar("Critic/critic_loss", critic_loss.item(), self.update_step)

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        if self.update_step % self.policy_freq == 0:
            joint_action, p_action_logits = self.actor(state)
            p_action_probs = F.softmax(p_action_logits, dim=-1)

            q_values = []
            for p_a in range(self.p_action_dim):
                p_a_id = torch.tensor(p_a, device=device).expand(batch_size)
                p_action_onehot = F.one_hot(p_a_id, self.p_action_dim).float()
                q1, q2 = self.critic(state, joint_action, p_action_onehot)  # continuous branch come through Q,
                qs = torch.stack([q1, q2], dim=1)

                min_q, _ = torch.min(qs, dim=1)
                weighted_q = qs * p_action_probs[:, p_a]  # discrete branch come through the expectation weighting of Q.
                q_values.append(weighted_q)
            actor_loss = -torch.stack(q_values, dim=1).sum(dim=1).mean()

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()
            grads = []
            for p in self.actor.parameters():
                if p.grad is not None:
                    # grad_norm += p.grad.norm().item()
                    grads.append(p.grad.view(-1))
            grads = torch.cat(grads)  # shape: (num_params,)
            grad_norm = grads.norm(p=2)
            grad_var = grads.var(unbiased=False)
            self.writer.add_scalar("Actor/actor_loss", actor_loss.item(), self.update_step)
            self.writer.add_scalar("Actor/grad_norm", grad_norm, self.update_step)
            self.writer.add_scalar("Actor/grad_variance", grad_var.item(), self.update_step)
            # self.writer.add_scalar("Actor/discrete_entropy", entropy.item(), self.update_step)

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