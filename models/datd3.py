import copy
import numpy as np
import time
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def path_init(dir_path):
    # dir_path = os.path.dirname(file_path)
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, p_action_dim):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, action_dim)
        self.p_action = nn.Linear(512, p_action_dim)

        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        # Discrete action prediction
        p_action_logits = self.p_action(x)

        # Continuous action prediction
        joint_action = self.max_action * torch.tanh(self.l3(x))
        return joint_action, p_action_logits

    def sample_discrete(self, p_action_logits, epsilon=0.2):
        p_action_probs = F.softmax(p_action_logits, dim=-1)
        batch_size = p_action_logits.shape[0]
        if np.random.random() < epsilon:
            return torch.randint(0, p_action_logits.shape[-1], (batch_size, 1), device=device)
        return torch.argmax(p_action_probs, dim=-1)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim + 2, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, 1)

    def forward(self, state, action, p_action_id):
        if p_action_id.dim() == 1:
            p_action_id = p_action_id.unsqueeze(-1)  # (B, 1)
        xu = torch.cat([state, action, p_action_id], -1)

        q = F.relu(self.l1(xu))
        q = F.relu(self.l2(q))
        q = self.l3(q)

        return q

class Agent(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            p_action_dim,
            warmup=20000,
            writer=None,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            lr=1e-4,
            normalizer=None,
            chkpt_dir=None
    ):
        self.device = device

        self.actor1 = Actor(state_dim, action_dim, max_action, p_action_dim).to(self.device)
        self.actor1_target = copy.deepcopy(self.actor1)
        self.actor1_optimizer = torch.optim.Adam(self.actor1.parameters(), lr=lr)

        self.actor2 = Actor(state_dim, action_dim, max_action, p_action_dim).to(self.device)
        self.actor2_target = copy.deepcopy(self.actor2)
        self.actor2_optimizer = torch.optim.Adam(self.actor2.parameters(), lr=lr)

        self.critic1 = Critic(state_dim, action_dim).to(self.device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr)

        self.critic2 = Critic(state_dim, action_dim).to(self.device)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr)
        self.action_dim = action_dim
        self.max_action = max_action
        self.p_action_dim = p_action_dim
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.normalizer = normalizer
        self.chkpt_dir = chkpt_dir
        self.warmup = warmup
        self.writer = writer
        self.update_step = 0
        self.seed = 0
        self.eps_min = 0.05
        self.eps_dec = 1e-5
        self.eps = 0
        path_init(self.chkpt_dir)

    def choose_action(self, state, noise_scale=0.2, validation=False):
        rng = np.random.default_rng(self.seed)  # Local RNG
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action1, p_action_logits1 = self.actor1(state)
        action2, p_action_logits2 = self.actor2(state)
        p_action_id1 = self.actor1.sample_discrete(p_action_logits1,
                                                   epsilon=self.eps if self.update_step < self.warmup else 0.0)
        p_action_id2 = self.actor2.sample_discrete(p_action_logits2,
                                                   epsilon=self.eps if self.update_step < self.warmup else 0.0)
        p_action_id1 = p_action_id1.view(-1)  # đảm bảo shape (B,)
        p_action_id2 = p_action_id2.view(-1)
        p_action_onehot1 = F.one_hot(p_action_id1, self.p_action_dim).float()
        p_action_onehot2 = F.one_hot(p_action_id2, self.p_action_dim).float()
        q1 = self.critic1(state, action1, p_action_onehot1)
        q2 = self.critic2(state, action2, p_action_onehot2)
        if q1 >= q2:
            action = action1
            p_action = p_action_id1
        else:
            action = action2
            p_action = p_action_id2

        if not validation:
            if self.update_step < self.warmup:
                action = torch.tensor(rng.uniform(-self.max_action, self.max_action, size=(self.action_dim,)),
                                      dtype=torch.float, device=device)

            else:
                current_noise_scale = noise_scale * max(0.0, 1 - self.update_step / 500000)
                action = action + torch.normal(0, current_noise_scale, size=action.shape, device=device)
            action = torch.clamp(action, -self.max_action, self.max_action)
        self.decrement_epsilon()
        self.seed += 1
        return action.cpu().data.numpy().flatten(), p_action.cpu().data.numpy().flatten()

    def decrement_epsilon(self):
        # Which one is the best choice for discrete learning
        self.eps = max(self.eps_min, self.eps - self.eps_dec)

    def learn(self, replay_buffer, batch_size=256):
        ## cross update scheme
        self.train_one_q_and_pi(replay_buffer, True, batch_size=batch_size)
        self.train_one_q_and_pi(replay_buffer, False, batch_size=batch_size)

    def train_one_q_and_pi(self, replay_buffer, update_a1=True, batch_size=256):

        state, next_state, action, reward, done = replay_buffer.sample(batch_size)
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        action, p_action = action[:, :-1], action[:, -1]
        action = torch.FloatTensor(action).to(device)
        p_action = torch.LongTensor(p_action).to(device)
        not_done = torch.FloatTensor(1 - done).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)

        with torch.no_grad():
            next_action1, next_p_action_logits1 = self.actor1_target(next_state)
            next_action2, next_p_action_logits2 = self.actor2_target(next_state)
            next_p_action_id1 = self.actor1_target.sample_discrete(next_p_action_logits1,
                                                                   epsilon=0.0)  # Deterministic for target
            next_p_action_id2 = self.actor2_target.sample_discrete(next_p_action_logits2,
                                                                   epsilon=0.0)  # Deterministic for target

            next_p_action_onehot1 = F.one_hot(next_p_action_id1, self.p_action_dim).float()
            next_p_action_onehot2 = F.one_hot(next_p_action_id2, self.p_action_dim).float()

            noise = torch.randn(
                (action.shape[0], action.shape[1]),
                dtype=action.dtype, layout=action.layout, device=action.device
            ) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)

            next_action1 = (next_action1 + noise).clamp(-self.max_action, self.max_action)
            next_action2 = (next_action2 + noise).clamp(-self.max_action, self.max_action)

            next_Q1_a1 = self.critic1_target(next_state, next_action1, next_p_action_onehot1)
            next_Q2_a1 = self.critic2_target(next_state, next_action1, next_p_action_onehot1)

            next_Q1_a2 = self.critic1_target(next_state, next_action2, next_p_action_onehot2)
            next_Q2_a2 = self.critic2_target(next_state, next_action2, next_p_action_onehot2)
            ## min first, max afterward to avoid underestimation bias
            next_Q1 = torch.min(next_Q1_a1, next_Q2_a1)
            next_Q2 = torch.min(next_Q1_a2, next_Q2_a2)

            next_Q = torch.max(next_Q1, next_Q2)
            target_Q = reward + not_done * self.discount * next_Q

        p_action_onehot = F.one_hot(p_action, self.p_action_dim).float()
        if update_a1:
            current_Q1 = self.critic1(state, action, p_action_onehot)
            critic1_loss = F.mse_loss(current_Q1, target_Q)

            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            self.critic1_optimizer.step()

            joint_action, p_action_logits = self.actor1(state)
            p_action_probs = F.softmax(p_action_logits, dim=-1)
            p_action_id = self.actor1_target.sample_discrete(next_p_action_logits1,
                                                             epsilon=0.0)  # Deterministic for target
            p_action_onehot = F.one_hot(p_action_id, self.p_action_dim).float()
            actor1_loss = -self.critic1(state, joint_action, p_action_onehot).mean()

            self.actor1_optimizer.zero_grad()
            actor1_loss.backward()
            self.actor1_optimizer.step()

            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor1.parameters(), self.actor1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        else:
            current_Q2 = self.critic2(state, action, p_action_onehot)
            critic2_loss = F.mse_loss(current_Q2, target_Q)

            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            self.critic2_optimizer.step()

            joint_action, p_action_logits = self.actor2(state)
            p_action_probs = F.softmax(p_action_logits, dim=-1)
            p_action_id = self.actor2_target.sample_discrete(next_p_action_logits1,
                                                             epsilon=0.0)  # Deterministic for target
            p_action_onehot = F.one_hot(p_action_id, self.p_action_dim).float()
            actor2_loss = -self.critic2(state, joint_action, p_action_onehot).mean()

            self.actor2_optimizer.zero_grad()
            actor2_loss.backward()
            self.actor2_optimizer.step()

            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor2.parameters(), self.actor2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        self.update_step += 1

    def save(self):
        torch.save(self.critic1.state_dict(), self.chkpt_dir + "/_critic1")
        torch.save(self.critic1_optimizer.state_dict(), self.chkpt_dir + "/_critic1_optimizer")
        torch.save(self.actor1.state_dict(), self.chkpt_dir + "/_actor1")
        torch.save(self.actor1_optimizer.state_dict(), self.chkpt_dir + "/_actor1_optimizer")

        torch.save(self.critic2.state_dict(), self.chkpt_dir + "/_critic2")
        torch.save(self.critic2_optimizer.state_dict(), self.chkpt_dir + "/_critic2_optimizer")
        torch.save(self.actor2.state_dict(), self.chkpt_dir + "/_actor2")
        torch.save(self.actor2_optimizer.state_dict(), self.chkpt_dir + "/_actor2_optimizer")

    def load(self):
        self.critic1.load_state_dict(torch.load(self.chkpt_dir + "/_critic1"))
        self.critic1_optimizer.load_state_dict(torch.load(self.chkpt_dir + "/_critic1_optimizer"))
        self.actor1.load_state_dict(torch.load(self.chkpt_dir + "/_actor1"))
        self.actor1_optimizer.load_state_dict(torch.load(self.chkpt_dir + "/_actor1_optimizer"))

        self.critic2.load_state_dict(torch.load(self.chkpt_dir + "/_critic2"))
        self.critic2_optimizer.load_state_dict(torch.load(self.chkpt_dir + "/_critic2_optimizer"))
        self.actor2.load_state_dict(torch.load(self.chkpt_dir + "/_actor2"))
        self.actor2_optimizer.load_state_dict(torch.load(self.chkpt_dir + "/_actor2_optimizer"))