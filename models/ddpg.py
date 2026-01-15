import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import shutil

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
        p_action_logits = self.p_action(x)

        return self.max_action * torch.tanh(self.l3(x)), p_action_logits

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
        q = F.relu(self.l1(torch.cat([state, action, p_action_id], 1)))
        q = F.relu(self.l2(q))
        return self.l3(q)

class Agent(object):
    def __init__(self, state_dim, action_dim, max_action,
                 p_action_dim,
                 discount=0.99,
                 tau=0.005,
                 actor_lr=3e-4,
                 critic_lr=3e-4,
                 writer=None,
                 chkpt_dir=None,
                 epsilon=1.0,
                 eps_min=0.05,
                 eps_dec=1e-5):
        self.actor = Actor(state_dim, action_dim, max_action, p_action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.chkpt_dir = chkpt_dir
        self.discount = discount
        self.tau = tau
        self.eps = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.seed = 0
        self.update_step = 0
        self.writer = writer
        self.warmup = 20000
        self.action_dim = action_dim
        self.max_action = max_action
        self.p_action_dim = p_action_dim
        path_init(self.chkpt_dir)

    def choose_action(self, observation):
        # Local RNG for deterministic sequences
        rng = np.random.default_rng(self.seed)
        self.actor.eval()
        state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)
        joint_action, p_action_logits = self.actor(state)
        p_action_id = self.actor.sample_discrete(p_action_logits,
                                                 epsilon=self.eps if self.update_step < self.warmup else 0.0)
        if self.update_step < self.warmup:
            mu = torch.tensor(rng.uniform(-self.max_action, self.max_action, size=(self.action_dim,)),
                          dtype=torch.float32, device=device)
        else:
            mu = joint_action  # assuming actor returns [batch, action_dim]
        mu_prime = mu

        self.actor.train()
        self.seed += 1
        return mu_prime.cpu().data.numpy().flatten(), p_action_id.cpu().data.numpy().flatten()

    def decrement_epsilon(self):
        self.eps = max(self.eps_min, self.eps - self.eps_dec)  # Additive decay (eps_min=0.05, eps=1.0, eps_decay=0.0001)
        # self.eps = max(self.eps_min, self.eps*self.eps_dec)  # Multiplicative decay (eps_min=0.01, eps=1.0, eps_decay=0.995)

    def learn(self, replay_buffer, batch_size=100):
        # Sample replay buffer
        state, next_state, action, reward, done = replay_buffer.sample(batch_size)
        state = torch.tensor(state, dtype=torch.float32, device=device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=device)

        action, p_action = action[:, :-1], action[:, -1]
        action = torch.FloatTensor(action).to(device)
        p_action = torch.LongTensor(p_action).to(device)
        not_done = torch.FloatTensor(1 - done).to(device)
        reward = torch.FloatTensor(reward).to(device)

        # Compute the target Q value
        next_action, next_p_action_logits = self.actor_target(next_state)
        next_p_action_id = self.actor_target.sample_discrete(next_p_action_logits,
                                                             epsilon=0.0)  # Deterministic for target
        p_action_onehot = F.one_hot(p_action, self.p_action_dim).float()
        next_p_action_onehot = F.one_hot(next_p_action_id, self.p_action_dim).float()

        target_Q = self.critic_target(next_state, next_action, next_p_action_onehot)
        self.bias_dis = torch.mean(target_Q)
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic(state, action, p_action_onehot)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        joint_action, p_action_logits = self.actor(state)
        p_action_probs = F.softmax(p_action_logits, dim=-1)
        p_a_id = self.actor_target.sample_discrete(p_action_logits,
                                                   epsilon=0.0)  # Deterministic for target
        p_action_onehot = F.one_hot(p_a_id, self.p_action_dim).float()
        actor_loss = -self.critic(state, joint_action, p_action_onehot).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        if self.update_step % 2 == 0:
            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        self.update_step += 1

    def save(self):
        torch.save(self.critic.state_dict(), self.chkpt_dir + "/_critic")
        torch.save(self.critic_optimizer.state_dict(), self.chkpt_dir + "/_critic_optimizer")

        torch.save(self.actor.state_dict(), self.chkpt_dir + "/_actor")
        torch.save(self.actor_optimizer.state_dict(), self.chkpt_dir + "/_actor_optimizer")

    def load(self):
        self.critic.load_state_dict(torch.load(self.chkpt_dir + "/_critic"))
        self.critic_optimizer.load_state_dict(torch.load(self.chkpt_dir + "/_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(self.chkpt_dir + "/_actor"))
        self.actor_optimizer.load_state_dict(torch.load(self.chkpt_dir + "/_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)