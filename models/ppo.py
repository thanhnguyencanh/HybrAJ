import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import shutil

def path_init(file_path):
    dir_path = os.path.dirname(file_path)
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.prob_conts = []
        self.prob_discs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return np.array(self.states), \
               np.array(self.actions), \
               np.array(self.prob_conts), \
               np.array(self.prob_discs), \
               np.array(self.vals), \
               np.array(self.rewards), \
               np.array(self.dones), \
               batches

    def store_memory(self, state, action, prob_cont, prob_disc, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.prob_conts.append(prob_cont)
        self.prob_discs.append(prob_disc)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.prob_conts = []
        self.prob_discs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


class ActorNetwork(nn.Module):
    def __init__(self, action_dims, input_dims, alpha, p_action_dim=2,
                 fc1_dims=512, fc2_dims=512, chkpt_dir='tmp/ppo'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor')
        self.actor = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
        )
        self.p_action = nn.Linear(fc2_dims, p_action_dim)
        self.mu = nn.Linear(fc2_dims, action_dims)
        self.sigma = nn.Linear(fc2_dims, action_dims)
        path_init(self.checkpoint_file)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.actor(state)
        # discrete
        p_action_logits = self.p_action(x)
        dist_d = Categorical(logits=p_action_logits)

        # continuous (conditioned on discrete action later)
        mu = self.mu(x)
        sigma = T.clamp(self.sigma(x), -20, 2)
        sigma = sigma.exp()
        dist_c = Normal(mu, sigma)

        return dist_c, dist_d

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=512, fc2_dims=512,
                 chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic')
        self.critic = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )
        path_init(self.checkpoint_file)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        return self.critic(state)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent:
    def __init__(self, action_dims, input_dims, p_action_dim, gamma=0.99, alpha=1e-4, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=64, n_epochs=10, chkpt=None):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(action_dims, input_dims, alpha, p_action_dim, chkpt_dir=chkpt)
        self.critic = CriticNetwork(input_dims, alpha, chkpt_dir=chkpt)
        self.memory = PPOMemory(batch_size)

    def add(self, state, action, prob_cont, prob_disc, vals, reward, done):
        self.memory.store_memory(state, action, prob_cont, prob_disc, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        dist_c, dist_d = self.actor(state)
        value = self.critic(state)
        a_c = dist_c.sample()
        a_d = dist_d.sample()
        logp_c = dist_c.log_prob(a_c).sum(dim=-1)
        logp_d = dist_d.log_prob(a_d)
        action = T.cat([a_c, a_d.unsqueeze(-1)], dim=-1)

        return action.cpu().numpy()[0], (logp_c + logp_d).item(), value.item(), logp_c.item(), logp_d.item()

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_cont, old_prob_disc, vals_arr, \
            reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            action_cont, action_disc = action_arr[:, :-1], action_arr[:, -1]
            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * \
                                       (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float32).to(self.actor.device)
                logp_d_old = T.tensor(old_prob_disc[batch], dtype=T.float32).to(self.actor.device)
                logp_c_old = T.tensor(old_prob_cont[batch], dtype=T.float32).to(self.actor.device)
                action_cont_batch = T.tensor(action_cont[batch], dtype=T.float32).to(self.actor.device)
                action_disc_batch = T.tensor(action_disc[batch], dtype=T.long).to(self.actor.device)

                dist_c, dist_d = self.actor(states)
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                logp_c_new = dist_c.log_prob(action_cont_batch).sum(dim=-1)
                logp_d_new = dist_d.log_prob(action_disc_batch)

                logp_new = logp_d_new + logp_c_new
                logp_old = logp_d_old + logp_c_old

                prob_ratio = (logp_new - logp_old).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip,
                                                 1 + self.policy_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch].detach()
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()

