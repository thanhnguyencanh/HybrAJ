import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np
import shutil

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON = 1e-6

def path_init(file_path):
    dir_path = os.path.dirname(file_path)
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, n_actions=None, p_action_dim=2,
            name='critic', chkpt_dir='ckpt/sac', lr=None):
        super(CriticNetwork, self).__init__()
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.q = nn.Sequential(
            nn.Linear(input_dims + n_actions, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        self.q2 = nn.Sequential(
            nn.Linear(input_dims + n_actions, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        path_init(self.checkpoint_file)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.apply(weights_init_)

    def forward(self, state, action):
        combo = T.cat([state, action], dim=1)
        q = self.q(combo)
        q2 = self.q2(combo)

        return q, q2

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, action_space=None, n_actions=2, p_action_dim=2, name='actor', chkpt_dir='ckpt/sac', lr=None):
        super(ActorNetwork, self).__init__()
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.reparam_noise = 1e-6
        self.fc1 = nn.Linear(input_dims, 512)
        self.fc2 = nn.Linear(512, 512)

        self.mu = nn.Linear(512, n_actions)
        self.sigma = nn.Linear(512, n_actions)
        self.p_action = nn.Linear(512, p_action_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        path_init(self.checkpoint_file)
        self.apply(weights_init_)

        # action rescaling (based on max action)
        if action_space is None:
            self.action_scale = T.tensor(1.).to(self.device)
            self.action_bias = T.tensor(0.).to(self.device)
        else:
            self.action_scale = T.FloatTensor((action_space.high - action_space.low) / 2).to(self.device)
            self.action_bias = T.FloatTensor((action_space.high + action_space.low) / 2).to(self.device)

    def forward(self, state):
        prob = F.relu(self.fc1(state))
        prob = F.relu(self.fc2(prob))
        p_action_logits = self.p_action(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)
        sigma = T.clamp(sigma, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return mu, sigma, p_action_logits

    def sample_normal(self, state, reparametrization=True):
        prob = F.relu(self.fc1(state))
        prob = F.relu(self.fc2(prob))
        p_action_logits = self.p_action(prob)
        dist_d = T.distributions.Categorical(logits=p_action_logits)
        p_action_id = dist_d.sample()
        prob_d = dist_d.probs
        log_prob_d = T.log(prob_d + 1e-8)

        mu = self.mu(prob)
        sigma = self.sigma(prob)
        sigma = T.clamp(sigma, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        probabilities = Normal(mu, sigma.exp())
        if reparametrization:
            actions = probabilities.rsample()  # random but learnable
        else:
            actions = probabilities.sample()
        action = T.tanh(actions) * self.action_scale + self.action_bias
        log_prob = probabilities.log_prob(actions)
        log_prob -= T.log(self.action_scale * (1 - action.pow(2)) + EPSILON)
        log_pi_c = log_prob.sum(-1, keepdim=True)
        mu = T.tanh(mu) * self.action_scale + self.action_bias

        return action, log_pi_c, mu, p_action_logits, log_prob_d, prob_d, p_action_id

    def sample_discrete(self, p_action_logits, epsilon=0.2):
        p_action_probs = F.softmax(p_action_logits, dim=-1)
        batch_size = p_action_logits.shape[0]
        if np.random.random() < epsilon:
            return T.randint(0, p_action_logits.shape[-1], (batch_size, 1), device=self.device)
        dist = T.distributions.Categorical(probs=p_action_probs)
        return dist.sample().unsqueeze(-1)  # [B,1]

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class Agent(object):
    def __init__(self, input_dim, action_space, p_action_dim, lr=0.0001, epsilon=1.0, eps_min=0.05, eps_dec=1e-5,
                 target_update_interval=1, alpha=0.02, tau=0.005, discount=0.99, writer=None, a_lr=0.0001, chkpt_dir=None):

        self.alpha = alpha
        self.alpha_d = alpha
        self.discount = discount
        self.tau = tau
        self.target_update_interval = target_update_interval
        self.writer = writer
        self.updates = 0
        self.eps = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.critic = CriticNetwork(input_dims=input_dim, n_actions=action_space.shape[0], p_action_dim=p_action_dim,
                                    lr=lr, name=f'critic', chkpt_dir=chkpt_dir)
        self.critic_target = CriticNetwork(input_dims=input_dim, n_actions=action_space.shape[0], p_action_dim=p_action_dim
                                           , lr=lr, name=f'critic_target')
        self.actor = ActorNetwork(input_dims=input_dim, n_actions=action_space.shape[0], action_space=action_space, p_action_dim=p_action_dim,
                                  name=f'actor', chkpt_dir=chkpt_dir, lr=lr)
        self.update_network_parameters(target=self.critic_target, source=self.critic, tau=1)
        # self.target_entropy = -T.prod(T.Tensor(action_space.shape[0]).to(self.actor.device)).item()
        self.p_action_dim = p_action_dim
        self.warmup = 20000
        self.update_step = 0
        self.target_entropy = -float(action_space.shape[0])
        self.target_entropy_d = -float(action_space.shape[0])
        self.log_alpha = T.zeros(1, requires_grad=True, device=self.actor.device)
        self.log_alpha_d = T.zeros(1, requires_grad=True, device=self.actor.device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=a_lr)
        self.alpha_optim_d = optim.Adam([self.log_alpha], lr=a_lr)

    def choose_action(self, observation, evaluate=False):
        state = T.FloatTensor(observation).to(self.actor.device).unsqueeze(0)
        if evaluate is False:
            actions, _, _, p_action_logits, log_pi_d, prob_d, p_action_id = self.actor.sample_normal(state, reparametrization=False)
        else:
            _, _, actions, p_action_logits, log_pi_d, prob_d, _ = self.actor.sample_normal(state, reparametrization=False)
            p_action_id = self.actor.sample_discrete(p_action_logits,
                                                     epsilon=self.eps if self.update_step < self.warmup else 0.0)
        return actions.detach().cpu().numpy()[0], p_action_id.cpu().data.numpy().flatten()

    def learn(self, memory, batch_size):
        state_batch, next_state_batch, action_batch, reward_batch, mask_batch = memory.sample(batch_size)
        state_batch = T.tensor(state_batch, dtype=T.float32, device=self.actor.device)
        next_state_batch = T.tensor(next_state_batch, dtype=T.float32, device=self.actor.device)
        action_batch = T.tensor(action_batch, dtype=T.float32, device=self.actor.device)
        reward_batch = T.tensor(reward_batch, dtype=T.float32, device=self.actor.device).unsqueeze(1)
        mask_batch = T.tensor(mask_batch, dtype=T.float32, device=self.actor.device).unsqueeze(1)

        # Split continuous and discrete parts
        action = action_batch[:, :-1]  # continuous part
        p_action = action_batch[:, -1].long()  # discrete part, safely converted to long

        # compute critic loss
        with T.no_grad():
            next_state_action, next_state_log_pi_c, _, next_p_action_logits, next_state_log_pi_d, next_prob_d, next_p_action_id = self.actor.sample_normal(next_state_batch,
                                                                               reparametrization=False)
            qf1_next_target, qf2_next_target = self.critic_target.forward(next_state_batch, next_state_action)

            # min_qf_next_target = T.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            min_qf_next_target = next_prob_d * (T.min(qf1_next_target, qf2_next_target) - self.alpha * next_prob_d * next_state_log_pi_c
                                                 - self.alpha_d * next_state_log_pi_d)
            target_Q = reward_batch + mask_batch * self.discount * (min_qf_next_target)

        qf1, qf2 = self.critic(state_batch, action)

        qf1_loss = F.mse_loss(qf1, target_Q)
        qf2_loss = F.mse_loss(qf2, target_Q)
        qf_loss = qf1_loss + qf2_loss

        self.critic.optimizer.zero_grad()
        qf_loss.backward()
        self.critic.optimizer.step()

        pi, log_pi_c, _, p_action_logits, log_pi_d, prob_d, p_action_id = self.actor.sample_normal(state_batch, reparametrization=True)
        qf1_pi, qf2_pi = self.critic.forward(state_batch, pi)
        min_qf_pi = T.min(qf1_pi, qf2_pi)

        policy_loss_d = (prob_d * (self.alpha_d * log_pi_d - min_qf_pi)).sum(1).mean()
        policy_loss_c = (prob_d * (self.alpha * prob_d * log_pi_c - min_qf_pi)).sum(1).mean()
        policy_loss = policy_loss_d + policy_loss_c
        self.actor.optimizer.zero_grad()
        policy_loss.backward()
        self.actor.optimizer.step()

        alpha_loss = (-self.log_alpha * prob_d.detach() * (prob_d.detach() * log_pi_c.detach() + self.target_entropy)).sum(1).mean()
        alpha_d_loss = (-self.log_alpha_d * prob_d.detach() * (log_pi_d.detach() + self.target_entropy_d)).sum(1).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        alpha = self.log_alpha.exp().detach().cpu().item()

        self.alpha_optim_d.zero_grad()
        alpha_d_loss.backward()
        self.alpha_optim_d.step()
        alpha_d = self.log_alpha_d.exp().detach().cpu().item()

        if self.updates % self.target_update_interval == 0:
            # define soft update function
            self.update_network_parameters(target=self.critic_target, source=self.critic)

        self.updates += 1

    def update_network_parameters(self, target, source, tau=None):
        if tau is None:
            tau = self.tau
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def save(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()