import torch
import numpy as np
import torch
from torch.nn import Module, Linear
from torch.distributions import Distribution, Normal
from torch.nn.functional import relu, logsigmoid, softmax, one_hot
import os
import shutil
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def path_init(dir_path):
    # dir_path = os.path.dirname(file_path)
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

def quantile_huber_loss_f(quantiles, samples):
    pairwise_delta = samples[:, None, None, :] - quantiles[:, :, :, None]  # batch x nets x quantiles x samples
    abs_pairwise_delta = torch.abs(pairwise_delta)
    huber_loss = torch.where(abs_pairwise_delta > 1,
                             abs_pairwise_delta - 0.5,
                             pairwise_delta ** 2 * 0.5)

    n_quantiles = quantiles.shape[2]
    tau = torch.arange(n_quantiles, device=DEVICE).float() / n_quantiles + 1 / 2 / n_quantiles
    loss = (torch.abs(tau[None, None, :, None] - (pairwise_delta < 0).float()) * huber_loss).mean()
    return loss

LOG_STD_MIN_MAX = (-20, 2)

class MLP(Module):
    def __init__(
            self,
            input_size,
            hidden_sizes,
            output_size,
            p_action_size,
    ):
        super().__init__()
        self.fcs = []
        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = Linear(in_size, next_size)
            self.add_module(f'fc{i}', fc)
            self.fcs.append(fc)
            in_size = next_size
        self.last_fc = Linear(in_size, output_size)
        self.p_action = Linear(in_size, p_action_size)

    def forward(self, input):
        h = input
        for fc in self.fcs:
            h = relu(fc(h))
        output = self.last_fc(h)
        p_action_logits = self.p_action(h)
        return output, p_action_logits

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.transition_names = ('state', 'action', 'next_state', 'reward', 'not_done')
        sizes = (state_dim, action_dim, state_dim, 1, 1)
        for name, size in zip(self.transition_names, sizes):
            setattr(self, name, np.empty((max_size, size)))

    def pack_action(self, action):
        ac, ad = action
        return np.concatenate([ac, ad])

    def add(self, state, action, next_state, reward, done):
        values = (state, self.pack_action(action), next_state, reward, 1. - done)
        for name, value in zip(self.transition_names, values):
            getattr(self, name)[self.ptr] = value

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        names = self.transition_names
        return (torch.FloatTensor(getattr(self, name)[ind]).to(DEVICE) for name in names)

    def states_by_ptr(self, ptr_list, cpu=False):
        ind = np.array([], dtype='int64')
        for interval in ptr_list:
            if interval[0] < interval[1]:
                ind = np.concatenate((ind, np.arange(interval[0], interval[1])))
            elif interval[0] > interval[1]:
                ind = np.concatenate((ind, np.arange(interval[0], self.max_size)))
                ind = np.concatenate((ind, np.arange(0, interval[1])))

        names = ('state', 'action')
        if cpu:
            return (torch.FloatTensor(getattr(self, name)[ind]) for name in names)
        else:
            return (torch.FloatTensor(getattr(self, name)[ind]).to(DEVICE) for name in names)

class Critic(Module):
    def __init__(self, state_dim, action_dim, n_quantiles, n_nets, p_action_size):
        super().__init__()
        self.nets = []
        self.n_quantiles = n_quantiles
        self.n_nets = n_nets
        for i in range(n_nets):
            net = MLP(state_dim + action_dim + p_action_size, [512, 512, 512], n_quantiles, p_action_size)
            self.add_module(f'qf{i}', net)
            self.nets.append(net)

    def forward(self, state, action, p_action_id):
        if p_action_id.dim() == 1:
            p_action_id = p_action_id.unsqueeze(-1)  # (B, 1)
        sa = torch.cat((state, action, p_action_id), dim=1)
        quantiles = torch.stack(tuple(net(sa) for net in self.nets), dim=1)
        return quantiles  #output dim will be [batch, n_quantiles * n_nets]

class Actor(Module):
    def __init__(self, state_dim, action_dim, p_action_size):
        super().__init__()
        self.action_dim = action_dim
        self.net = MLP(state_dim, [256, 256], 2 * action_dim, p_action_size)

    def forward(self, obs):
        # mean, log_std, p_action_logits = self.net(obs).split([self.action_dim, self.action_dim], dim=1)
        cont_out, p_action_logits = self.net(obs)
        mean, log_std = cont_out.split(self.action_dim, dim=1)
        log_std = log_std.clamp(*LOG_STD_MIN_MAX)

        if self.training:
            std = torch.exp(log_std)
            tanh_normal = TanhNormal(mean, std)
            action, pre_tanh = tanh_normal.rsample()
            log_prob = tanh_normal.log_prob(pre_tanh)
            log_prob = log_prob.sum(dim=1, keepdim=True)
            dist_d = torch.distributions.Categorical(logits=p_action_logits)
            p_action_id = dist_d.sample()
            prob_d = dist_d.probs
            log_prob_d = torch.log(prob_d + 1e-8)

        else:  # deterministic eval without log_prob computation
            action = torch.tanh(mean)
            log_prob = None
            p_action_id = torch.argmax(p_action_logits, dim=-1)
            log_prob_d = None
            prob_d = None
        return action, log_prob, p_action_logits, log_prob_d, prob_d, p_action_id

    def choose_action(self, obs, evaluate=False):
        obs = torch.FloatTensor(obs).to(DEVICE)[None, :]
        action, mu, p_action_logits, log_prob_d, prob_d, p_action_id = self.forward(obs)
        if evaluate is False:
            action = action
        else:
            action = mu
            p_action_id = self.actor.sample_discrete(p_action_logits,
                                                     epsilon=self.eps if self.update_step < self.warmup else 0.0)
        action = action[0].cpu().detach().numpy()
        return action, p_action_id.cpu().detach().numpy()

class TanhNormal(Distribution):
    def __init__(self, normal_mean, normal_std):
        super().__init__()
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.standard_normal = Normal(torch.zeros_like(self.normal_mean, device=DEVICE),
                                      torch.ones_like(self.normal_std, device=DEVICE))
        self.normal = Normal(normal_mean, normal_std)

    def log_prob(self, pre_tanh):
        log_det = 2 * np.log(2) + logsigmoid(2 * pre_tanh) + logsigmoid(-2 * pre_tanh)
        result = self.normal.log_prob(pre_tanh) - log_det
        return result

    def rsample(self):
        pretanh = self.normal_mean + self.normal_std * self.standard_normal.sample()
        return torch.tanh(pretanh), pretanh

class Agent(object):
    def __init__(
            self,
            *,
            actor,
            critic,
            critic_target,
            discount,
            tau,
            top_quantiles_to_drop,
            target_entropy,
            use_acc,
            lr_dropped_quantiles,
            adjusted_dropped_quantiles_init,
            adjusted_dropped_quantiles_max,
            diff_ma_coef,
            num_critic_updates,
            p_action_size,
            alpha,
            writer,
            chkpt_dir
    ):

        self.actor = actor
        self.critic = critic
        self.critic_target = critic_target
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.discount = discount
        self.tau = tau
        self.top_quantiles_to_drop = top_quantiles_to_drop
        self.quantiles_total = critic.n_quantiles * critic.n_nets
        self.total_it = 0
        self.p_action_dim = p_action_size
        self.writer = writer
        self.chkpt_dir = chkpt_dir
        self.use_acc = use_acc
        self.num_critic_updates = num_critic_updates
        path_init(chkpt_dir)
        self.alpha = alpha
        self.alpha_d = alpha
        self.log_alpha = torch.zeros((1,), requires_grad=True, device=DEVICE)
        self.log_alpha_d = torch.zeros(1, requires_grad=True, device=DEVICE)
        self.target_entropy = target_entropy
        self.target_entropy_d = target_entropy
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=3e-4)
        self.alpha_optim_d = torch.optim.Adam([self.log_alpha_d], lr=3e-4)
        if use_acc:
            self.adjusted_dropped_quantiles = torch.tensor(adjusted_dropped_quantiles_init, requires_grad=True)
            self.adjusted_dropped_quantiles_max = adjusted_dropped_quantiles_max
            self.dropped_quantiles_dropped_optimizer = torch.optim.SGD([self.adjusted_dropped_quantiles], lr=lr_dropped_quantiles)
            self.first_training = True
            self.diff_ma_coef = diff_ma_coef

    def train(self, replay_buffer, batch_size=256, ptr_list=None, disc_return=None, do_beta_update=False):
        if ptr_list is not None and do_beta_update:
            self.update_beta(replay_buffer, ptr_list, disc_return)

        for it in range(self.num_critic_updates):
            state, actions, next_state, reward, not_done = replay_buffer.sample(batch_size)
            alpha = torch.exp(self.log_alpha)

            # Split continuous and discrete parts
            action = actions[:, :-1]  # continuous part
            p_action = actions[:, -1].long()  # discrete part, safely converted to long
            # --- Q loss ---
            with torch.no_grad():
                # get policy action
                new_next_action, next_state_log_pi_c, next_p_action_logits, next_state_log_pi_d, next_prob_d, next_p_action_id = self.actor(next_state)
                next_p_action_onehot = one_hot(next_p_action_id.squeeze(-1), self.p_action_dim).float()
                # compute and cut quantiles at the next state
                next_z = self.critic_target(next_state, new_next_action, next_p_action_onehot)  # batch x nets x quantiles
                sorted_z, _ = torch.sort(next_z.reshape(batch_size, -1))
                if self.use_acc:
                    sorted_z_part = sorted_z[:, :self.quantiles_total - round(self.critic.n_nets * self.adjusted_dropped_quantiles.item())]
                else:
                    sorted_z_part = sorted_z[:, :self.quantiles_total - self.top_quantiles_to_drop]

                # compute target
                min_next_target = next_prob_d * (sorted_z_part - self.alpha * next_state_log_pi_c - self.alpha_d * next_state_log_pi_d)
                # compute target
                target = reward + not_done * self.discount * min_next_target
            p_action_onehot = one_hot(p_action.squeeze(-1), self.p_action_dim).float()
            cur_z = self.critic(state, action, p_action_onehot)
            critic_loss = quantile_huber_loss_f(cur_z, target.detach())

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # --- Policy and alpha loss ---
        new_action, log_pi_c, _, p_action_logits, log_pi_d, prob_d, p_action_id = self.actor(state)
        disc_onehot = one_hot(p_action_id.squeeze(-1), self.p_action_dim).float()
        # dist_d = torch.distributions.Categorical(logits=p_action_logits)
        # log_pi_d = dist_d.log_prob(p_action_id.squeeze(-1)).unsqueeze(-1)  # log(dist_d(if p_action_id:))
        # log_pi_joint = log_pi + log_pi_d

        actor_loss_c = (prob_d * (self.alpha * log_pi_c - self.critic(state, new_action, disc_onehot))).sum(1).mean()
        actor_loss_d = (prob_d * (self.alpha_d * log_pi_d - self.critic(state, new_action, disc_onehot))).sum(1).mean()
        actor_loss = actor_loss_d + actor_loss_c

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = (-self.log_alpha * prob_d.detach() * (log_pi_c.detach() + self.target_entropy)).sum(1).mean()
        alpha_d_loss = (-self.log_alpha_d * prob_d.detach() * (log_pi_d.detach() + self.target_entropy_d)).sum(1).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        alpha = self.log_alpha.exp()

        self.alpha_optim_d.zero_grad()
        alpha_d_loss.backward()
        self.alpha_optim_d.step()
        alpha_d = self.log_alpha_d.exp()

        self.total_it += 1

        if self.total_it % 1000 == 0:
            self.writer.add_scalar('learner/critic_loss', critic_loss.detach().cpu().numpy(), self.total_it)
            self.writer.add_scalar('learner/actor_loss', actor_loss.detach().cpu().numpy(), self.total_it)
            self.writer.add_scalar('learner/alpha_loss', alpha_loss.detach().cpu().numpy(), self.total_it)
            self.writer.add_scalar('learner/alpha', alpha.detach().cpu().numpy(), self.total_it)
            self.writer.add_scalar('learner/Q_estimate', cur_z.mean().detach().cpu().numpy(), self.total_it)  # Q of current state
            self.writer.add_scalar('learner/target_Q_estimate', target.mean().detach().cpu().numpy(), self.total_it)  # Q of current state

    def update_beta(self, replay_buffer, ptr_list=None, disc_return=None):
        state, actions = replay_buffer.states_by_ptr(ptr_list)
        disc_return = torch.FloatTensor(disc_return).to(DEVICE)
        action = actions[:, :-1]  # continuous part
        p_action = actions[:, -1].long()  # discrete part, safely converted to long
        p_action_onehot = one_hot(p_action.squeeze(-1), self.p_action_dim).float()
        assert disc_return.shape[0] == state.shape[0]

        mean_Q_last_eps = self.critic(state, action, p_action_onehot).mean(2).mean(1, keepdim=True).mean().detach()
        mean_return_last_eps = torch.mean(disc_return).detach()

        if self.first_training:
            self.diff_mvavg = torch.abs(mean_return_last_eps - mean_Q_last_eps).detach()
            self.first_training = False
        else:
            self.diff_mvavg = (1 - self.diff_ma_coef) * self.diff_mvavg \
                              + self.diff_ma_coef * torch.abs(mean_return_last_eps - mean_Q_last_eps).detach()

        diff_qret = ((mean_return_last_eps - mean_Q_last_eps) / (self.diff_mvavg + 1e-8)).detach()
        aux_loss = self.adjusted_dropped_quantiles * diff_qret
        self.dropped_quantiles_dropped_optimizer.zero_grad()
        aux_loss.backward()
        self.dropped_quantiles_dropped_optimizer.step()
        self.adjusted_dropped_quantiles.data = self.adjusted_dropped_quantiles.clamp(min=0., max=self.adjusted_dropped_quantiles_max)

        self.writer.add_scalar('learner/adjusted_dropped_quantiles', self.adjusted_dropped_quantiles, self.total_it)

    def save(self):
        torch.save(self.critic.state_dict(), self.chkpt_dir + "/_critic")
        torch.save(self.critic_target.state_dict(), self.chkpt_dir + "/_critic_target")
        torch.save(self.critic_optimizer.state_dict(), self.chkpt_dir + "/_critic_optimizer")
        torch.save(self.actor.state_dict(), self.chkpt_dir + "/_actor")
        torch.save(self.actor_optimizer.state_dict(), self.chkpt_dir + "/_actor_optimizer")
        torch.save(self.log_alpha, self.chkpt_dir + "/_log_alpha")
        torch.save(self.log_alpha_d, self.chkpt_dir + "/_log_alpha_d")
        torch.save(self.alpha_optim.state_dict(), self.chkpt_dir + "/_alpha_optimizer")
        torch.save(self.alpha_optim_d.state_dict(), self.chkpt_dir + "/_alpha_optimizer_d")

    def load(self):
        self.critic.load_state_dict(torch.load(self.chkpt_dir + "/_critic"))
        self.critic_target.load_state_dict(torch.load(self.chkpt_dir + "/_critic_target"))
        self.critic_optimizer.load_state_dict(torch.load(self.chkpt_dir + "/_critic_optimizer"))
        self.actor.load_state_dict(torch.load(self.chkpt_dir + "/_actor"))
        self.actor_optimizer.load_state_dict(torch.load(self.chkpt_dir + "/_actor_optimizer"))
        self.log_alpha = torch.load(self.chkpt_dir + "/_log_alpha")
        self.log_alpha_d = torch.load(self.chkpt_dir + "/_log_alpha_d")
        self.alpha_optim.load_state_dict(torch.load(self.chkpt_dir + "/_alpha_optimizer"))
        self.alpha_optim_d.load_state_dict(torch.load(self.chkpt_dir + "/_alpha_optimizer_d"))
