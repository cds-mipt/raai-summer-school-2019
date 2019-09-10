import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from collections import defaultdict, deque, Counter
import gym_car_intersect
import numpy as np
import gym


# Models for SAC ===============================================================
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


# Convolution layer:
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ConvModel(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.state_dim = state_dim
        # convolution net:
        self.conv = nn.Sequential(nn.Conv2d(4, 32, kernel_size=(5, 5), stride=(2, 2)),
                                  nn.ELU(),
                                  nn.MaxPool2d(kernel_size=(3, 3)),
                                  nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1)),
                                  nn.ELU(),
                                  nn.MaxPool2d(kernel_size=(3, 3)),
                                  Flatten())
        self.size = np.prod(self.conv(torch.zeros(1, *self.state_dim)).shape[1:])

    def forward(self, x):
        x = self.conv(x)
        return x


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, num_options, hidden_dim, conv_layer=False):
        super(QNetwork, self).__init__()

        # Conv layer:
        self.conv_layer = conv_layer
        if conv_layer:
            self.conv = ConvModel(num_inputs)
            num_inputs = self.conv.size
        else:
            num_inputs = np.product(num_inputs)

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_options)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, num_options)

        self.apply(weights_init_)

    def forward(self, state, action):

        if self.conv_layer:
            state = self.conv(state)

        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, num_options, hidden_dim, conv_layer=False):
        super(ValueNetwork, self).__init__()

        # Conv layer:
        self.conv_layer = conv_layer
        if conv_layer:
            self.conv = ConvModel(num_inputs)
            num_inputs = self.conv.size
        else:
            num_inputs = np.product(num_inputs)

        # Beta for options
        self.beta = nn.Sequential(nn.Linear(num_inputs, hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(hidden_dim, hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(hidden_dim, num_options),
                                  nn.Sigmoid())

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        if self.conv_layer:
            state = self.conv(state)

        beta = self.beta(state)

        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x, beta


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, num_options, hidden_dim, action_space=None, conv_layer=False):
        super(GaussianPolicy, self).__init__()
        self.num_actions = num_actions
        self.num_options = num_options

        # Conv layer:
        self.conv_layer = conv_layer
        if conv_layer:
            self.conv = ConvModel(num_inputs)
            num_inputs = self.conv.size
        else:
            num_inputs = np.product(num_inputs)

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions*num_options)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions*num_options)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        if self.conv_layer:
            state = self.conv(state)

        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x).view(-1, self.num_options, self.num_actions)
        log_std = self.log_std_linear(x).view(-1, self.num_options, self.num_actions)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state, option):
        # make form for case of 1 batch
        self.batch = 1 if state.dim()==1 else state.shape[0]
        # sample policy based on options
        mean, log_std = self.forward(state)
        mean = mean[range(self.batch), option.squeeze()]
        log_std = log_std[range(self.batch), option.squeeze()]
        std = log_std.exp()
        # create probability distribution
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mean)

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


# SOC agent ====================================================================
import os
from torch.optim import Adam
from sac_torch.utils import soft_update, hard_update


class SOC(object):
    def __init__(self, num_inputs, action_space, num_options, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.beta_reg = args.beta_reg

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.critic = QNetwork(num_inputs, action_space.shape[0], num_options, args.hidden_size, conv_layer=args.conv_layer).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], num_options, args.hidden_size, conv_layer=args.conv_layer).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.value = ValueNetwork(num_inputs, num_options, args.hidden_size, conv_layer=args.conv_layer).to(device=self.device)
        self.value_optim = Adam(self.value.parameters(), lr=args.lr)

        self.value_target = ValueNetwork(num_inputs, num_options, args.hidden_size, conv_layer=args.conv_layer).to(device=self.device)
        hard_update(self.value_target, self.value)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning == True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], num_options, args.hidden_size, action_space, conv_layer=args.conv_layer).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def options_soft_eps(self, state, eval=False):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        x, betas = self.value(state)
        if eval:
            options = x.argmax(dim=1)
        else:
            dist_critic = Categorical(logits=x)
            options = dist_critic.sample()
        return options[:,None], betas

    def choose_option(self, state, option, eval=False):
        soft_option, betas = self.options_soft_eps(state, eval)
        if option is None:
            new_option = soft_option
        else:
            if eval:
                mask = (betas.gather(1, option) > 0.5).long()
            else:
                mask = (betas.gather(1, option) > torch.rand(*option.size()).to(self.device)).long()
            new_option = (1-mask)*option + mask*soft_option
        return new_option

    def select_action(self, state, option, eval=False):
        if state.ndim < 2:
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            if eval == False:
                action, _, _ = self.policy.sample(state, option)
            else:
                _, _, action = self.policy.sample(state, option)
            action = action.detach().cpu().numpy()[0]
        else:
            state = torch.FloatTensor(state).to(self.device)
            if eval == False:
                action, _, _ = self.policy.sample(state, option)
            else:
                _, _, action = self.policy.sample(state, option)
            action = action.detach().cpu().numpy()
        return action

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch, option_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        option_batch = torch.LongTensor(option_batch).to(self.device)

        with torch.no_grad():
            next_option_batch = self.choose_option(state_batch, option_batch, eval=True)
            # next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch, next_option_batch)
            # qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            # min_qf_next_target = torch.min(qf1_next_target.gather(1, next_option_batch), qf2_next_target.gather(1, next_option_batch)) - self.alpha * next_state_log_pi
            # next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
            next_v_value, _ = self.value_target(next_state_batch)
            next_q_value = reward_batch + mask_batch * self.gamma * next_v_value.gather(1, next_option_batch) # considering value function

        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_o, qf2_o = qf1.gather(1, option_batch), qf2.gather(1, option_batch)
        qf1_loss = F.mse_loss(qf1_o, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2_o, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss # modified to use only one conv layer

        # Loss for policy
        pi, log_pi, _ = self.policy.sample(state_batch, option_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_o, qf2_o)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi.detach()).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        # Loss for value function:
        vf, beta = self.value(state_batch)
        vf_o = vf.gather(1, option_batch)
        v_value = torch.min(qf1_o, qf2_o) - self.alpha * log_pi # min Q(st,at) - α * logπ(f(εt;st)|st)
        vf_loss = F.mse_loss(vf_o, v_value.detach())

        # Now train loss for options:
        vf_target, _ = self.value_target(state_batch)
        adv_beta = vf_o - torch.max(vf_target.gather(1, option_batch), dim=1, keepdim=True)[0] + self.beta_reg
        beta_loss = (beta.gather(1, option_batch) * adv_beta.detach() * mask_batch).mean()
        value_loss = vf_loss + beta_loss

        # self.critic_optim.zero_grad()
        # qf1_loss.backward()
        # self.critic_optim.step()
        #
        # self.critic_optim.zero_grad()
        # qf2_loss.backward()
        # self.critic_optim.step()

        # modified to use only one conv layer:
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        self.value_optim.zero_grad()
        value_loss.backward()
        self.value_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
            soft_update(self.value_target, self.value, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), vf_loss.item(), beta_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
