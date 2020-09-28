import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork, DeterministicPolicy, Sys_R, SysModel


class SAC_FORK(object):
    def __init__(self, num_inputs, action_space, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy_type
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.sysmodel = SysModel(num_inputs, action_space.shape[0], args.sys_hidden_size,args.sys_hidden_size).to(self.device)
        self.sysmodel_optimizer = Adam(self.sysmodel.parameters(), lr=args.lr)

        self.obs_upper_bound = 0 #state space upper bound
        self.obs_lower_bound = 0  #state space lower bound

        self.sysr = Sys_R(num_inputs, action_space.shape[0],args.sysr_hidden_size,args.sysr_hidden_size).to(self.device)
        self.sysr_optimizer = torch.optim.Adam(self.sysr.parameters(), lr=args.lr)

        self.sys_threshold = args.sys_threshold
        self.sys_weight = args.sys_weight
        self.sysmodel_loss = 0
        self.sysr_loss = 0

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()


        predict_next_state = self.sysmodel(state_batch, action_batch)
        predict_next_state = predict_next_state.clamp(self.obs_lower_bound,self.obs_upper_bound)
        sysmodel_loss = F.smooth_l1_loss(predict_next_state, next_state_batch.detach())
        self.sysmodel_optimizer.zero_grad()
        sysmodel_loss.backward()
        self.sysmodel_optimizer.step()
        self.sysmodel_loss = sysmodel_loss.item()

        predict_reward = self.sysr(state_batch,next_state_batch,action_batch)
        sysr_loss = F.mse_loss(predict_reward, reward_batch.detach())
        self.sysr_optimizer.zero_grad()
        sysr_loss.backward()
        self.sysr_optimizer.step()
        self.sysr_loss = sysr_loss.item()

        s_flag = 1 if sysmodel_loss.item() < self.sys_threshold else 0

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        if s_flag == 1 and self.sys_weight != 0:
            p_next_state = self.sysmodel(state_batch,pi)
            p_next_state = p_next_state.clamp(self.obs_lower_bound,self.obs_upper_bound)
            p_next_r = self.sysr(state_batch,p_next_state.detach(),pi)

            pi2, log_pi2, _ = self.policy.sample(p_next_state.detach())
            p_next_state2 = self.sysmodel(p_next_state,pi2)
            p_next_state2 = p_next_state2.clamp(self.obs_lower_bound,self.obs_upper_bound)
            p_next_r2 = self.sysr(p_next_state.detach(),p_next_state2.detach(),pi2)

            pi3, log_pi3, _ = self.policy.sample(p_next_state2.detach())
            qf3_pi, qf4_pi = self.critic(p_next_state2.detach(), pi3)
            min_qf_pi2 = torch.min(qf3_pi, qf4_pi)
            sys_loss = -(p_next_r + self.gamma * p_next_r2 + self.gamma ** 2 * min_qf_pi2).mean()
            policy_loss += self.sys_weight * sys_loss
            self.update_sys += 1

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()


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

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optim.state_dict(), filename + "_critic_optimizer")

        torch.save(self.policy.state_dict(), filename + "_actor")
        torch.save(self.policy_optim.state_dict(), filename + "_actor_optimizer")

        torch.save(self.sysmodel.state_dict(), filename + "_sysmodel")
        torch.save(self.sysmodel_optimizer.state_dict(), filename + "_sysmodel_optimizer")

        torch.save(self.sysr.state_dict(), filename + "_reward_model")
        torch.save(self.sysr_optimizer.state_dict(), filename + "_reward_model_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic.pth"))
        self.critic_optim.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.policy.load_state_dict(torch.load(filename + "_actor.pth"))
        self.policy_optim.load_state_dict(torch.load(filename + "_actor_optimizer"))

        self.sysmodel.load_state_dict(torch.load(filename + "_sysmodel.pth"))
        relf.sysmodel_optimizer.load_state_dict(torch.load(filename + "_sysmodel_optimizer"))

        self.sysr.load_state_dict(torch.load(filename + "_reward_model.pth"))
        relf.sysr_optimizer.load_state_dict(torch.load(filename + "_reward_model_optimizer"))
