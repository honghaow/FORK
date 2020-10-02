import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)

		self.max_action = max_action


	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1

class Sys_R(nn.Module):
	def __init__(self,state_dim, action_dim, fc1_units, fc2_units):
		super(Sys_R, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(2 * state_dim + action_dim, fc1_units)
		self.l2 = nn.Linear(fc1_units,fc2_units)
		self.l3 = nn.Linear(fc2_units, 1)


	def forward(self, state,next_state, action):
		sa = torch.cat([state,next_state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class SysModel(nn.Module):
	def __init__(self, state_size, action_size, fc1_units, fc2_units):
		super(SysModel, self).__init__()
		self.l1 = nn.Linear(state_size + action_size, fc1_units)
		self.l2 = nn.Linear(fc1_units, fc2_units)
		self.l3 = nn.Linear(fc2_units, state_size)

	def forward(self, state, action):
		"""Build a system model to predict the next state at a given state."""
		xa = torch.cat([state, action], 1)

		x1 = F.relu(self.l1(xa))
		x1 = F.relu(self.l2(x1))
		x1 = self.l3(x1)
		return x1


class TD3_FORK(object):
	def __init__(
		self,
		env,
		policy,
		state_dim,
		action_dim,
		max_action,
		sys1_units = 400,
		sys2_units = 300,
		r1_units = 256,
		r2_units = 256,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		sys_weight = 0.5,
		sys_weight2 = 0.4,
		sys_threshold = 0.020,
	):

		self.env = env
		self.policy = policy

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)


		self.sysmodel = SysModel(state_dim, action_dim, sys1_units,sys2_units).to(device)
		self.sysmodel_optimizer = torch.optim.Adam(self.sysmodel.parameters(), lr=3e-4)
		self.sysmodel.apply(self.init_weights)

		self.sysr = Sys_R(state_dim, action_dim, r1_units, r2_units).to(device)
		self.sysr_optimizer = torch.optim.Adam(self.sysr.parameters(), lr=3e-4)

		self.obs_upper_bound = float(self.env.observation_space.high[0]) #state space upper bound
		self.obs_lower_bound = float(self.env.observation_space.low[0])  #state space lower bound

		self.reward_lower_bound = 0
		self.reward_upper_bound = 0

		if self.obs_upper_bound == float('inf'):
			self.obs_upper_bound,self.obs_lower_bound = 0,0

		self.sysmodel_loss = 0
		self.sysr_loss = 0

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq
		self.sys_weight = sys_weight
		self.sys_weight2 = sys_weight2
		self.sys_threshold = sys_threshold

		self.total_it = 0


	def init_weights(self,m):
		if type(m) == nn.Linear:
			torch.nn.init.xavier_uniform_(m.weight)
			m.bias.data.fill_(0.001)


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size=100,train_steps=1):
		for _ in range(train_steps):
			self.total_it += 1

			# Sample replay buffer
			state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

			with torch.no_grad():
				# Select action according to policy and add clipped noise
				noise = (
					torch.randn_like(action) * self.policy_noise
					).clamp(-self.noise_clip, self.noise_clip)

				next_action = (
					self.actor_target(next_state) + noise
				).clamp(-self.max_action, self.max_action)

				# Compute the target Q value
				target_Q1, target_Q2 = self.critic_target(next_state, next_action)
				target_Q = torch.min(target_Q1, target_Q2)
				target_Q = reward + not_done * self.discount * target_Q

			# Get current Q estimates
			current_Q1, current_Q2 = self.critic(state, action)

			# Compute critic loss
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

			# Optimize the critic
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()


			#Train system and reward model

			predict_next_state = self.sysmodel(state, action)
			predict_next_state = predict_next_state.clamp(self.obs_lower_bound,self.obs_upper_bound)
			sysmodel_loss = F.smooth_l1_loss(predict_next_state, next_state.detach())

			self.sysmodel_optimizer.zero_grad()
			sysmodel_loss.backward()
			self.sysmodel_optimizer.step()
			self.sysmodel_loss = sysmodel_loss.item()


			predict_reward = self.sysr(state,next_state,action)
			sysr_loss = F.mse_loss(predict_reward, reward.detach())
			self.sysr_optimizer.zero_grad()
			sysr_loss.backward()
			self.sysr_optimizer.step()
			self.sysr_loss = sysr_loss.item()

			s_flag = 1 if sysmodel_loss.item() < self.sys_threshold else 0

			#Delayed policy updates
			if self.total_it % self.policy_freq == 0:

				#Compute actor losse
				actor_loss1 = -self.critic.Q1(state, self.actor(state)).mean()

				if s_flag == 1:
					p_next_state = self.sysmodel(state, self.actor(state))
					p_next_state = p_next_state.clamp(self.obs_lower_bound,self.obs_upper_bound)
					actions2 = self.actor(p_next_state.detach())

					if self.policy in ['TD3_FORK_Q','TD3_FORK_Q_F','TD3_FORK_DQ','TD3_FORK_DQ_F']:
						actor_loss2 =  self.critic.Q1(p_next_state.detach(),actions2)

						if self.policy in ['TD3_FORK_DQ','TD3_FORK_DQ_F']:
							p_next_state2 = self.sysmodel(p_next_state, self.actor(p_next_state.detach()))
							p_next_state2 = p_next_state2.clamp(self.obs_lower_bound,self.obs_upper_bound)
							actions3 = self.actor(p_next_state2.detach())
							actor_loss22 =  self.critic.Q1(p_next_state2.detach(),actions3)
							actor_loss3 =  - actor_loss2.mean() - self.sys_weight2 * actor_loss22.mean()
						else:
							actor_loss3 =  - actor_loss2.mean()

					elif self.policy in ['TD3_FORK_S','TD3_FORK_S_F','TD3_FORK','TD3_FORK_F']:
						p_next_r = self.sysr(state,p_next_state.detach(),self.actor(state))
						if self.policy in ['TD3_FORK_S','TD3_FORK_S_F']:
							actor_loss2 =  self.critic.Q1(p_next_state.detach(),actions2)
							actor_loss3 =  -(p_next_r + self.discount * actor_loss2).mean()
						else:

							p_next_state2 = self.sysmodel(p_next_state, self.actor(p_next_state.detach()))
							p_next_state2 = p_next_state2.clamp(self.obs_lower_bound,self.obs_upper_bound)
							p_next_r2 = self.sysr(p_next_state.detach(),p_next_state2.detach(),self.actor(p_next_state.detach()))
							actions3 = self.actor(p_next_state2.detach())

							actor_loss2 =  self.critic.Q1(p_next_state2.detach(),actions3)
							actor_loss3 =  -(p_next_r + self.discount * p_next_r2 + self.discount ** 2 * actor_loss2).mean()
					actor_loss =   (actor_loss1 + self.sys_weight * actor_loss3)
					self.update_sys += 1
				else:
					actor_loss = actor_loss1

				# Optimize the actor
				self.critic_optimizer.zero_grad()
				self.sysmodel_optimizer.zero_grad()
				self.actor_optimizer.zero_grad()
				actor_loss.backward()
				self.actor_optimizer.step()

				# Update the frozen target models
				for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

				for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

		torch.save(self.sysmodel.state_dict(), filename + "_sysmodel")
		torch.save(self.sysmodel_optimizer.state_dict(), filename + "_sysmodel_optimizer")

		torch.save(self.sysr.state_dict(), filename + "_reward_model")
		torch.save(self.sysr_optimizer.state_dict(), filename + "_reward_model_optimizer")

	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic.pth"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor.pth"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)

		self.sysmodel.load_state_dict(torch.load(filename + "_sysmodel.pth"))
		relf.sysmodel_optimizer.load_state_dict(torch.load(filename + "_sysmodel_optimizer"))

		self.sysr.load_state_dict(torch.load(filename + "_reward_model.pth"))
		relf.sysr_optimizer.load_state_dict(torch.load(filename + "_reward_model_optimizer"))
