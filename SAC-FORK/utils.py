import numpy as np
import torch
import math


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.welford_state_n = 1
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def normalize_state(self, states, update=True):
		"""
		Use Welford's algorithm to normalize a state, and optionally update the statistics
		for normalizing states using the new state, online.
		"""
		states = torch.Tensor(states)
		states2 = states.data.clone()
		ii = 0
		for state in states:
			if self.welford_state_n == 1:
				self.welford_state_mean = torch.zeros(state.size(-1))
				self.welford_state_mean_diff = torch.ones(state.size(-1))

			if update:
				if len(state.size()) == 1: # if we get a single state vector
					state_old = self.welford_state_mean
					self.welford_state_mean += (state - state_old) / self.welford_state_n
					self.welford_state_mean_diff += (state - state_old) * (state - state_old)
					self.welford_state_n += 1
				else:
					raise RuntimeError # this really should not happen
			states2[ii] = (state - self.welford_state_mean) / np.sqrt(self.welford_state_mean_diff / self.welford_state_n)
			ii += 1
		return states2

	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0,int(self.size), size=batch_size)
		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			#torch.FloatTensor(self.normalize_state(self.state[ind])).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


def create_log_gaussian(mean, log_std, t):
	quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
	l = mean.shape
	log_z = log_std
	z = l[-1] * math.log(2 * math.pi)
	log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
	return log_p

def logsumexp(inputs, dim=None, keepdim=False):
	if dim is None:
		inputs = inputs.view(-1)
		dim = 0
	s, _ = torch.max(inputs, dim=dim, keepdim=True)
	outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
	if not keepdim:
		outputs = outputs.squeeze(dim)
	return outputs

def soft_update(target, source, tau):
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(param.data)
