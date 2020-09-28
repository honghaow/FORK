import numpy as np
import torch
import gym
import argparse
import os
import copy
import utils
import TD3
import pandas as pd
import json,os
import TD3_FORK

def eval_policy(policy, env_name,eval_episodes=10):
	eval_env = gym.make(env_name)
	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes
	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="TD3")                      # Policy name (TD3,or TD3_FORK)
	parser.add_argument("--env", default="HalfCheetah-v2")              # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)                  # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=1e4, type=int)     # Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5e3, type=int)           # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)       # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1)                    # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=100, type=int)          # Batch size for both actor and critic
	parser.add_argument("--max_reward", default=100, type=int)          # max_reward for dynamic weight
	parser.add_argument("--discount", default=0.99)                     # Discount factor
	parser.add_argument("--tau", default=0.005)                         # Target network update rate
	parser.add_argument("--policy_noise", default=0.2,type=float)       # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5,type=float)         # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)           # Frequency of delayed policy updates
	parser.add_argument("--sys_neurons1", default=400, type=int)        #units of the first layer in system model
	parser.add_argument("--sys_neurons2", default=300, type=int)        #units of the second layer in system model
	parser.add_argument("--r_neurons1", default=256, type=int)          #units of the first layer in reward model
	parser.add_argument("--r_neurons2", default=256, type=int)          #units of the second layer in reward model
	parser.add_argument("--save_model", default="False")                # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                     # Model load file name, "" doesn't load, "default" uses file_name
	parser.add_argument("--training_mode", default="Online")            #training_mode Offline or Online
	parser.add_argument("--sys_weight", default=0.5,type=float)         # weight for FORK
	parser.add_argument("--base_weight", default=0.6,type=float)        # base weight if using dynamic_weight
	parser.add_argument("--sys_threshold", default=0.020,type=float)    # threshold for FORK
	parser.add_argument("--sys_dynamic_weight", default="False")        # whether use dynamic weight or not
	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_{args.seed}_{args.training_mode}"
	if args.sys_dynamic_weight == 'True':
		file_name += f"_DW_{args.sys_dynamic_weight}"

	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}, Weight: {args.sys_weight},Training_mode: {args.training_mode}, Dynamic_weight: {args.sys_dynamic_weight}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model == "True" and not os.path.exists("./models"):
		os.makedirs("./models")

	env = gym.make(args.env)
	# Set seeds
	env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	state_dim = env.observation_space.shape[0]
	state_max = env.observation_space.shape
	action_dim = env.action_space.shape[0]
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

	# Initialize policy
	if args.policy == "TD3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		policy = TD3.TD3(**kwargs)
		variant = dict(
			algorithm='TD3',
			env=args.env,
		)
	elif args.policy == "TD3_FORK":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["env"] = env
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		kwargs["sys_weight"] = args.sys_weight
		kwargs["sys_threshold"] = args.sys_threshold
		kwargs["sys1_units"] = args.sys_neurons1
		kwargs["sys2_units"] = args.sys_neurons2
		kwargs["r1_units"] = args.r_neurons1
		kwargs["r2_units"] = args.r_neurons2
		policy = TD3_FORK.TD3_FORK(**kwargs)

		variant = dict(
			algorithm='TD3_FORK',
			env=args.env,
			sys_weight=args.sys_weight,
			sys_threshold=args.sys_threshold,
			max_reward=args.max_reward,
			sys1_units=args.sys_neurons1,
			sys2_units=args.sys_neurons2,
			r1_units=args.r_neurons1,
			r2_units=args.r_neurons2
		)
	if not os.path.exists(f"./data/{args.env}/{args.policy}/seed{args.seed}"):
		os.makedirs(f'./data/{args.env}/{args.policy}/seed{args.seed}')
	with open(f'./data/{args.env}/{args.policy}/seed{int(args.seed)}/variant.json', 'w') as outfile:
		json.dump(variant,outfile)

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	# Evaluate untrained policy
	evaluations = [eval_policy(policy, args.env)]

	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0
	policy.update_sys = 0  #monitoring how many updated times of FORK
	ep_reward_list = []
	base_weight = args.base_weight

	for t in range(int(args.max_timesteps)):
		episode_timesteps += 1
		# Select action randomly or according to policy
		if t < args.start_timesteps:
			action = env.action_space.sample()
		else:
			action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)
		# Perform action
		next_state, reward, done, _ = env.step(action)
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0
		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state
		# Store observation and reward bounds
		policy.obs_upper_bound = np.amax(state) if policy.obs_upper_bound < np.amax(state) else policy.obs_upper_bound
		policy.obs_lower_bound = np.amin(state) if policy.obs_lower_bound > np.amin(state) else policy.obs_lower_bound
		policy.reward_lower_bound = (reward) if policy.reward_lower_bound > reward else policy.reward_lower_bound
		policy.reward_upper_bound = (reward) if policy.reward_upper_bound < reward else policy.reward_upper_bound

		episode_reward += reward
		# Train agent after collecting sufficient data
		if args.training_mode == 'Online':
			if t >= args.start_timesteps:
				policy.train(replay_buffer, args.batch_size,train_steps = 1)
		if done:
			ep_reward_list.append(episode_reward)
			if args.sys_dynamic_weight == "True":
				policy.sys_weight =  np.round((1 - np.clip(np.mean(ep_reward_list[-100:])/args.max_reward, 0, 1)),4) * base_weight
			if args.policy == "TD3_FORK":
				print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} Sysmodel_Loss: {policy.sysmodel_loss} Reward_loss: {policy.sysr_loss} Sys updated times: {policy.update_sys} Sys_weight: {policy.sys_weight}")
				policy.update_sys = 0
			else:
				 print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			if args.training_mode == 'Offline':
				if t >= args.start_timesteps:
					policy.train(replay_buffer, args.batch_size,train_steps = episode_timesteps)
			# Reset environment
			state, done = env.reset(), False

			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			evaluations.append(eval_policy(policy, args.env))
			if args.save_model == "True":
				policy.save(f"./models/{file_name}")

			data = np.array(evaluations)
			df = pd.DataFrame(data=data,columns=["Average Return"]).reset_index()
			df['Timesteps'] = df['index'] * args.eval_freq
			df['env'] = args.env
			df['algorithm_name'] = args.policy
			df.to_csv(f'./data/{args.env}/{args.policy}/seed{args.seed}/progress.csv', index = False)
