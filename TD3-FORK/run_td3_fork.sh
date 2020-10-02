#!/bin/bash

# Script to reproduce results

for ((i=0;i<5;i+=1))
do
	python main_td3_fork.py \
	--policy "TD3_FORK" \
	--env "Ant-v3" \
	--seed $i \
	--base_weight  0.6\
	--sys_threshold 0.15 \
	--batch_size 100 \
	--sys_dynamic_weight True \
	--max_reward 6200 \
	--save_model True

	python main_td3_fork.py \
	--policy "TD3_FORK" \
	--env "HalfCheetah-v3" \
	--seed $i \
	--base_weight  0.6\
	--sys_threshold 0.20 \
	--batch_size 100 \
	--sys_dynamic_weight True \
	--max_reward 12000 \
	--save_model True

	python main_td3_fork.py \
	--policy "TD3_FORK" \
	--env "Hopper-v3" \
	--seed $i \
	--base_weight  0.6\
	--sys_threshold 0.0020 \
	--batch_size 100 \
	--sys_dynamic_weight True \
	--max_reward 3800 \
	--save_model True

	python main_td3_fork.py \
	--policy "TD3_FORK" \
	--env "Walker2d-v3" \
	--seed $i \
	--base_weight  0.6\
	--sys_threshold 0.15 \
	--batch_size 100 \
	--sys_dynamic_weight True \
	--max_reward 4500 \
	--save_model True

	python main_td3_fork.py \
	--policy "TD3_FORK" \
	--env "BipedalWalker-v3" \
	--seed $i \
	--base_weight  0.6\
	--sys_threshold 0.010 \
	--batch_size 100 \
	--sys_dynamic_weight True \
	--max_reward 320 \
	--save_model True

	python main_td3_fork.py \
	--policy "TD3_FORK" \
	--env "Humanoid-v3" \
	--seed $i \
	--base_weight  0.6\
	--sys_threshold 0.20 \
	--batch_size 100 \
	--sys_dynamic_weight True \
	--max_reward 5200 \
	--save_model True
done
