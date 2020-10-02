#!/bin/bash

# Script to reproduce results

for ((i=0;i<5;i+=1))
do
	python main_sac_fork.py \
	--env "Hopper-v3" \
	--policy "SAC_FORK"\
	--seed $i \
	--automatic_entropy_tuning True \
	--batch_size 100 \
	--cuda \
	--sys_threshold 0.0020 \
	--sys_dynamic_weight True \
	--max_reward 2500 \
	--base_weight 0.1

	python main_sac_fork.py \
	--env "Ant-v3" \
	--policy "SAC_FORK"\
	--seed $i \
	--automatic_entropy_tuning True \
	--batch_size 100 \
	--cuda \
	--sys_threshold 0.020 \
	--sys_dynamic_weight True \
	--max_reward 5200 \
	--base_weight 0.40

	python main_sac_fork.py \
	--env "Walker2d-v3" \
	--policy "SAC_FORK"\
	--seed $i \
	--automatic_entropy_tuning True \
	--batch_size 100 \
	--cuda \
	--sys_threshold 0.15 \
	--sys_dynamic_weight True \
	--max_reward 3000 \
	--base_weight 0.1

	python main_sac_fork.py \
	--env "Humanoid-v3" \
	--policy "SAC_FORK"\
	--seed $i \
	--automatic_entropy_tuning True \
	--batch_size 100 \
	--cuda \
	--sys_threshold 0.10 \
	--sys_dynamic_weight True \
	--max_reward 4500 \
	--base_weight 0.10

	python main_sac_fork.py \
	--env "HalfCheetah-v3" \
	--policy "SAC_FORK"\
	--seed $i \
	--automatic_entropy_tuning True \
	--batch_size 100 \
	--cuda \
	--sys_threshold 0.10 \
	--sys_dynamic_weight True \
	--max_reward 6000 \
	--base_weight 0.10

	python main_sac_fork.py \
	--env "BipedalWalker-v3" \
	--policy "SAC_FORK"\
	--seed $i \
	--automatic_entropy_tuning True \
	--batch_size 100 \
	--cuda \
	--sys_threshold 0.01 \
	--sys_dynamic_weight True \
	--max_reward 3200 \
	--base_weight 0.40
done
