#!/bin/bash

# Script to reproduce results

for ((i=0;i<5;i+=1))
do
	python main_sac_fork.py \
	--env "Walker2d-v3" \
	--policy "SAC" \
	--seed $i \
	--automatic_entropy_tuning True \
	--batch_size 100 \
	--cuda

	python main_sac_fork.py \
	--env "Hopper-v3" \
	--policy "SAC" \
	--seed $i \
	--automatic_entropy_tuning True \
	--batch_size 100 \
	--cuda

	python main_sac_fork.py \
	--env "Ant-v3" \
	--policy "SAC" \
	--seed $i \
	--automatic_entropy_tuning True \
	--batch_size 100 \
	--cuda

	python main_sac_fork.py \
	--env "Humanoid-v3" \
	--policy "SAC" \
	--seed $i \
	--automatic_entropy_tuning True \
	--batch_size 100 \
	--cuda

	python main_sac_fork.py \
	--env "HalfCheetah-v3" \
	--policy "SAC" \
	--seed $i \
	--automatic_entropy_tuning True \
	--batch_size 100 \
	--cuda

	python main_sac_fork.py \
	--env "BipedalWalker-v3" \
	--policy "SAC" \
	--seed $i \
	--automatic_entropy_tuning True \
	--batch_size 100 \
	--cuda
done
