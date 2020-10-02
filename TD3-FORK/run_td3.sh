#!/bin/bash

# Script to reproduce results

for ((i=0;i<5;i+=1))
do
	python main_td3_fork.py \
	--policy "TD3" \
	--env "HalfCheetah-v3" \
	--seed $i

	python main_td3_fork.py \
	--policy "TD3" \
	--env "Hopper-v3" \
	--seed $i

	python main_td3_fork.py \
	--policy "TD3" \
	--env "Walker2d-v3" \
	--seed $i \

	python main_td3_fork.py \
	--policy "TD3" \
	--env "Ant-v3" \
	--seed $i

	python main_td3_fork.py \
	--policy "TD3" \
	--env "Humanoid-v3" \
	--seed $i

	python main_td3_fork.py \
	--policy "TD3" \
	--env "BipedalWalker-v3" \
	--seed $i
done
