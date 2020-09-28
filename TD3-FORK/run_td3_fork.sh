#!/bin/bash

# Script to reproduce results

for ((i=1;i<2;i+=1))
do
	python main_td3_fork.py \
	--policy "TD3_FORK" \
	--env "HalfCheetah-v3" \
	--seed $i \
    --base_weight  0.6\
    --sys_threshold 0.20 \
    --sys_dynamic_weight True \
    --max_reward 12000 \
    --save_model True
done
