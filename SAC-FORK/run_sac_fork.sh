#!/bin/bash

# Script to reproduce results

for i in 4
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
    --sys_weight 0.10 \
    --max_reward 2500 \
    --base_weight 0.1
done
