#!/bin/bash

# Script to reproduce results

python main_sac_fork.py \
--env "Walker2d-v3" \
--policy "SAC" \
--seed 3 \
--automatic_entropy_tuning True \
--batch_size 100 \
--cuda
