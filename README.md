# FORK
This repository contains the code accompanying the [Paper](https://arxiv.org/abs/2010.01652) A Forward-Looking Actor From Model-free Reinforcement Learning. 

PyTorch implementation of FORK. If you use our code or data please cite the paper.

TD3-FORK and SAC-FORK are tested on [Mujoco](http://www.mujoco.org/) continuous control tasks in [OpenAI gym](https://gym.openai.com/). 

Neural Netorks are trained using Pytorch 1.4 and Python 3.7

# Usage
```
./run_td3.sh
./run_td3_fork.sh
./run_sac.sh
./run_sac_fork.sh
```



# Acknowledgement
The TD3 code was based on [TD3](https://github.com/sfujim/TD3)
The SAC code was based on [SAC1](https://github.com/denisyarats/pytorch_sac) and [SAC2](https://github.com/vitchyr/rlkit).
