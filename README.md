## FORK

Author's PyTorch implementation of FORK: A Forward-Looking Actor For Model-Free Reinforcement Learning. The paper can be found [here](https://arxiv.org/pdf/2010.01652.pdf).

We proposed a new type of Actor, named forward-looking Actor or FORK for short, for Actor-Critic algorithms. 

FORK can be easily integrated into a model-free Actor-Critic algorithm.



## Usage

TD3-FORK and SAC-FORK are tested on [Mujoco](http://www.mujoco.org/) continuous control tasks in [OpenAI gym](https://gym.openai.com/). 

Neural Networks are trained using Pytorch 1.4 and Python 3.7

The results in the paper can be reproduced by running:



```
./run_td3.sh
./run_td3_fork.sh
./run_sac.sh
./run_sac_fork.sh
```



## BipedalWalkerHardcore

BipedalWalkerHardcore is a advanced version of BipedalWalker with ladders, stumps, pitfalls.

![](https://github.com/honghaow/FORK/blob/master/BipedalWalkerHardcore/bipedalwalker-hardcore1.png)

TD3-FORK can slove the task with as few as four hours by using the defaulat GPU setting provided by [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb). 

You can view the performance on [Youtube](https://www.youtube.com/watch?v=pzzP8fA5Ipg).



## Bibtex

If you use our code or data please cite the paper.



```
@article{wei2020fork,
  title={FORK: A Forward-Looking Actor For Model-Free Reinforcement Learning},
  author={Wei, Honghao and Ying, Lei},
  booktitle={2021 IEEE 60th Annual Conference on Decision and Control (CDC)},
  year={2021}
}
```



## Acknowledgement

The TD3 code was based on [TD3](https://github.com/sfujim/TD3)
The SAC code was based on [SAC1](https://github.com/denisyarats/pytorch_sac) and [SAC2](https://github.com/vitchyr/rlkit).
