# Exploration through Hierarchical Meta Reinforcement Learning

![HalfCheetahDir](https://raw.githubusercontent.com/tristandeleu/pytorch-maml-rl/master/_assets/halfcheetahdir.gif)

Implementation of Exploration through Hierarchical Meta Reinforcement Learning in Pytorch. This implementation closely follows the implementation of MAML in Pytorch. [pytorch-maml-rl](https://github.com/tristandeleu/pytorch-maml-rl).

The script also follows Soft actor critic which is being used as the lower level base policy in our setup.

The repository also makes use of the SAC implementations in [rlkit](https://github.com/vitchyr/rlkit), however, since the rlkit repository is included with the package, there is no separate requirement to install the same.

This repository also consists of a Pytorch implementation [empowerment_skills.py](empowerment_skills.py) of the paper- [Diversity is all you need](https://arxiv.org/abs/1802.06070)

## Getting started
To avoid any conflict with your existing Python setup, and to keep this project self-contained, it is suggested to work in a virtual environment with [`virtualenv`](http://docs.python-guide.org/en/latest/dev/virtualenvs/). To install `virtualenv`:
```
pip install --upgrade virtualenv
```
Create a virtual environment, activate it and install the requirements in [`requirements.txt`](requirements.txt).
```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage
You can use the [`main.py`](main.py) script in order to run reinforcement learning experiments with the algorithm. This script was tested with Python 3.5. Note that some environments may also work with Python 2.7 (all experiments besides MuJoCo-based environments).
```
python main.py --env-name HalfCheetahDir-v1 --num-workers 8 --fast-lr 0.1 --max-kl 0.01 --fast-batch-size 20 --meta-batch-size 40 --num-layers 2 --hidden-size 100 --num-batches 1000 --gamma 0.99 --tau 1.0 --cg-damping 1e-5 --ls-max-steps 15 --output-folder maml-halfcheetah-dir --device cuda
```

You can use the [`t_maml_rl.py`](t_maml_rl.py) script in order to test reinforcement learning experiments with the algorithm. Baseline and MAML algorithms are also supported for comparisons.
