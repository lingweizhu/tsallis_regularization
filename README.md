
# Offline Reinforcement Learning via Tsallis Regularization

This is the code for the paper: Offline RL Via Tsallis Regularization accepted by Transaction on Machine Learning Research https://openreview.net/forum?id=HNqEKZDDRc.\
**authors**: Lingwei Zhu, Matthew Kyle Schlegel, Han Wang, Martha White



## Setup

Using python 3.9. Follow the instructions for [mujoco-py](https://github.com/openai/mujoco-py) to install mujoco. Then install
the required python dependencies.

```sh
pip install -r requirements.txt
```

Finally, run the `python setup_datasets.py` to download all the mujoco datasets through d4rl.



## Running a config file

the easiest way to run a config file is through the command:

```
./run_parallel.sh configs/ant/expert/tkl_policy.toml
```

which will run the entire sweep for an experiment. Configuration files with the name `tkl_policy` correspond to `Tsallis AWAC` and those with the name `tsallis_inac` correspond to `Tsallis InAC` in the paper. Data for the baseline algorithms is from the authors of ['The In-Sample Softmax for Offline Reinforcement Learning'](https://openreview.net/pdf?id=u-RuvyDYqCM).



