
# Offline Reinforcement Learning via Tsallis Regularization

**paper**: https://openreview.net/forum?id=HNqEKZDDRc&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DTMLR%2FAuthors%23your-submissions)
**authors**: Lingwei Zhu, Matthew Kyle Schlegel, Han Wang, Martha White

The Tsallis InAC code is modified based on the code provided by 'The In-Sample Softmax for Offline Reinforcement Learning' (https://openreview.net/pdf?id=u-RuvyDYqCM).


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



