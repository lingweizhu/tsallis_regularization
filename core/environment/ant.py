import os
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'

import gym
import d4rl
import numpy as np

from core.utils.torch_utils import random_seed


class Ant:
    def __init__(self, seed=0):

        self.state_dim = (27,)
        self.action_dim = 8
        # self.env = gym.make('Ant-v2')

        # Loading d4rl env. For the convinience
        # of getting normalized score from d4rl
        self.env = gym.make('ant-random-v2')

        # control timeout setting in agent
        self.env._max_episode_steps = np.inf
        self.state = None

    def reset(self, seed):
        return self.env.reset(seed=seed)[0:27]

    def step(self, a):
        ret = self.env.step(a[0])
        state, reward, done, info = ret
        state = state[0:27]
        self.state = state
        return np.asarray(state), np.asarray(reward), np.asarray(done), info

    def get_visualization_segment(self):
        raise NotImplementedError

    def get_useful(self, state=None):
        if state:
            return state[0:27]
        else:
            return np.array(self.env.state[0:27])

    def info(self, key):
        return
