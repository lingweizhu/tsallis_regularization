import os
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'

import gym
import d4rl
import numpy as np

from core.utils.torch_utils import random_seed


class Hopper:
    def __init__(self, seed=0):
        self.state_dim = (11,)
        self.action_dim = 3
        # self.env = gym.make('Hopper-v2')
        self.env = gym.make('hopper-random-v2') # Loading d4rl env. For the convinience of getting normalized score from d4rl
        self.env._max_episode_steps = np.inf # control timeout setting in agent
        self.state = None

    def reset(self, seed):
        return self.env.reset(seed=seed)

    def step(self, a):
        ret = self.env.step(a[0])
        state, reward, done, info = ret
        self.state = state
        # self.env.env.render()
        return np.asarray(state), np.asarray(reward), np.asarray(done), info

    def get_visualization_segment(self):
        raise NotImplementedError

    def get_useful(self, state=None):
        if state:
            return state
        else:
            return np.array(self.env.state)

    def info(self, key):
        return
