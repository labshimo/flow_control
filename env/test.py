import gym

import random 
import numpy as np
from gym import wrappers
from collections import deque

class Config():
    def __init__(self):
        # environment
        self.monitor               = False
        self.monitor_mov_dir       = None
        # parameter
        self.input_shape           = (None, 2)
        self.gamma                 = 0.99
        self.n_step                = 1
        self.gamma_n               = self.gamma ** self.n_step
        self.weight_update_period  = 200
        self.max_episodes          = 1000
        self.action_space          = 2
        self.network_update_period = 200
        self.learning_rate         = 0.00025/4
        self.batch_size            = 32
        self.min_experiences       = 512

class Env():
    def __init__(self, config):
        assert len(config.input_shape) == 2
        self.config = config
    def reset(self):
        self.index = 0
        return np.zeros(self.config.input_shape[1])

    def step(self, action):
        self.index += 1
        observartion = np.ones(self.config.input_shape[1]) * self.index
        reward = 1 
        done = True if self.index > 20 else False
        return observartion, reward, done, None
