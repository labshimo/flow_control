import os
import gym
import time
import datetime
import numpy as np
from gym import wrappers
from memory import PERProportionalMemory
from model import *

class Config():
    def __init__(self):
        # environment
        self.monitor               = False
        self.monitor_mov_dir       = None
        # parameter
        self.input_shape           = (None, 4)
        self.gamma                 = 0.99
        self.n_step                = 3
        self.gamma_n               = self.gamma ** self.n_step
        self.weight_update_period  = 200
        self.max_episodes          = 100
        self.action_space          = 2
        self.network_update_period = 200
        self.learning_rate         = 0.001
        self.batch_size            = 32
        self.min_experiences       = 5000
        self.num_actors            = 8
        # memory
        self.capacity              = 2**14
        self.memory                = PERProportionalMemory
        # network
        self.network               = Network
        # training
        # Path to store the model weights and TensorBoard logs
        self.results_path          = os.path.join(os.path.dirname(os.path.realpath(__file__)),\
                                        "../results", os.path.basename(__file__)[:-3],\
                                        datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))  

class Env():
    def __init__(self, config):
        assert len(config.input_shape) == 2
        self._env                   = gym.make("CartPole-v0")
        if config.monitor:
            self._env   = wrappers.Monitor(self._env, config.monitor_mov_dir, video_callable=(lambda ep: ep % 50 == 0))
        
    def reset(self):
        return self._env.reset()

    def step(self, action):
        time.sleep(0.001)
        return self._env.step(action)
