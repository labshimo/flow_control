import os
import gym
import random 
import datetime
import numpy as np
from gym import wrappers
from skimage.color import rgb2gray
from skimage.transform import resize
from collections import deque
from PIL import Image, ImageDraw
from memory import PERProportionalMemory
from model import DuelingNetwork

class Config():
    def __init__(self):
        # environment
        self.image_size            = 84
        self.random_step           = 5
        self.monitor               = False
        self.monitor_mov_dir       = None
        # parameter
        self.input_shape           = (None, 84, 84, 4)
        self.gamma                 = 0.99
        self.n_step                = 3
        self.gamma_n               = self.gamma ** self.n_step
        self.weight_update_period  = 200
        self.max_episodes          = 30
        self.action_space          = actionSpace().n
        self.network_update_period = 200
        self.learning_rate         = 0.00025/4
        self.batch_size            = 32
        self.min_experiences       = 50
        self.num_actors            = 8
        # memory
        self.capacity              = 2**14
        self.memory                = PERProportionalMemory
        # network
        self.netowork              = DuelingNetwork
        # training
        # Path to store the model weights and TensorBoard logs
        self.results_path          = os.path.join(os.path.dirname(os.path.realpath(__file__)),\
                                        "../results", os.path.basename(__file__)[:-3],\
                                        datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))  

class actionSpace():
    def __init__(self):
        self.dic =  {
                0: [-2.0], 
                1: [-1.0], 
                2: [0.0], 
                3: [+1.0],
                4: [+2.0],
            }
        self.n  = len(self.dic)

class Env():
    def __init__(self, config):
        assert len(config.input_shape) == 4
        self.image_size             = config.image_size
        (_, self.r, self.c, self.l) = config.input_shape
        self.random_step            = config.random_step
        self.mode                   = "train"
        self._env                   = gym.make("Pendulum-v0")
        if config.monitor:
            self._env   = wrappers.Monitor(self._env, config.monitor_mov_dir, video_callable=(lambda ep: ep % 50 == 0))


        self.action_space           = actionSpace()
        [self.state.append(np.zeros((self.r, self.c))) for _ in range(self.l)] 

        assert self.r == self.c == self.image_size
        
    def reset(self):
        self.state = deque(maxlen=self.l)
        self.state.append(self._get_rgb_state(self._env.reset()))
        for _ in range(random.randint(0, self.random_step)):
            action = np.random.randint(0, self.action_space.n)
            observation, _, _, _ = self._env.step(self.action_space.dic[action]) 
            self.state.append(self._get_rgb_state(observation))
        state_T = np.array(self.state).transpose() # (l, r, c) -> (r, c, l)
        #print(state_T)
        return state_T

    def step(self, action):
        observation, reward_ori, terminal, info = self._env.step(self.action_space.dic[action])
        self.state.append(self._get_rgb_state(observation))
        reward  = self.process_reward(reward_ori)
        state_T = np.array(self.state).transpose() # (l, r, c) -> (r, c, l)

        return state_T, reward, terminal, info

    def process_reward(self, reward):
        if self.mode == "test":  
            return reward
        # normalize -16.5-0 to -1-1 
        self.max = 0
        self.min = -16.5
        # min max normarization
        if (self.max - self.min) == 0:
            return 0
        M = 1
        m = -0.5
        return ((reward - self.min) / (self.max - self.min))*(M - m) + m

    def _get_rgb_state(self, state):
        h_size   = self.image_size/1.1
        img      = Image.new("RGB", (self.image_size, self.image_size), (255, 255, 255))
        dr       = ImageDraw.Draw(img)
       
        l        = self.image_size/2.0 * 3.0/ 2.0
        dr.line(((h_size - l * state[1], h_size - l * state[0]), (h_size, h_size)), (0, 0, 0), 1)
        buff     = self.image_size/32.0
        dr.ellipse(((h_size - buff, h_size - buff), (h_size + buff, h_size + buff)), 
                   outline=(0, 0, 0), fill=(255, 0, 0))

        # convert GrayScale
        img_arr  = np.asarray(img.convert("L"))
        
        return img_arr/255.0
