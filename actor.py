import os
import gym
import ray 
import time
import numpy as np
from tensorflow.python.keras.backend_config import epsilon
from dataclasses import dataclass

@dataclass
class History:
    pid:              int
    total_reward:     np.float
    q_values:         np.float
    epsilon:          np.float
    episode:          int
    step:             int
    global_step:      int

@ray.remote(num_cpus=1)
class Actor():
    def __init__(self, pid, env, epsilon, weight_q, config, mode="train"):
        self.pid            = pid
        self.mode           = mode
        self.config         = config
        self.env            = env(config)
        self.epsilon        = epsilon
        self.global_steps   = 0
        # build Q network
        self.q_network      = self.config.network(config) #network
        # Queue
        self.weight_q       = weight_q
        # action
        self.forward        = self.sample_action if mode=="test" else self.greedy_action
        self.weight_dir     = self.config.monitor_mov_dir + "/test/weight" 
        if not os.path.exists(self.weight_dir):
            os.makedirs(self.weight_dir)
    
    def initialize(self,episode):
        self.buffer       = []
        self.experiences  = []
        self.td_errors    = []
        self.R            = 0
        self.forward(np.random.rand(*self.config.input_shape[1:]))
        return self.env.reset(self.mode, episode)
    
    def test_play(self,episode):
        self.update_weight()
        historys = []
        for i in range(self.config.repeat_test):
            _, _, hist = self.play_with_error(episode+0.1*i)
            hist.total_reward += 10 * i 
            historys.append(hist)
        self.q_network.save_weights(self.weight_dir + "/episode{0}.h5".format(episode))
        return None, None, historys
            
    def play_with_error(self,episode):
        if self.mode == "train":
            self.update_weight()
        try:
            return self.play(episode)
        except:
            print ('error during control. try again.')
            self.env.exp.stop()
            return self.play_with_error(episode)
        
    def play(self,episode):
        total_reward = 0
        steps        = 0
        max_Qs       = 0
        done         = False
        state        = self.initialize(episode)
        
        while not done:
            action, max_q  = self.forward(state)

            next_state, reward, done, _ = self.env.step(action)
            self.backward(*(state, action, reward, next_state, done, max_q))
            state = next_state
                        
            steps         += 1
            total_reward  += reward
            max_Qs        += max_q

        self.global_steps += steps
        history            = History(self.pid, total_reward, max_Qs/steps, self.epsilon, 
                                episode, steps, self.global_steps)
        
        return self.td_errors, self.experiences, history

    def sample_action(self, state):
        q = self.q_network.predict([state])
        if np.random.random() < self.epsilon:
            return np.random.choice(self.config.action_space), np.max(q)
        else:
            return np.argmax(q), np.max(q)
        
    def greedy_action(self,state):
        q = self.q_network.predict([state])
        return np.argmax(q), np.max(q)

    def get_sample(self, n):
        s, a, _,  _, _  = self.buffer[0]
        _, _, _, s_, max_q_  = self.buffer[n-1]
        p = self.R + self.config.gamma_n * max_q_
        return s, a, self.R, s_, p

    def n_step_transition(self, reward, done):
        self.R = round((self.R + reward * self.config.gamma_n) / self.config.gamma,3)

        # n-step transition
        if done: # terminal state
            while len(self.buffer) > 0:
                n = len(self.buffer)
                s, a, r, s_, p = self.get_sample(n)
                # add to local memory
                self.experiences.append((s, a, r, s_))
                self.td_errors.append(p)
                self.R = round((self.R - self.buffer[0][2]) / self.config.gamma,3)
                self.buffer.pop(0)
            self.R = 0

        if len(self.buffer) >= self.config.n_step:
            s, a, r, s_, p = self.get_sample(self.config.n_step) 
            # add to local memory
            self.experiences.append((s, a, r, s_))
            self.td_errors.append(p)
            self.R = self.R - self.buffer[0][2]
            self.buffer.pop(0)

    def backward(self, state, action, reward, next_state, done, q):
        self.buffer.append((state, action, reward, next_state, q))
        self.n_step_transition(reward, done)

        # if self.global_steps % self.config.weight_update_period == 0:
        # if done:
        #     self.update_weight()
            
    def update_weight(self):
        while self.weight_q.empty():
            time.sleep(1)
            print("waiting for weights...")
        self.q_network.set_weights(self.weight_q.get(timeout=1))