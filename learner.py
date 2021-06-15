import os
import ray
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from model import DuelingNetwork


@dataclass
class History:
    update: int
    loss:   float

@ray.remote(num_cpus=1, num_gpus=1.0)
class Learner:
    def __init__(self, weight_q, config):
        self.config         = config
        # build Q network
        self.q_network      = self.config.network(config)
        self.target_network = self.config.network(config)
        self.target_network.set_weights(self.q_network.get_weights())
        
        # Queue
        self.weight_q       = weight_q
        # parameter 
        self.num_learn      = 0
        self.loss           = 0

    def update_network(self, minibatchs):
        td_errors_list, indexes_list, history_list = [], [], []
        if not minibatchs:
            self.get_weight()

        while minibatchs:
            if self.num_learn % self.config.network_update_period == 0:
                self.target_network.set_weights(self.q_network.get_weights())

            (states, actions, rewards, next_states), indexes, weights = self.get_minibatch(minibatchs.pop())
            self.get_weight()
            next_acts     = np.argmax(self.q_network.predict(next_states), axis=1)
            next_Qs       = self.target_network.predict(next_states)
            # print(states[0], next_states[0], next_Qs[0])
            target_values = [reward + self.config.gamma_n * next_q[next_a]
                                for reward, next_a, next_q in zip(rewards, next_acts, next_Qs)]
            
            td_errors, self.loss = self.q_network.update(np.array(states), np.array(actions), np.array(target_values), weights)
            td_errors_list.extend(td_errors), indexes_list.extend(indexes)
            history_list.append(History(self.num_learn, self.loss))
            self.num_learn += 1


        return indexes_list, td_errors_list, history_list
        
    def get_minibatch(self, minibatchs):
        """Experience Replay mechanism
        """
        
        (indexes, transitions, weights) = minibatchs
        
        minibatch = ([], [], [], []) # states, actions, rewarsds, next_states
        
        [mb.append(t)  for exp in transitions for mb, t in zip(minibatch, exp)]
        return minibatch, indexes, weights
    
    def get_weight(self):
        # return self.q_network.get_weights()
        while not self.weight_q.full():
            self.weight_q.put(self.q_network.get_weights())
