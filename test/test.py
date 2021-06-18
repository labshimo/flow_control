import sys
sys.path.append('../')
import importlib
import numpy as np
from copy import deepcopy
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

class Tester():
    def __init__(self, pid, env, epsilon, config, mode="train"):
        self.pid            = pid
        self.mode           = mode
        self.config         = config
        self.env            = env(config)
        self.epsilon        = epsilon
        self.global_steps   = 0
        # build Q network
        self.q_network      = self.config.network(config) #network
        self.q_network.summary()
    
    def initialize(self, episodes):
        self.buffer       = []
        self.experiences  = []
        self.td_errors    = []
        self.R            = 0
        self.forward(np.random.rand(*self.config.input_shape[1:]))
 
        return self.env.reset(self.mode, episodes)
    
    def play_with_error(self,episodes):
        try:
            return self.play(episodes)
        except Exception as e:
            print ('=== error ===')
            print ('type:' + str(type(e)))
            print ('args:' + str(e.args))
            print ('e :' + str(e))
            self.env.exp.stop()
            return self.play_with_error(episodes)
        
    def play(self, episodes):
        total_reward  = 0
        steps         = 0
        max_Qs        = 0
        done          = False
        state         = self.initialize(episodes)

        while not done:
            action, max_q = self.forward(state)
            next_state, reward, done, _ = self.env.step(action)
            self.backward(*(state, action, reward, next_state, done, max_q))
            state = next_state
                        
            steps             += 1
            total_reward      += reward
            max_Qs            += max_q
        
        self.global_steps += steps
        history  = History(self.pid, total_reward, max_Qs/steps, self.epsilon, 
                    episodes, steps, self.global_steps)
        
        return self.td_errors, self.experiences, history



    def sample_action(self, state):
        q = self.q_network.predict([state])
        if np.random.random() < self.epsilon:
            return np.random.choice(self.config.action_space), np.max(q)
        else:
            return np.argmax(q), np.max(q)

    def forward(self, state):
        if self.mode == "train":
            return self.sample_action(state)
        else: 
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

def main():
    env_name    = "flowcontrol"
    module      = importlib.import_module("env." + env_name)
    env         = module.Env
    config      = module.Config()
    agent       = Tester(0, env, 0.5, config)
    agent_test  = Tester(0, env, 0.5, config, mode="test")
    
    for i in range(20):
        a, b, c = agent.play_with_error(i)
        print(i, c)
        if i%5==0:
            a, b, c = agent_test.play_with_error(i)
            print(i, c)
        
        

if __name__ == "__main__":
    main()
