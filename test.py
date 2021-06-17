import time
import importlib
import numpy as np
from copy import deepcopy
from line_profiler import LineProfiler
from model import DuelingNetwork
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
    finished:         bool

class Tester():
    def __init__(self, pid, env, epsilon, config):
        self.pid            = pid
        self.config         = config
        self.env            = env(config)
        self.epsilon        = epsilon
        self.global_steps   = 0
        self.episodes       = 0
        # build Q network
        self.q_network      = self.config.network(config) #network
        self.q_network.summary()
    
    def initialize(self):
        self.buffer       = []
        self.experiences  = []
        self.td_errors    = []
        self.R            = 0
        self.forward(np.random.rand(*self.config.input_shape[1:]))
 
        return self.env.reset()

    def play(self):
        total_reward = 0
        steps        = 0
        max_Qs       = 0
        done         = False
        state        = self.initialize()

        while not done:
            action, max_q = self.forward(state)
            next_state, reward, done, _ = self.env.step(action)
            self.backward(*(state, action, reward, next_state, done, max_q))
            state = next_state
                        
            steps             += 1
            total_reward      += reward
            self.global_steps += 1
            max_Qs            += max_q
        
        finished = True if self.episodes > self.config.max_episodes else False
        
        history  = History(self.pid, total_reward, max_Qs/steps, self.epsilon, 
                    self.episodes, steps, self.global_steps, finished)
        self.episodes += 1

        return self.td_errors, self.experiences, history


    def sample_action(self, state):
        q = self.q_network.predict([state])
        if np.random.random() < self.epsilon:
            return np.random.choice(self.config.action_space), np.max(q)
        else:
            return np.argmax(q), np.max(q)

    def forward(self, state):
        return self.sample_action(state)

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


# @ray.remote    
# class Tester:

def main():
    capacity    = 2**14
    env_name    = "flowcontrol"
    module      = importlib.import_module("env." + env_name)
    env         = module.Env
    config      = module.Config()
    agent       = Tester(0, env, 0.5, config)
    for i in range(100):
        print(i)
        try:
            agent.play()
        except Exception as e:
            print ('=== error ===')
            print ('type:' + str(type(e)))
            print ('args:' + str(e.args))
            print ('e :' + str(e))
            agent.env.exp.stop()

if __name__ == "__main__":
    main()
    
    # prof = LineProfiler()    
    # prof.add_module(TestActor)
    # prof.add_function(main)  
    # prof.runcall(main)   
    # prof.print_stats(output_unit=1e-9)