# coding: utf-8
import os
import csv
import datetime
import numpy as np
import pandas as pd
from collections import deque
from scipy import signal
from scipy.ndimage.interpolation import shift 
import nidaqmx
from nidaqmx.constants import (Edge, TriggerType, AcquisitionType, LineGrouping, Level, TaskMode, RegenerationMode)
from nidaqmx.utils import flatten_channel_string
from nidaqmx.stream_readers import (AnalogMultiChannelReader)
from nidaqmx.stream_writers import (AnalogSingleChannelWriter)

from memory import PERProportionalMemory
from model import *

class Config():
    def __init__(self):
        # environment
        self.monitor               = False
        self.monitor_mov_dir       = None
        self.reward_threshhold     = -0.55
        self.sample_rate           = 200000
        self.number_of_samples     = 2000
        self.dynamic_pressre       = 57
        self.input_channels        = "Dev1/ai0:4"
        self.output_channel        = "Dev1/ao0"
        self.state_channel         = [0,3,4]
        self.reward_channel        = 4
        self.cofficients           = np.array([[22.984,23.261,22.850,35.801,25.222]])/1e6*900
        self.num_channel           = len(self.cofficients)
        self.num_state             = len(self.state_channel) 
        self.unit                  = 1e6
        self.gain                  = 900       
        self.timeout               = 2
        self.total_time            = 1
        self.plasma_actuator_csv   = {"base_frequency": 12000,"burst_frequency": [100,600],"burst_ratio":[0.1],"voltage":[4,6]}
        # parameter
        self.input_shape           = (None, 10, 10, 4)
        self.gamma                 = 0.99
        self.n_step                = 3
        self.gamma_n               = self.gamma ** self.n_step
        self.weight_update_period  = 200
        self.max_episodes          = 100
        self.action_space          = 7
        self.network_update_period = 200
        self.learning_rate         = 0.001
        self.batch_size            = 32
        self.min_experiences       = 5000
        self.num_actors            = 1
        # memory
        self.capacity              = 2**14
        self.memory                = PERProportionalMemory
        # network
        self.network               = Conv1DDuelingNetwork
        # training
        # Path to store the model weights and TensorBoard logs
        self.results_path          = os.path.join(os.path.dirname(os.path.realpath(__file__)),\
                                        "../results", os.path.basename(__file__)[:-3],\
                                        datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))  
        
        self.log_dir               = './flow_control/'

class Env():
    def __init__(self, config):
        self.config                 = config
        (_, self.r, self.c, self.l) = config.input_shape
        self.exp                    = ExpFlowSeparation(config)
        self.mode                   = "train"
        # log
        self.data_path              = self.config.log_dir + "data/"
        self.run_path               = self.config.log_dir + "run/"

        if not (os.path.exists(self.data_path) or os.path.exists(self.run_path)):
            os.makedirs(self.data_path)
            os.makedirs(self.run_path)

    def reset(self):
        self.state         = deque(maxlen=self.l)
        observation_ori    = self.exp.reset()
        observation        = self.process_observation(observation_ori)
        self.buffer_memory = np.zeros((0,3+self.config.action_space))
        self.env_memory    = observation_ori

        for _ in range(self.config.no_op_steps):
            observation_ori, reward_ori, _  = self.exp.step(self.config.action_space)  # Do nothing
            self.state.append(self.process_observation(observation_ori))
            data                   = [0, reward_ori, self.process_reward(reward_ori)].extend(np.zeros(self.config.action_space))
            self.buffer_memory     = np.append(self.buffer_memory,[data],axis=0)
            self.env_memory        = np.append(self.env_memory,observation_ori,axis=0)

        return np.array(self.state).transpose() # (l, r, c) -> (r, c, l)
    
    def step(self, action, q_values):
        observation_ori, reward_ori, terminal = self.exp.step(action)
        observation        = self.process_observation(observation_ori)
        reward             = self.process_reward(reward_ori)
        self.state.append(self.process_observation(observation_ori))
        data               = [action, reward_ori, reward].extend(q_values)
        self.buffer_memory = np.append(self.buffer_memory,[data],axis=0)
        self.env_memory    = np.append(self.env_memory,observation_ori,axis=0)
        if terminal:
            return np.array(self.state).transpose(), reward, terminal
        else:
            self.stop()
            return np.array(self.state).transpose(), reward, terminal

    def process_observation(self, read):
        obs = np.array([ np.convolve(read[:,self.config.state_channel[i]], self.b, mode='same') for i in range(self.config.num_state) ]).T
        return np.round((np.average(np.split(obs,self.num_mave,0),axis=1)),2)

    def process_reward(self, reward_ori):
        reward     = self.config.reward_threshhold < reward_ori
        return reward.astype(np.int),0

    def stop(self):
        for _ in range(self.config.no_op_steps):
            observation_ori, reward_ori, _  = self.exp.step(self.config.action_space)  # Do nothing
            self.state.append(self.process_observation(observation_ori))
            data                   = [0, reward_ori, self.process_reward(reward_ori)].extend(np.zeros(self.config.action_space))
            self.buffer_memory     = np.append(self.buffer_memory,[data],axis=0)
            self.env_memory        = np.append(self.env_memory,observation_ori,axis=0)

        self.exp.stop()

    def save_simlator_data(self,episode):
        data_save_path = self.data_path + '/data{:0=5}.csv'.format(episode)
        run_save_path  = self.run_path + '/run{:0=5}.csv'.format(episode)
        df_d = pd.DataFrame(self.env_memory)
        df_r = pd.DataFrame(self.buffer_memory)
        df_d.to_csv(data_save_path)
        df_r.to_csv(run_save_path)

class ExpFlowSeparation():
    def __init__(self, config):
        self.config     = config
        self.dt         = 1/self.config.sample_rate
        self.n_loop     = int(self.config.sample_rate*self.config.total_time/self.config.number_of_samples)
        self.burst_wave = self.get_burst_wave(self.config.plasma_actuator_csv)
        # moving averege setting
        self.num_mave          = 10
        self.b                 = np.ones(self.num_mave)/self.num_mave
        self.a_buffer          = np.zeros(self.config.action_spaces)

    def create_burst_wave(self, base_frequency, burst_frequency, burst_ratio, voltage):
        ### example -> burst_freq=600[Hz], burst_ratio=0.1[-], voltage=3[kV]
        time            = np.linspace(0.0, self.dt, self.config.number_of_samples)
        tmp_sin         = np.sin(2*np.pi*int(base_frequency)*time)
        tmp_sq          = [(signal.square(2 * np.pi * bf_i * time, duty=br_i)+1)/2 for br_i in burst_ratio for bf_i in burst_frequency]
        wave            = [(tmp_sin * tmp_sq_i) * vi / 2 for tmp_sq_i in tmp_sq for vi in voltage] 
        zero_wave       = np.zeros((1,self.config.number_of_samples)) 
        wave            = np.append(wave,zero_wave,axis=0)
        return wave

    def get_punish(self, wave, reference):
        wave_min   = np.min(wave, axis=1)
        wave_max   = np.max(wave, axis=1)
        wave_ratio = 1-np.count_nonzero(wave==0, axis=1)/wave.shape[1]
        punish = wave_ratio * np.round(wave_max-wave_min, decimals=2)/reference
        return np.round(punish, decimals=2)

    def setup_DAQmx(self):
        self.read_task         = nidaqmx.Task() 
        self.write_task        = nidaqmx.Task() 
        self.sample_clk_task   = nidaqmx.Task()
        # Use a counter output pulse train task as the sample clock source
        # for both the AI and AO tasks.
        self.sample_clk_task.co_channels.add_co_pulse_chan_freq('Dev1/ctr0', freq=self.config.sample_rate, idle_state=Level.LOW)
        self.sample_clk_task.timing.cfg_implicit_timing(sample_mode=AcquisitionType.CONTINUOUS,samps_per_chan=self.config.number_of_samples)
        self.sample_clk_task.control(TaskMode.TASK_COMMIT)
        samp_clk_terminal = '/Dev1/Ctr0InternalOutput'

        self.read_task.ai_channels.add_ai_voltage_chan(self.config.input_channel, max_val=10, min_val=-10)
        self.read_task.timing.cfg_samp_clk_timing(self.config.sample_rate, source=samp_clk_terminal, \
            active_edge=Edge.FALLING,sample_mode=AcquisitionType.CONTINUOUS, samps_per_chan=self.config.number_of_samples)

        self.write_task.ao_channels.add_ao_voltage_chan(self.config.output_channel, max_val=10, min_val=-10)
        self.write_task.timing.cfg_samp_clk_timing(self.config.sample_rate, source=samp_clk_terminal, active_edge=Edge.FALLING, \
            sample_mode=AcquisitionType.CONTINUOUS, samps_per_chan=self.config.number_of_samples)

        self.write_task.out_stream.regen_mode = RegenerationMode.DONT_ALLOW_REGENERATION
        self.write_task.out_stream.auto_start = False

        self.writer = AnalogSingleChannelWriter(self.write_task.out_stream)
        self.reader = AnalogMultiChannelReader(self.read_task.in_stream)

    def reset(self):
        self.env_memory    = np.zeros((0,self.config.num_channel))
        self.buffer_memory = np.zeros((0,4+self.config.action_space))
        self.step_count    = 0
        self.setup_DAQmx()
        # start read analog 
        self._start_reading()
        self._start_writing()
        return self._reading()

    def step(self, action):
        self._writing(action)
        read        = self._reading() 
        observation = self._observation(read)
        reward      = self._reward(read)
            
        if self.step_count < self.n_loop: 
            return observation, reward, False
        else:
            return reward, reward, True
    
    def _reward(self, read):
        return np.average(read[:,self.config.reward_channel])
         
    def _start_reading(self):
        # start read analog 
        self.read_task.start()
        self.sample_clk_task.start()

    def _start_writing(self):
        self._writing(0)   
        self.write_task.start()
        for _ in range(2):
            self._writing(0)     
        
    def stop(self):
        self.read_task.close()
        self.write_task.close()
        self.sample_clk_task.close()
    
    def _reading(self):
        values_read = np.zeros((self.config.num_channel,self.config.number_of_samples), dtype=np.float32)
        self.reader.read_many_sample(values_read, number_of_samples_per_channel=self.config.number_of_samples,timeout=2)
        return (((values_read / self.config.cofficients) + self.config.dynamic_pressre ) / self.config.dynamic_pressre).T
        
    def _writing(self,action):
        print(self.burst_wave[action].shape)
        print(self.burst_wave[action])
        self.writer.write_many_sample(self.burst_wave[action])
    
    