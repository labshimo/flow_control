import sys
sys.path.append('../')
import importlib
import numpy as np
from matplotlib import pyplot

env_name    = "flowcontrol"
module      = importlib.import_module("env." + env_name)
config      = module.Config()
env         = module.Env(config)

def test_burst_wave():
    # check burst frequeny
    burst_wave = env.exp.burst_wave
    # check number of enable avtions
    assert len(burst_wave) == config.action_space
    # check burst ratio
    for wave in burst_wave[:-1]:
        assert np.round(1-np.count_nonzero(wave==0)/len(wave),5) == config.action_setting["burst_ratio"]
        pyplot.plot(wave)
        pyplot.show()
    
    wave = burst_wave[-1]
    assert np.round(1-np.count_nonzero(wave==0)/len(wave),5) == 0
    pyplot.plot(wave)
    pyplot.show()
    
def test_pinish():
    # check punish
    burst_wave = env.exp.burst_wave
    punish     = env.exp.get_punish(burst_wave,4)
    
    assert punish[0] == 0.1 and punish[1] == 0.15
    assert punish[2] == 0.1 and punish[3] == 0.15
    assert punish[4] == 0

        

if __name__ == "__main__":
    test_burst_wave()
    test_pinish()