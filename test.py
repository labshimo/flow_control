import sys
import ray
import importlib
from actor import Actor
    
def main(weight_path):
    ray.init() 
    env_name    = "flowcontrol"
    module      = importlib.import_module("env." + env_name)
    env         = module.Env
    config      = module.Config()
    tester      = Actor.remote(1, env, 0, None, config, mode="test")
    _, _, tests = ray.get(tester.test_play.remote(0, weight_path))
    for test in tests:
        print(test)

if __name__ == "__main__":
    main(sys.argv[1])