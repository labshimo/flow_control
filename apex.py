import ray
import importlib
import numpy as np
from copy import deepcopy
from ray.util.queue import Queue
from memory import PERProportionalMemory
from actor import Actor
from learner import Learner
from line_profiler import LineProfiler
from tensorflow import summary 
from PIL import Image
import io


class ApexAgent():
    def __init__(self, env_name):
        module                 = importlib.import_module("env." + env_name)
        self.env               = module.Env
        self.config            = module.Config()
        self.memory            = self.config.memory(self.config.capacity)
        self.actors_flag       = True
        self.wip_learner       = []
        # log
        self.actor_histories   = []
        self.learner_histories = []
        self._create_process()

    def _create_process(self):
        ray.init() 
        epsilons      = np.linspace(0.6, 0.005, self.config.num_actors)
        self.weight_q = Queue(maxsize=self.config.num_actors+2)
        self.actors   = [Actor.remote(i, self.env, epsilons[i], self.weight_q, self.config) for i in range(self.config.num_actors)]
        self.learner  = Learner.remote(self.weight_q, self.config)
        ray.get(self.learner.get_weight.remote())


    def run(self):
        # Write everything in TensorBoard
        self.writer = summary.create_file_writer(self.config.results_path)
        print(
            "\nTraining...\nRun tensorboard --logdir ./results and go to http://localhost:6006/ to see in real time the training performance.\n"
        )

        # Save hyperparameters to TensorBoard
        hp_table = [f"| {key} | {value} |" for key, value in self.config.__dict__.items()]
        with self.writer.as_default():
            summary.text("Hyperparameters","| Parameter | Value |\n|-------|-------|\n" + "\n".join(hp_table), step=0)
        

        minibatchs       = []
        self.wip_actors  = [actor.play.remote() for actor in self.actors]
        self.wip_learner = [self.learner.update_network.remote(minibatchs)]
       
        while self.wip_actors:
            # if not self.weight_q.full():
            #     weight = ray.get(self.learner.get_weight.remote())
            #     self.weight_q.put(weight)
            self.acotrs_play()
            if self.memory.tree.write > self.config.min_experiences:
                minibatchs = [self.memory.sample(batch_size=self.config.batch_size,step=0) for _ in range(16) ]
            self.learner_play(minibatchs)     
        
        self.writer.close()    
        ray.shutdown()
        

    def acotrs_play(self):
        finished, self.wip_actors = ray.wait(self.wip_actors, num_returns=1)
        td_errors, transitions, history = ray.get(finished[0])
        if not history.finished:
            self.wip_actors.extend([self.actors[history.pid].play.remote()])
        [self.memory.add_p(td_errors[i], transitions[i]) for i in range(len(td_errors))]
        if history.pid == self.config.num_actors - 1:
            print("Actor {0:d} Episode {1:6d} Total reward {2:3.0f} Step {3:3d}| Q value {4:3.2f} | Epsilon {5:2.4f}"\
            .format(history.pid,history.episode,history.total_reward,history.step,history.q_values,history.epsilon))
        with self.writer.as_default():
            summary.scalar("{0}.Actor{0}/Total_reward".format(history.pid), history.total_reward, history.episode,)
            summary.scalar("{0}.Actor{0}/q_values".format(history.pid), history.q_values, history.episode,)
            summary.scalar("{0}.Actor{0}/epsilon".format(history.pid), history.epsilon, history.episode,)
            summary.scalar("{0}.Actor{0}/played_steps".format(history.pid), history.step, history.episode)

    def learner_play(self,minibatch):
        finished, self.wip_learner  = ray.wait(self.wip_learner, num_returns=1, timeout=0)
        if finished:
            indexes, td_errors, historys = ray.get(finished[0])
            self.wip_learner.extend([self.learner.update_network.remote(minibatch)]) 
            [self.memory.update(indexes[i], td_errors[i]) for i in range(len(td_errors))]
            
            with self.writer.as_default():
                for h in historys:
                    summary.scalar("{0}.Learner/Loss".format(self.config.num_actors), h.loss, h.update)
                    
# @ray.remote    
# class Tester:

def main():
    env_name    = "carpole"
    agent       = ApexAgent(env_name)
    agent.run()

if __name__ == "__main__":
    main()
    
    # prof = LineProfiler()    
    # prof.add_module(ApexAgent)
    # prof.add_module(Actor)
    # prof.add_function(main)  
    # prof.runcall(main)   
    # prof.print_stats(output_unit=1e-9)