import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl

class Network(tf.keras.Model):
    def __init__(self, config):
        super(Network, self).__init__()
        self.action_space = config.action_space
        self.dense1       = kl.Dense(32, activation="relu", name="dense1", kernel_initializer="he_normal")
        self.dense2       = kl.Dense(32, activation="relu", name="dense2", kernel_initializer="he_normal")
        self.out          = kl.Dense(config.action_space, name="output", kernel_initializer="he_normal")
        
        self.optimizer    = tf.keras.optimizers.RMSprop(lr=config.learning_rate, rho=0.95, epsilon=1.5e-7, centered=True, clipnorm=40) 
        self.loss_func    = tf.losses.Huber()
        super().build(config.input_shape)
    @tf.function
    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        out = self.out(x)
        return out

    def huber_loss(self, errors):
        cond = tf.abs(errors) < 1.0
        L2 = tf.square(errors) * 0.5
        L1 = tf.abs(errors) - 0.5

        return tf.where(cond, L2, L1)

    def predict(self, state):
        return self(np.asarray(state).astype(np.float32)).numpy()
    
    def update(self, states, selected_actions, target_values, weights):   
        with tf.GradientTape() as tape:
            selected_actions_onehot = tf.one_hot(selected_actions,self.action_space)
            selected_action_values  = tf.reduce_sum(self(states) * selected_actions_onehot, axis=1)
            td_errors               = target_values - selected_action_values
    
            loss = tf.reduce_mean(self.huber_loss(td_errors*weights))
            
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return td_errors, loss

if __name__ == "__main__":
    states = np.array([[-0.10430691, -1.55866031, 0.19466207, 2.51363456],
                       [-0.10430691, -1.55866031, 0.19466207, 2.51363456],
                       [-0.10430691, -1.55866031, 0.19466207, 2.51363456]])
    states.astype(np.float32)
    
    actions       = [0, 1, 1]
    target_values = [1, 1, 1]
    qnet          = Network(action_space=2)
    pred          = qnet.predict(states)

    print(pred)

    qnet.update(states, actions, target_values)