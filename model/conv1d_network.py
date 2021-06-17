import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.backend as K

class Conv1DDuelingNetwork(tf.keras.Model):
    def __init__(self, config):
        super(Conv1DDuelingNetwork, self).__init__()
        (_, self.r, self.c, self.l) = config.input_shape
        self.action_space    = config.action_space
        self.R    = kl.Reshape((self.r*self.c, self.l))
        self.c1   = kl.Conv1D(32, 10, padding="same", activation="relu",
                            kernel_initializer="he_normal")
        self.c2   = kl.Conv1D(32, 10, padding="same", activation="relu",
                            kernel_initializer="he_normal")
        self.c3   = kl.Conv1D(32, 10, padding="same", activation="relu",
                            kernel_initializer="he_normal")
        self.fltn = kl.Flatten()
        
        self.dense1       = kl.Dense(32, activation="relu", name="dense1", kernel_initializer="he_normal")
        self.val1         = kl.Dense(32, activation='relu',   name="val1", kernel_initializer="he_normal")
        self.val2         = kl.Dense(1,   activation='linear', name="val2", kernel_initializer="he_normal")
        self.adv1         = kl.Dense(32, activation='relu',   name="advantage1", kernel_initializer="he_normal")
        self.adv2         = kl.Dense(config.action_space, activation='linear', name="advantage2", kernel_initializer="he_normal")
        self.concat       = kl.Concatenate()
        self.out          = kl.Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - tf.stop_gradient(K.mean(a[:,1:],keepdims=True)), output_shape=(config.action_space,))

        self.optimizer    = tf.keras.optimizers.RMSprop(lr=config.learning_rate, rho=0.95, epsilon=1.5e-7, centered=True, clipnorm=40) #tf.keras.optimizers.Adam(lr=lr)
        self.loss_func    = tf.losses.Huber()
        super().build(config.input_shape)
        
    @tf.function
    def call(self, x):
        x  = self.R(x)
        x  = self.c1(x)
        x  = self.c2(x)
        x  = self.c3(x)
        x  = self.fltn(x)
        x      = self.dense1(x)
        v      = self.val1(x)
        v      = self.val2(v)
        adv    = self.adv1(x)
        adv    = self.adv2(adv)
        concat = self.concat([v,adv])
        out    = self.out(concat)
        return out

    def huber_loss(self, errors):
        cond = tf.abs(errors) < 1.0
        L2   = tf.square(errors) * 0.5
        L1   = tf.abs(errors) - 0.5

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
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
    print()
    states = np.array([[-0.10430691, -1.55866031, 0.19466207, 2.51363456],
                       [-0.10430691, -1.55866031, 0.19466207, 2.51363456],
                       [-0.10430691, -1.55866031, 0.19466207, 2.51363456]])
    states.astype(np.float32)
    
    actions       = [0, 1, 1]
    target_values = [1, 1, 1]
    qnet          = Conv1DDuelingNetwork((None,4),action_space=2)
    pred          = qnet.predict(states)

    print(pred)
  

    frames = np.zeros((84, 84, 4))
    for i in range(4):
        frames[:, :, i] = np.random.randint(255, size=(84, 84)) / 255
    frames = frames.astype(np.float32)

    qnet = Conv1DDuelingNetwork((None, 84, 84, 4), action_space=4)
    #print(qnet.summary())

    pred = qnet.predict(frames)
    print(pred)