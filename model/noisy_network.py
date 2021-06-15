import numpy as np
import random

import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras import activations, initializers, regularizers, constraints

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.backend as K

class NoisyDuelingNetwork(tf.keras.Model):
    def __init__(self, input_shape, action_space, lr=0.001):
        super(NoisyDuelingNetwork, self).__init__()

        self.action_space = action_space
        self.dense1       = kl.Dense(32, activation="relu", name="dense1", kernel_initializer="he_normal")
        self.dense2       = kl.Dense(32, activation="relu", name="dense2", kernel_initializer="he_normal")

        self.val1         = kl.Dense(32, activation='relu',   name="val1", kernel_initializer="he_normal")
        self.val2         = NoisyDense(1,  sigma_init=0.1, activation='linear', name="val2")
        self.adv1         = kl.Dense(32, activation='relu',   name="advantage1", kernel_initializer="he_normal")
        self.adv2         = NoisyDense(action_space, sigma_init=0.1, activation='linear', name="advantage2")
        self.concat       = kl.Concatenate()
        self.out          = kl.Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - tf.stop_gradient(K.mean(a[:,1:],keepdims=True)), output_shape=(action_space,))
        #self.out          = kl.Dense(action_space, name="output", kernel_initializer="he_normal")
        
        self.optimizer    = tf.keras.optimizers.RMSprop(lr=lr, rho=0.95, epsilon=1.5e-7, centered=True, clipnorm=40) #tf.keras.optimizers.Adam(lr=lr)
        self.loss_func    = tf.losses.Huber()
        super().build(input_shape)
        
    @tf.function
    def call(self, x):
        x      = self.dense1(x)
        x      = self.dense2(x)
        v      = self.val1(x)
        v      = self.val2(v)
        adv    = self.adv1(x)
        adv    = self.adv2(adv)
        concat = self.concat([v,adv])
        out    = self.out(concat)
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
        #self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.optimizer.apply_gradients((grad, var) for (grad, var) in zip(gradients, self.trainable_variables) if grad is not None)

        return td_errors, loss

class NoisyDense(Layer):
    def __init__(self, units,
                sigma_init=0.02,
                activation=None,
                use_bias=True,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                **kwargs):

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(NoisyDense, self).__init__(**kwargs)
        self.units                = units
        self.sigma_init           = sigma_init
        self.activation           = activations.get(activation)
        self.use_bias             = use_bias
        self.kernel_initializer   = initializers.get(kernel_initializer)
        self.bias_initializer     = initializers.get(bias_initializer)
        self.kernel_regularizer   = regularizers.get(kernel_regularizer)
        self.bias_regularizer     = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint    = constraints.get(kernel_constraint)
        self.bias_constraint      = constraints.get(bias_constraint)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.input_dim    = input_shape[-1]
        self.kernel_shape = tf.constant((self.input_dim, self.units))
        self.bias_shape   = tf.constant((self.units,))

        self.kernel = self.add_weight(shape=(self.input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.sigma_kernel = self.add_weight(shape=(self.input_dim, self.units),
                                      initializer=initializers.Constant(value=self.sigma_init),
                                      name='sigma_kernel')


        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)

            self.sigma_bias = self.add_weight(shape=(self.units,),
                                        initializer=initializers.Constant(value=self.sigma_init),
                                        name='sigma_bias')
        else:
            self.bias         = None
            self.epsilon_bias = None

        self.epsilon_kernel = K.zeros(shape=(self.input_dim, self.units))
        self.epsilon_bias   = K.zeros(shape=(self.units,))

        self.sample_noise()
        super(NoisyDense, self).build(input_shape)


    def call(self, X):
        perturbed_kernel = self.kernel + self.sigma_kernel * K.random_uniform(shape=self.kernel_shape)
        output = K.dot(X, perturbed_kernel)
        if self.use_bias:
            perturbed_bias = self.bias + self.sigma_bias * K.random_uniform(shape=self.bias_shape)
            output = K.bias_add(output, perturbed_bias)

        return self.activation(output) if self.activation is not None else output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape     = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def sample_noise(self):
        K.set_value(self.epsilon_kernel, np.random.normal(0, 1, (self.input_dim, self.units)))
        K.set_value(self.epsilon_bias,   np.random.normal(0, 1, (self.units,)))

    def remove_noise(self):
        K.set_value(self.epsilon_kernel, np.zeros(shape=(self.input_dim, self.units)))
        K.set_value(self.epsilon_bias, np.zeros(shape=self.units,))