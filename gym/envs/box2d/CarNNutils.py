import tensorflow as tf
import numpy as np
import random
import os
import glob
import time
import json

from self_configs import NN_PATH, my_possible_actions, num_actions, num_episodes, epsilon, batch_size, max_ep_frames, penalty_factor, discount, buff_size, times_for_action

class CarNN(tf.keras.Model):
    """Dense neural network class."""
    def __init__(self,outs):
        super(CarNN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(256, activation="relu")
        self.dense2 = tf.keras.layers.Dense(64, activation="relu")
        self.dense3 = tf.keras.layers.Dense(outs, dtype=tf.float32) # No activation
        
    def call(self, x):
        """Forward pass."""
        # x = tf.keras.backend.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)

optimizer = tf.keras.optimizers.Adam(1e-4)
mse = tf.keras.losses.MeanSquaredError()

class mydeq():

    def __init__(self,maxsize):
        self.buff = [None]*maxsize
        self.start = 0
        self.fin = 0
        self.size = 0
        self.maxsize = maxsize

    def append(self,elem):
        if self.size < self.maxsize:
            self.buff[self.fin] = elem
            self.size += 1
            self.fin = (self.fin + 1)%self.maxsize
        else:
            self.start = (self.start + 1)%self.maxsize
            self.buff[self.fin] = elem
            self.fin = (self.fin + 1)%self.maxsize

    def __getitem__(self, index):
        return self.buff[(self.start + index)%self.maxsize]

    def __len__(self):
        return self.size

    def mean(self, verbose = False):
        if self.size < self.maxsize: 
            return 0
        else:
            if verbose: print(self.buff)
            return np.mean(self.buff)

class ReplayBuffer(object):
    """Experience replay buffer that samples uniformly."""
    def __init__(self, size):
        self.buffer = mydeq(maxsize=size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def __len__(self):
        return len(self.buffer)

    def sample(self, num_samples):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        idx = np.random.choice(len(self.buffer), num_samples)
        for i in idx:
            elem = self.buffer[i]
            state, action, reward, next_state, done = elem
            states.append(np.array(state, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            next_states.append(np.array(next_state, copy=False))
            dones.append(done)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states)
        dones = np.array(dones, dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def get_all(self):

        states = [np.asarray(x[0]) for x in self.buffer.buff if x is not None]
        actions = [np.asarray(x[1]) for x in self.buffer.buff if x is not None]
        rewards = [x[2] for x in self.buffer.buff if x is not None]
        nexts = [np.asarray(x[3]) for x in self.buffer.buff if x is not None]
        dones = [x[4] for x in self.buffer.buff if x is not None]

        #print(rewards)

        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(nexts), np.array(dones, dtype=np.float32)

@tf.function
def train_step(states, actions, rewards, next_states, dones, main_nn, target_nn):
    """Perform a training iteration on a batch of data sampled from the experience
    replay buffer."""
    # Calculate targets.
    next_qs = target_nn(next_states)
    max_next_qs = tf.reduce_max(next_qs, axis=-1)
    target = rewards + (1. - dones) * discount * max_next_qs
    with tf.GradientTape() as tape:
        qs = main_nn(states)
        action_masks = tf.one_hot(actions, num_actions)
        masked_qs = tf.reduce_sum(action_masks * qs, axis=-1)
        loss = mse(target, masked_qs)
    grads = tape.gradient(loss, main_nn.trainable_variables)
    optimizer.apply_gradients(zip(grads, main_nn.trainable_variables))
    return loss


def save_nn(name,main_nn):

    save_folder = NN_PATH + "/" + name

    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    ws = main_nn.get_weights()

    for i,w in enumerate(ws):
        filename = "weights" + str(i)
        np.save(save_folder + "/" + filename,w)

    filename = "actions.txt"
    np.savetxt(save_folder + "/" + filename,np.asarray(my_possible_actions),fmt='%.2f')

    filename = "config.json"

    conf = {
        "times_for_action":times_for_action,
        "discount":discount,
    }

    f = open(save_folder + "/" + filename,"wt")
    json.dump(conf,f)

def load_weights(name):

    save_folder = NN_PATH + "/" + name

    if not os.path.isdir(save_folder):
        return None

    files = glob.glob(save_folder + "/**.npy")

    files.sort()
    ws = []

    for f in files:
        print(f"loading {f}")
        ws.append(np.load(f))
    
    for w in ws:
        print(w.shape)

    return ws

def load_actions(name):
    save_folder = NN_PATH + "/" + name

    if not os.path.isdir(save_folder):
        return None

    filename = "actions.txt"
    return np.loadtxt(save_folder + "/" + filename)

def prepare_state(st):
    state = np.asarray(st[:,:,1])/256
    state = state.flatten()
    return state
