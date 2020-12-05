import tensorflow as tf
import gym
import matplotlib.pyplot as plt
import Box2D
import numpy as np
import random

from collections import deque
import sys
import os
import glob
import time
from car_racing import CarRacing

# import memdebug
# memdebug.start(11223)

NN_PATH = "D:/Info 2018/RN/formula1/networks"

print("Hello tf!")

env = CarRacing()
env.verbose = False

my_possible_actions = [
    [0,0,0],
    [0,1,0],
    [0,0,0],
    [0,1,0],
    [0,0,0],
    [0,1,0],
    [0,0,0],
    [0,1,0],
    [0,0,0],
    [0,1,0],
    [0,0,0.5],
]

for i in [1,0.5,0,-0.5,-1]:
    my_possible_actions.append([i,0,0])

# my_possible_actions = [
#     [0,1,0],
#     [0,0,0],
#     [-1,0,0],
#     [1,0,0]
# ]

print(my_possible_actions)

num_actions = len(my_possible_actions)

class CarNN(tf.keras.Model):
    """Dense neural network class."""
    def __init__(self):
        super(CarNN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(256, activation="relu")
        self.dense2 = tf.keras.layers.Dense(64, activation="relu")
        self.dense3 = tf.keras.layers.Dense(num_actions, dtype=tf.float32) # No activation
        
    def call(self, x):
        """Forward pass."""
        # x = tf.keras.backend.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)

main_nn = CarNN()
target_nn = CarNN()

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

def select_epsilon_greedy_action(state, epsilon):
    """Take random action with probability epsilon, else take best action."""
    result = tf.random.uniform((1,))
    if result < epsilon:
        return random.randint(0,len(my_possible_actions)-1) # Random action (left or right).
    else:
        #print(state.shape)
        state = np.asarray([state])
        return tf.argmax(main_nn(state))[0] # Greedy action for state.

@tf.function
def train_step(states, actions, rewards, next_states, dones):
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

# Hyperparameters.
num_episodes = 1000
epsilon = 1.0
batch_size = 32
discount = 0.99
buffer = ReplayBuffer(1001)
curr_frame = 0

def prepare_state(st):
    state = np.asarray(st[:,:,1])/256
    state = state.flatten()
    return state

def save_main_nn(name):

    save_folder = NN_PATH + "/" + name

    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    ws = main_nn.get_weights()

    for i,w in enumerate(ws):
        filename = "weights" + str(i)
        np.save(save_folder + "/" + filename,w)

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

# Start training. Play game once and then train with a batch.

def get_penalty(state,action):

    speed = np.sum(state[84:,12,1])
    green = np.mean(state[60:80,38:58,1])

    pen = 0

    if speed > 255*2 and action[1] > 0:
        pen += action[1] * (speed/255-2)

    if green > 100:
        pen += (green - 100) * 0.1

    return pen


def do_stuff():

    start_time = time.time()

    last_100_ep_rewards = mydeq(10)
    num_episodes = 1500
    epsilon = 0.9
    batch_size = 32
    max_ep_frames = 5000
    penalty_factor = 0.03

    name = "trial3"
    load_name = None
    loaded = False
    
    buffer = ReplayBuffer(1001)
    curr_frame = 0
    for episode in range(num_episodes+1):

        if load_name and episode > 0 and not loaded:
            main_nn.set_weights(load_weights(load_name))
            target_nn.set_weights(load_weights(load_name))
            loaded = True

        state = env.reset()
        # env.close()
        ep_reward, done = 0, False
        ep_frames = 0
        ep_penalty = 0
        startt = time.time()

        isopen = True
        while not done and ep_frames < max_ep_frames and isopen:
            #state_in = tf.expand_dims(state, axis=0)

            curr_state = prepare_state(state)
            
            act = select_epsilon_greedy_action(curr_state, epsilon)
            action = my_possible_actions[act]
            #print(action)
            next_state, reward, done, info = env.step(action)

            penalty = get_penalty(state,action) * penalty_factor
            ep_penalty += penalty

            # isopen = env.render()
            ep_reward += reward
            ep_frames += 1
            
            #print(sys.getsizeof(buffer))

            next_state_ = prepare_state(next_state)
            
            # Save to experience replay.
            buffer.add(curr_state, act, reward, next_state_, done)
            #print("buffer",len(buffer))
            state = next_state
            curr_frame += 1
            # Copy main_nn weights to target_nn.
            if curr_frame % 2000 == 0:
                print("Copying weights")
                target_nn.set_weights(main_nn.get_weights())

            # Train neural network.
            # if len(buffer) >= batch_size and curr_frame%100 == 0:
            #     #print("training now")
            #     states, actions, rewards, next_states, dones = buffer.sample(batch_size*10)
            #     loss = train_step(states, actions, rewards, next_states, dones)

        # ws = main_nn.get_weights()
        # print(len(ws))


        print("ep_frames",ep_frames)
        duration = time.time() - startt
        print(f"FPS: {ep_frames/duration:.2f}")
        print("training now")

        states, actions, rewards, next_states, dones = buffer.get_all()
        loss1 = train_step(states, actions, rewards, next_states, dones)
        loss2 = train_step(states, actions, rewards, next_states, dones)
        loss3 = train_step(states, actions, rewards, next_states, dones)
        print(f"loss {loss1 + loss2 + loss3}")

        #save_main_nn("trial1")
        #target_nn.set_weights(load_weights("trial1"))

        if epsilon > 0.05:
            epsilon -= 0.001

        # if len(last_100_ep_rewards) == 100:
        #     last_100_ep_rewards = last_100_ep_rewards[1:]
        last_100_ep_rewards.append(ep_reward)
            
        if episode % 1 == 0:
            print(f'Episode {episode}/{num_episodes}. Epsilon: {epsilon:.3f}. Last ep reward: {ep_reward:.2f}. '
                f'Last ep penalty {ep_penalty:.2f}. Reward in last 10 episodes: {last_100_ep_rewards.mean():.3f}')

        if episode % 10 == 0:
            print(f"Time unitll episode {episode}: {time.time() - start_time:.2f}")

        if episode % 100 == 0:
            save_main_nn("temp_" + name)

    save_main_nn(name)

    env.close()

    

do_stuff()