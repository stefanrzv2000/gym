import tensorflow as tf
import numpy as np
import random
import os
import glob
import time
import json

from car_racing import CarRacing
from self_configs import NN_PATH, my_possible_actions, num_actions, num_episodes, epsilon, batch_size, max_ep_frames, penalty_factor, discount, buff_size, times_for_action
from CarNNutils import CarNN, optimizer, mse, mydeq, ReplayBuffer, train_step, save_nn, load_weights, load_actions, prepare_state

# import memdebug
# memdebug.start(11223)

print("Hello tf!")

env = CarRacing()
env.verbose = False

# my_possible_actions_stef = [
#     [0,0,0],
#     [0,1,0],
#     [0,0,0],
#     [0,1,0],
#     [0,0,0],
#     [0,1,0],
#     [0,0,0],
#     [0,1,0],
#     [0,0,0],
#     [0,1,0],
#     [0,0,0.5],
# ]

# for i in [1,0.5,0,-0.5,-1]:
#     my_possible_actions_stef.append([i,0,0])

# my_possible_actions_astrid = [
#     [0,1,0],
#     [0,1,0],
#     [0,0,0],
#     [0,0,0],
#     [-1,0,0],
#     [-1,0,0],
#     [-1,0,0],
#     [1,0,0],
#     [1,0,0],
#     [0,0,0.5],
# ]

# NN_PATH = "D:/Info 2018/RN/formula1/networks"
# my_possible_actions = my_possible_actions_stef
# if not os.path.isdir(NN_PATH):
#     NN_PATH = "C:/Users/Astrid/PycharmProjects/gym/networks"
#     my_possible_actions = my_possible_actions_astrid

print(my_possible_actions)

num_actions = len(my_possible_actions)

# class CarNN(tf.keras.Model):
#     """Dense neural network class."""
#     def __init__(self,outs):
#         super(CarNN, self).__init__()
#         self.dense1 = tf.keras.layers.Dense(256, activation="relu")
#         self.dense2 = tf.keras.layers.Dense(64, activation="relu")
#         self.dense3 = tf.keras.layers.Dense(outs, dtype=tf.float32) # No activation
        
#     def call(self, x):
#         """Forward pass."""
#         # x = tf.keras.backend.flatten(x)
#         x = self.dense1(x)
#         x = self.dense2(x)
#         return self.dense3(x)

main_nn = CarNN(num_actions)
target_nn = CarNN(num_actions)
buffer = ReplayBuffer(buff_size)

# optimizer = tf.keras.optimizers.Adam(1e-4)
# mse = tf.keras.losses.MeanSquaredError()


def select_epsilon_greedy_action(state, epsilon, verbose=False):
    """Take random action with probability epsilon, else take best action."""
    result = tf.random.uniform((1,))
    if result < epsilon:
        return random.randint(0,len(my_possible_actions)-1) # Random action (left or right).
    else:
        #print(state.shape)
        state = np.asarray([state])
        scores = main_nn(state)
        act = tf.argmax(scores[0])
        if verbose:
            print("scores",scores)
            print("action",act,my_possible_actions[act])
        return act # Greedy action for state.

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

# Start training. Play game once and then train with a batch.

def get_penalty(state,action):

    speed = np.sum(state[84:,12,1])
    green = np.mean(state[60:80,38:58,1])

    pen = 0

    if speed > 255*2 and action[1] > 0:
        pen += action[1] * (speed/255-2) * 0.2

    if green > 100:
        pen += (green - 100) * 0.3

    if speed < 255 and action[1] == 0:
        pen += 1

    return pen

def apply_action(act,times):
    rew = 0
    for t in range(times):
        next_state, reward, done, info = env.step(act)
        rew += reward

    return next_state,rew,done,info


## main stuff

show_image = False

start_time = time.time()

last_100_ep_rewards = mydeq(10)

testing = False
name = "trial5"
load_name = "temp_trial4"
if load_name and testing:
    epsilon = 0.1
    show_image = True

loaded = False

curr_frame = 0
for episode in range(num_episodes+1):

    state = env.reset()
    # env.close()

    if episode == 0:
        if load_name:
            my_possible_actions = load_actions(load_name)
            num_actions = len(my_possible_actions)
            main_nn = CarNN(num_actions)
            target_nn = CarNN(num_actions)

        main_nn(np.asarray([prepare_state(state)]))
        target_nn(np.asarray([prepare_state(state)]))

        if load_name and not loaded:
            main_nn.set_weights(load_weights(load_name))
            target_nn.set_weights(load_weights(load_name))
            loaded = True

    ep_reward, done = 0, False
    ep_frames = 0
    ep_penalty = 0
    startt = time.time()

    isopen = True
    while not done and ep_frames < max_ep_frames and isopen:
        #state_in = tf.expand_dims(state, axis=0)
        curr_state = prepare_state(state)
        
        act = select_epsilon_greedy_action(curr_state, epsilon, verbose=False)
        action = my_possible_actions[act]
        #print(action)
        next_state, reward, done, info = apply_action(action,times_for_action)

        penalty = get_penalty(state,action) * penalty_factor
        ep_penalty += penalty

        if show_image:
            isopen = env.render()
        ep_reward += reward
        ep_frames += 1
        
        #print(sys.getsizeof(buffer))

        next_state_ = prepare_state(next_state)
        
        # Save to experience replay.
        buffer.add(curr_state, act, reward - penalty, next_state_, done)
        #print("buffer",len(buffer))
        state = next_state
        curr_frame += 1
        # Copy main_nn weights to target_nn.
        if curr_frame % 3000 == 0:
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
            f'Last ep penalty {ep_penalty:.2f}. Reward in last 10 episodes: {last_100_ep_rewards.mean():.3f}\n')

    if episode % 10 == 0:
        print(f"Time unitll episode {episode}: {time.time() - start_time:.2f}\n")

    if episode % 100 == 0 and episode > 0:
        save_nn("temp_" + name, main_nn)

save_nn(name, main_nn)

env.close()
