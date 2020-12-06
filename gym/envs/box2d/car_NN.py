import tensorflow as tf
import numpy as np
import random
import time

from car_racing import CarRacing
from self_configs import NN_PATH, my_possible_actions, num_actions, num_episodes, epsilon, batch_size, max_ep_frames, penalty_factor, discount, buff_size, times_for_action, train_steps
from CarNNutils import CarNN, optimizer, mse, mydeq, ReplayBuffer, train_step, save_nn, load_weights, load_actions, prepare_state

print("Hello tf!")

env = CarRacing()
env.verbose = False

print(my_possible_actions)

num_actions = len(my_possible_actions)

main_nn = CarNN(num_actions)
target_nn = CarNN(num_actions)
buffer = ReplayBuffer(buff_size)

def softmax(vec):
    ex = np.exp2(vec-np.min(vec))
    return ex/np.sum(ex)

def select_epsilon_greedy_action(state, epsilon, verbose=False):
    """Take random action with probability epsilon, else take best action."""
    result = tf.random.uniform((1,))
    if result < epsilon:
        return random.randint(0,len(my_possible_actions)-1) # Random action (left or right).
    else:
        #print(state.shape)
        state = np.asarray([state])
        scores = main_nn(state)
        wscores = softmax(scores[0])
        act = random.choices(range(len(wscores)), weights=wscores)[0]
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
        if done:
            return next_state, rew, done, info

    return next_state,rew,done,info

def track_coverage(frames, score):

    if frames < max_ep_frames:
        score += 100

    score += times_for_action*frames*0.1
    return score/10

## main stuff

show_image = False

start_time = time.time()

last_100_ep_rewards = mydeq(10)

testing = False
name = "trial5"
load_name = None
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
    max_ep_reward = 0
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
            if not isopen:
                env.reset()
                env.render()

        ep_reward += reward
        max_ep_reward = ep_reward if ep_reward > max_ep_reward else max_ep_reward
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

    duration = time.time() - startt

    states, actions, rewards, next_states, dones = buffer.get_all()

    loss = 0
    for i in range(train_steps):
        loss += train_step(states, actions, rewards, next_states, dones)

    # print(f"loss {loss}")

    #save_main_nn("trial1")
    #target_nn.set_weights(load_weights("trial1"))

    if epsilon > 0.05:
        epsilon -= 0.001

    # if len(last_100_ep_rewards) == 100:
    #     last_100_ep_rewards = last_100_ep_rewards[1:]
    last_100_ep_rewards.append(ep_reward)

    if episode % 1 == 0:
        print(f'Episode {episode}/{num_episodes}.')
        print("ep_frames",ep_frames)
        print(f"FPS: {ep_frames/duration:.2f}")
        print(f'Epsilon: {epsilon:.3f}.\nLast ep reward: {ep_reward:.2f}. '
            f'Last ep penalty: {ep_penalty:.2f}.')
        print(f'Max ep reward: {max_ep_reward:.2f} Track coverage: {track_coverage(ep_frames,ep_reward):.2f}%')
        print(f'Reward in last 10 episodes: {last_100_ep_rewards.mean():.3f}\n')

    if episode % 10 == 0:
        print(f"Time unitll episode {episode}: {time.time() - start_time:.2f}\n")

    if episode % 100 == 0 and episode > 0:
        save_nn("temp_" + name, main_nn)

save_nn(name, main_nn)

env.close()
