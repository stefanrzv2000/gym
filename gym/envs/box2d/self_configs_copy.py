NN_PATH = "D:/Info 2018/RN/formula1/networks"

my_possible_actions_stef = [
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
    my_possible_actions_stef.append([i,0,0])

my_possible_actions = my_possible_actions_stef
num_actions = len(my_possible_actions)

num_episodes = 1000
epsilon = 0.5
batch_size = 32
max_ep_frames = 2000
penalty_factor = 0.05
discount = 0.99
buff_size = 10000
times_for_action = 3