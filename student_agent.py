# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
import pickle
from collections import defaultdict
with open('q_table.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
    q_table = defaultdict(lambda: 0, loaded_dict)  # Replace 0 with your default value
global stations
global goal_id
global pickup
stations = np.zeros((4,2))
goal_id = -1
pickup = False
pickup_id = 4
drop_id = 5
def get_state_obs(obs):
    global stations
    global goal_id
    global pickup
    taxi_row, taxi_col, stations[0][0], stations[0][1] , stations[1][0], stations[1][1],stations[2][0],stations[2][1],stations[3][0],stations[3][1],obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs
    if goal_id == -1:
        goal_id = np.random.randint(4)
    relative_pos = (taxi_row-stations[goal_id][0],taxi_col-stations[goal_id][1])
    return relative_pos,(obstacle_north,obstacle_south,obstacle_east,obstacle_west),passenger_look,destination_look,pickup
def get_action(obs):
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.
    state = get_state_obs(obs)
    global pickup
    relative_pos,_,passenger_look,_,_ = state
    if state not in q_table.keys():
        action = np.random.randint(6)
    else:
        action = np.argmax(q_table[state])
    if action==pickup_id and relative_pos==(0,0) and passenger_look:
        pickup = True
        next_goal_id = goal_id
        while next_goal_id == goal_id:
            next_goal_id = np.random.choice(4)
        goal_id = next_goal_id
    elif action == drop_id and pickup:
        pickup=False
    return action # Choose a random action
    # You can submit this random agent to evaluate the performance of a purely random strategy.

