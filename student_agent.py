# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
import pickle
from collections import defaultdict
"""
q_table trained by https://colab.research.google.com/gist/089487/bd898be4a527f5513a5272622a8f17c2/drl_assignment1_q4.ipynb
"""
#np.random.seed(42)
#random.seed(42)
with open('q_table.pkl', 'rb') as f:
    print('load')
    loaded_dict = pickle.load(f)
    q_table = defaultdict(lambda: np.zeros(6), loaded_dict)  # Replace 0 with your default value

#print('len of q_table',len(q_table.keys()))
global stations, candidates_p,candidates_goal, pickup, last_action
stations = [[0,0] for _ in range(4)]
candidates_p = [i for i in stations]
candidates_goal = [i for i in stations]
goal_id = -1
pickup=False
action_size = 6
last_action = None
pickup_id = 4
drop_id = 5
def cmp(a,b):
    #return a-b
    if a>b:
        return 1
    if a<b:
        return -1
    return 0
def get_state_obs(obs,action):
    global stations,pickup,candidates_p,candidates_goal
    #print(candidates_p)
    obstacles = [0 for _ in range(5)]
    taxi_row, taxi_col, stations[0][0], stations[0][1] , stations[1][0], stations[1][1],stations[2][0],stations[2][1],stations[3][0],stations[3][1],obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs
    agent_pos = (taxi_row,taxi_col)
    if action==None:
        # initialize
        candidates_goal = [tuple(i) for i in stations]
        candidates_p = [tuple(i) for i in stations]
        pickup=False
    if passenger_look:
        #print('before p',candidates_p)
        candidates_p = [ tuple(x) for x in candidates_p if abs(x[0]-agent_pos[0])+abs(x[1]-agent_pos[1]) <=1 ]
        #print('after p',candidates_p)
    else:
        #print('before p',candidates_p)
        candidates_p = [ tuple(x) for x in candidates_p if abs(x[0]-agent_pos[0])+abs(x[1]-agent_pos[1]) >1 ]
        #print('after p',candidates_p)
    if destination_look:
        #print('before g',candidates_goal)
        candidates_goal = [ tuple(x) for x in candidates_goal if abs(x[0]-agent_pos[0])+abs(x[1]-agent_pos[1]) <=1 ]
        #print('after g',candidates_goal)
    else:
        #print('before g',candidates_goal)
        candidates_goal = [ tuple(x) for x in candidates_goal if abs(x[0]-agent_pos[0])+abs(x[1]-agent_pos[1]) >1 ]
        #print('after g',candidates_goal)
    if action==pickup_id and not pickup and agent_pos in candidates_p:
        pickup = True
        candidates_p = []
    elif action == drop_id and pickup:
        pickup=False
        candidates_p = [agent_pos]
    return (pickup, len(candidates_p), len(candidates_goal), passenger_look, destination_look, (obstacle_north,obstacle_south,obstacle_east,obstacle_west))
def get_action(obs):
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.
    global last_action
    state = get_state_obs(obs,last_action)
    #print(q_table.keys())
    #print(obs)
    #print(state)
    if state not in q_table.keys():
        print(state)
        assert(0)
        action = np.random.randint(action_size)
    else:
        #print(q_table[state])
        if np.random.choice(2,p=[0.05,0.95])==1:
            action = np.argmax(q_table[state])
        else:
            action = np.random.randint(6)
    last_action=action
    return action # Choose a random action
    # You can submit this random agent to evaluate the performance of a purely random strategy.

