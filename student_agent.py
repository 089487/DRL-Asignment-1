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
    q_table = loaded_dict  # Replace 0 with your default value

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
        if a==b:
            return 0
        return 1 if a<b else -1
            
def get_state_obs(obs,action):
    global stations,pickup,candidates_p,candidates_goal
    #print(candidates_p)
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
        candidates_p.append(agent_pos)
    cmp_pos = (0,0)
    if not pickup:
        cmp_pos = candidates_p[0]
    else:
        cmp_pos = candidates_goal[0]
    passenger_look = passenger_look and agent_pos in candidates_p
    destination_look = destination_look and agent_pos in candidates_goal
    relative_pos = (cmp(agent_pos[0],cmp_pos[0]),cmp(agent_pos[1],cmp_pos[1]))
    return (relative_pos,pickup, passenger_look, destination_look, (obstacle_north,obstacle_south,obstacle_east,obstacle_west),action)

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
        action = np.argmax(q_table[state])
    next_state = get_state_obs(obs,action)
    #q_table[state][action] = q_table[state][action] + 0.089487*(-0.1+0.89487*np.max(q_table[next_state])-q_table[state][action])
    last_action=action
    return action # Choose a random action
    # You can submit this random agent to evaluate the performance of a purely random strategy.

