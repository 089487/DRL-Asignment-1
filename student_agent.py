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
np.random.seed(42)
#random.seed(42)
with open('q_table.pkl', 'rb') as f:
    print('load')
    loaded_dict = pickle.load(f)
    q_table_list = loaded_dict  # Replace 0 with your default value
    q_table = {k:np.array(v) for k,v in q_table_list.items()}

#print('len of q_table',len(q_table.keys()))
global stations, candidates_p,candidates_goal, pickup, last_action, last_record_action
stations = [[0,0] for _ in range(4)]
candidates_p = [i for i in stations]
candidates_goal = [i for i in stations]
goal_id = -1
pickup=False
action_size = 6
last_action = None
last_record_action = None
pickup_id = 4
drop_id = 5
def cmp(a,b):
        if a==b:
            return 0
        return 1 if a<b else -1
            
def get_state_obs(obs,action,last_action=None):
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
        # choose the one that is closest to the agent
        idx = np.argmin([abs(agent_pos[0]-i[0])+abs(agent_pos[1]-i[1]) for i in candidates_p])
        cmp_pos = candidates_p[idx]
    else:
        # choose the one that is closest to the agent
        idx = np.argmin([abs(agent_pos[0]-i[0])+abs(agent_pos[1]-i[1]) for i in candidates_goal])
        cmp_pos = candidates_goal[idx]
    passenger_look = passenger_look and agent_pos in candidates_p
    destination_look = destination_look and agent_pos in candidates_goal
    real_look = passenger_look if not pickup else destination_look
    relative_pos = (cmp(agent_pos[0],cmp_pos[0]),cmp(agent_pos[1],cmp_pos[1]))
    return (relative_pos,pickup,real_look, (obstacle_north,obstacle_south,obstacle_east,obstacle_west),last_action)

def get_action(obs):
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys.
    #       Otherwise, even if your agent performs well in training, it may fail during testing.
    global last_action,last_record_action
    state = get_state_obs(obs,last_action,last_record_action)
    action_name = ['Move North','Move South','Move East','Move West','Pick Up','Drop Off']
    if state not in q_table.keys():
        #print(state)
        print(state)
        assert(0)
        action = np.random.randint(action_size)
    else:
        #print(state,action_name[np.argmax(q_table[state])])
        q_table[state] = np.array(q_table[state])
        action = np.argmax(q_table[state])
    last_action = action
    if action in [0,1,2,3]:
        last_record_action = action
    #q_table[state][action] = q_table[state][action] + 0.089487*(-0.1+0.89487*np.max(q_table[state])-q_table[state][action])
    return action # Choose a random action