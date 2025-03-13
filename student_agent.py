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
global stations, candidates_p,candidates_goal, pickup, goal_id,last_action
stations = [[0,0] for _ in range(4)]
candidates_p = [i for i in stations]
candidates_goal = [i for i in stations]
goal_id = -1
pickup=False
action_size = 6
pickup_id = 4
drop_id = 5
np.random.seed(42)
def cmp(a,b):
    return a-b
    """if a>b:
        return 1
    if a<b:
        return -1
    return 0"""
last_action = None
def get_state_obs(obs,action):
    global stations,goal_id,pickup,candidates_p,candidates_goal
    #print(candidates_p)
    obstacles = [0 for _ in range(5)]
    taxi_row, taxi_col, stations[0][0], stations[0][1] , stations[1][0], stations[1][1],stations[2][0],stations[2][1],stations[3][0],stations[3][1],obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs
    agent_pos = (taxi_row,taxi_col)
    if action==None:
        # initialize
        candidates_goal = [tuple(i) for i in stations]
        candidates_p = [tuple(i) for i in stations]
        goal_id = np.random.randint(4)
        pickup=False
        relative_pos = (cmp(taxi_row,candidates_p[goal_id][0]),cmp(taxi_col,candidates_p[goal_id][1]))
        #print(f'Initialize\n\np : {candidates_p}, goal : {candidates_goal}, pickup : {pickup}, goal_id, {goal_id},p_loc{p_loc}')
        return (relative_pos,pickup, goal_id,len(candidates_p), len(candidates_goal), passenger_look, (obstacle_north,obstacle_south,obstacle_east,obstacle_west))
    if not pickup:
        #print(candidates_p,goal_id)
        relative_pos = (cmp(taxi_row,candidates_p[goal_id][0]),cmp(taxi_col,candidates_p[goal_id][1]))
    else:
        #print(candidates_goal,goal_id)
        relative_pos = (cmp(taxi_row,candidates_goal[goal_id][0]),cmp(taxi_col,candidates_goal[goal_id][1]))
    if action==pickup_id and not pickup and passenger_look and agent_pos in candidates_p:
        pickup = True
        candidates_p = []
        #print('pickup at',agent_pos)
        goal_id = np.random.choice(len(candidates_goal))
    elif action == drop_id and pickup:
        pickup=False
        candidates_p = [agent_pos]
        #print('drop at',agent_pos)
        goal_id = 0
    
    elif relative_pos == (0,0):
        #print(f'Before p : {candidates_p}, goal : {candidates_goal}, pickup : {pickup}, goal_id, {goal_id}')
        if passenger_look:
            candidates_p = [agent_pos]
            if not pickup:
                goal_id = 0
        else:
            #print(agent_pos,'not passenger look')
            if agent_pos in candidates_p:
                candidates_p.remove(agent_pos)
            if not pickup:
                try:
                    goal_id = np.random.choice(len(candidates_p))
                except:
                    print(candidates_p,agent_pos)
                    goal_id = np.random.choice(len(candidates_p))
        if destination_look:
            candidates_goal = [agent_pos]
            if pickup:
                goal_id = 0
        else:
            if agent_pos in candidates_goal:
                candidates_goal.remove(agent_pos)
            if pickup:
                goal_id = np.random.choice(len(candidates_goal))
        #print(f'After p : {candidates_p}, goal : {candidates_goal}, pickup : {pickup}, goal_id, {goal_id}')
        #print(f'Real passenger : {p_loc},ifpick up{ifpickup}')
        """if action==pickup_id and ifpickup:
            print(agent_pos,candidates_p)
            print(agent_pos in candidates_p)"""
    #print(candidates_p)
    relative_pos = (cmp(taxi_row,stations[goal_id][0]),cmp(taxi_col,stations[goal_id][1]))
    look_tag = passenger_look if not pickup else destination_look
    #print(f'p : {candidates_p}, goal : {candidates_goal}, pickup : {pickup}, goal_id, {goal_id},passenger_look {passenger_look}')
    return (relative_pos,pickup, goal_id,len(candidates_p), len(candidates_goal), look_tag, (obstacle_north,obstacle_south,obstacle_east,obstacle_west))
def get_action(obs):
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.
    global last_action
    state = get_state_obs(obs,last_action)
    if state not in q_table.keys():
        #assert(0)
        action = np.random.randint(action_size)
    else:
        
        action = np.argmax(q_table[state])
    last_action=action
    return action # Choose a random action
    # You can submit this random agent to evaluate the performance of a purely random strategy.

