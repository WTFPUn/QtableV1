#! /usr/bin/env python

import numpy as np
from math import *
from std_msgs.msg import String
from itertools import product
from sensor_msgs.msg import LaserScan
import time

STATE_SPACE_IND_MAX = 2916 - 1
STATE_SPACE_IND_MIN = 1 - 1
ACTIONS_IND_MAX = 3
ACTIONS_IND_MIN = 0

ANGLE_MAX = 360 - 1
ANGLE_MIN = 1 - 1

# HORIZON_WIDTH = 75 original
T_MIN = 0.001

# Create actions
def createActions(n_actions_enable):
    # actions = np.array([0,1,2,3,4,5,6,7])
    actions = np.arange(n_actions_enable)
    return actions
# forward, left, right,  superForward,

# Create state space for Q table
def createStateSpace():
    x1 = set((0,1,2))
    x2 = set((0,1,2))
    x3 = set((0,1,2))
    x4 = set((0,1,2))
    x5 = set((0,1,2))
    x6 = set((0,1,2))
    x7 = set((0,1,2,3))

    state_space = set(product(x1,x2,x3,x4,x5,x6,x7))
    return np.array(list(state_space))

# Create Q table, dim: n_states x n_actions
def createQTable(n_states, n_actions):
    # Q_table = np.random.uniform(low = -1, high = 1, size = (n_states,n_actions) )
    Q_table = np.zeros((n_states, n_actions))
    return Q_table

# Read Q table from path
def readQTable(path):
    Q_table = np.genfromtxt(path, delimiter = ' , ')
    return Q_table

# Write Q table to path
def saveQTable(path, Q_table):
    np.savetxt(path, Q_table, delimiter = ' , ')

# Select the best action a in state
def getBestAction(Q_table, state_ind, actions):
    if STATE_SPACE_IND_MIN <= state_ind <= STATE_SPACE_IND_MAX:
        status = 'getBestAction => OK'
        a_ind = np.argmax(Q_table[state_ind,:])
        a = actions[a_ind]
    else:
        status = 'getBestAction => INVALID STATE INDEX'
        a = getRandomAction(actions)

    return ( a, status )

# Select random action from actions
def getRandomAction(actions):
    n_actions = len(actions)
    a_ind = np.random.randint(n_actions)
    return actions[a_ind]

# Epsilog Greedy Exploration action chose
def epsiloGreedyExploration(Q_table, state_ind, actions, epsilon):
    if np.random.uniform() > epsilon and STATE_SPACE_IND_MIN <= state_ind <= STATE_SPACE_IND_MAX:
        status = 'epsiloGreedyExploration => OK'
        ( a, status_gba ) = getBestAction(Q_table, state_ind, actions)
        if status_gba == 'getBestAction => INVALID STATE INDEX':
            status = 'epsiloGreedyExploration => INVALID STATE INDEX'
    else:
        status = 'epsiloGreedyExploration => OK'
        a = getRandomAction(actions)

    return ( a, status )

# SoftMax Selection
def softMaxSelection(Q_table, state_ind, actions, T):
    if STATE_SPACE_IND_MIN <= state_ind <= STATE_SPACE_IND_MAX:
        status = 'softMaxSelection => OK'
        n_actions = len(actions)
        P_ac = np.zeros(n_actions)

        # Boltzman distribution
        P_ac = np.exp(Q_table[state_ind,:] / T) / np.sum(np.exp(Q_table[state_ind,:] / T))

        if T < T_MIN or np.any(np.isnan(P_ac)):
            ( a, status_gba ) = getBestAction(Q_table, state_ind, actions)
            if status_gba == 'getBestAction => INVALID STATE INDEX':
                status = 'softMaxSelection => INVALID STATE INDEX'
        else:
            # rnd = np.random.uniform()
            status = 'softMaxSelection => OK'
            try:
                a = np.random.choice(n_actions, 1, p = P_ac)
            ###################################    
            except:
                status = 'softMaxSelection => Boltzman distribution error => getBestAction '
                status = status + '\r\nP = (%f , %f , %f, %f, %f, %f, %f, %f) ' % (P_ac[0],P_ac[1],P_ac[2],P_ac[3])
                status = status + '\r\nQ(%d,:) = ( %f , %f , %f, %f, %f, %f, %f, %f) ' % (state_ind, Q_table[state_ind,0], Q_table[state_ind,1], Q_table[state_ind,2], Q_table[state_ind,3])
                ( a, status_gba ) = getBestAction(Q_table, state_ind, actions)
                if status_gba == 'getBestAction => INVALID STATE INDEX':
                    status = 'softMaxSelection => INVALID STATE INDEX'
    else:
        status = 'softMaxSelection => INVALID STATE INDEX'
        a = getRandomAction(actions)

    return ( a, status )

# Reward function for Q-learning - table
def getReward(  action, 
                prev_action, 
                crash, 
                current_position, 
                goal_position, 
                max_radius,   
                goal_radius, 
                n_action,
                state,
                lidar
                ):


    terminal_state = False
    # init reward
    reward = 0
    
    # deconstruct state
    [x1, x2, x3, x4, x5, x6, x7] = state

    # time penalty 
    dist = np.linalg.norm(np.array(current_position) - np.array(goal_position))
    #  nano time diff
    step_factor = (500-n_action) / 500
    radius = max_radius * step_factor
    if radius/max_radius < 0.1:
        radius = max_radius * 0.1
    if dist < radius:
        reward += 1
    else:
        reward += -1

    # Crash panelty
    if crash:
        reward += -200
        terminal_state = True
   
    # penalty if x1 and x5 state is too different(2 level)
    if abs(x1-x5) ==2 :
        reward += -3
    elif abs(x1-x5) ==1 :
        reward += -1
    else:
        reward += 0

    # penalty if x3 and x6 state is too different(2 level)
    if x3-x6 ==2 :
        reward += 3
    elif x3-x6 ==1 :
        reward += 1
    elif x3-x6 ==0 :
        reward += 0
    elif x3-x6 == -1 :
        reward += -1
    else:
        reward += -3


    # penalty if x2 on 0 state(almost crash)
    if x2 ==0 :
        reward += -3
    elif x2 ==1 :
        reward += -0.5
    else:
        reward += 0

    # penalty if x3 on 0 state(almost crash)
    if x3 ==0 :
        reward += -3
    elif x4 ==1 :
        reward += -0.5
    else:
        reward += 1        
    
    # penalty if x4 on 0 state(almost crash)
    if x4 ==0 :
        reward += -3
    elif x4 ==1 :
        reward += -0.5
    else:
        reward += 0
        
    # action and prev_action is same and action is left or right
    if (prev_action == 1 and action == 2) or (prev_action == 2 and action == 1):
        reward += -1

    #reach goal
    if dist<goal_radius:
        reward += 50
        terminal_state = True
    

    #away from goal panelty
    if x7 == 0:
        reward += -0.5

    #facing goal reward
    elif x7 == 1:
        reward += 1

    elif x7 == 2:
        reward += 0.5

    elif x7 == 3:
        reward += 0.5

    # if add reward forward or super forward
    if action == 0 or action == 3:
        reward += 3
    else :
        reward += -0.5

    # calculate distance reward
    reward +=  3* (np.exp(-dist) - np.exp(-max_radius)) / (1 - np.exp(-max_radius))
    
    # length_lidar = len(lidar) 
    # ratio = length_lidar / 360 
    # Angle_det = 24
    # lidar_x1 = min(lidar[round(ratio*(90- Angle_det/2)): round(ratio*(90+ Angle_det/2+1))])
    # lidar_x5 = min(lidar[round(ratio*(270- Angle_det/2)): round(ratio*(270+ Angle_det/2+1))])

    if x3 == 1:
        if x1 >= x5 and x1 >=1:
            if action == 2:
                reward += 3
            else : 
                reward += -1
        elif x1 <= x5 and x5 >=1:
            if action == 1:
                reward += 3
            else : 
                reward += -1
    # print("x3 is: ", x3, "\nx1: ", x1 , "x5: ", x5)


        # if max(lidar_x1, lidar_x5) == lidar_x1 and action == 2 :
        
    return (reward, terminal_state)


# Update Q-table values
def updateQTable(Q_table, state_ind, action, reward, next_state_ind, alpha, gamma):
    if STATE_SPACE_IND_MIN <= state_ind <= STATE_SPACE_IND_MAX and STATE_SPACE_IND_MIN <= next_state_ind <= STATE_SPACE_IND_MAX:
        status = 'updateQTable => OK'
        Q_table[state_ind,action] = ( 1 - alpha ) * Q_table[state_ind,action] + alpha * ( reward + gamma * max(Q_table[next_state_ind,:]) )
    else:
        status = 'updateQTable => INVALID STATE INDEX'
    return ( Q_table, status )

