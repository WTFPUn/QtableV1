#! /usr/bin/env python

import numpy as np
from math import *
from std_msgs.msg import String
from itertools import product
from sensor_msgs.msg import LaserScan
import time

STATE_SPACE_IND_MAX = 27648 - 1
STATE_SPACE_IND_MIN = 1 - 1
ACTIONS_IND_MAX = 7
ACTIONS_IND_MIN = 0

ANGLE_MAX = 360 - 1
ANGLE_MIN = 1 - 1
# HORIZON_WIDTH = 75 original
HORIZON_WIDTH = [9, 16, 56, 9]

T_MIN = 0.001

# Create actions
def createActions(n_actions_enable = 5):
    actions = np.arange(n_actions_enable)
    return actions
# forward, CW, CCW, superForward, stop

# Create state space 
def createStateSpace(x = 10):
    States = np.arange(x)
    return States


# Reward function for Q-learning - table
def getReward(  action, 
                prev_action,
                lidar, 
                prev_lidar, 
                crash, 
                current_position, 
                goal_position, 
                max_radius, 
                radius_reduce_rate, 
                nano_start_time, 
                nano_current_time, 
                goal_radius, 
                angle_state):

    terminal_state = False
    
    # init reward
    reward = 0

    # to do in learning_node file
    # add time start for each episode
    # add position start for each episode
    # add current position for each step
    # add goal position for each episode
    # add max radius for each episode
    # add radius reduce rate for each episode
    
    # time penalty 
    dist = np.linalg.norm(np.array(current_position) - np.array(goal_position))
    #  nano time diff
    time_diff = (nano_current_time - nano_start_time)
    radius = max_radius - radius_reduce_rate * (time_diff)
    if radius/max_radius < 0.1:
        radius = max_radius * 0.1
    if dist < radius:
        reward += .1
    else:
        reward += - .69

    # Crash panelty
    if crash:
        reward += -500

    # facing wall panelty/rewards
    lidar_horizon = np.concatenate((lidar[(ANGLE_MIN + sum(HORIZON_WIDTH[:2])):(ANGLE_MIN):-1],lidar[(ANGLE_MAX):(ANGLE_MAX - sum(HORIZON_WIDTH[:2])):-1]))
    prev_lidar_horizon = np.concatenate((prev_lidar[(ANGLE_MIN + sum(HORIZON_WIDTH[:2])):(ANGLE_MIN):-1],prev_lidar[(ANGLE_MAX):(ANGLE_MAX - sum(HORIZON_WIDTH[:2])):-1]))
    W = np.linspace(1, 1.1, len(lidar_horizon) // 2)
    W = np.append(W, np.linspace(1.1, 1, len(lidar_horizon) // 2))
    if np.sum( W * ( lidar_horizon - prev_lidar_horizon) ) >= 0:
        reward += +0.2
    else:
        reward += -0.2
        
    # action and prev_action is same and action is left or right
    if (prev_action == 3 and action == 4) or (prev_action == 4 and action == 3):
            reward += -5

    #repeat stop penelty
    if prev_action == 2 and action == 2:
            reward += -5

    #reach goal
    if dist<goal_radius:
        reward += 100
        terminal_state = True
    
    #away from goal panelty
    if angle_state == 0:
        reward += -1

    #facing goal reward
    elif angle_state == 1:
        reward += 5

    elif angle_state == 2:
        reward += 1

    elif angle_state == 3:
        reward += 1

    # calculate distance reward
    reward +=  3* (np.exp(-dist) - np.exp(-max_radius)) / (1 - np.exp(-max_radius))

    return (reward, terminal_state)

