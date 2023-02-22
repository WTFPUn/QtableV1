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
def getReward(  crash, 
                current_position, 
                goal_position, 
                n_action,
                max_radius,  
                goal_radius,
                action_mode):

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
    step_factor = (500-n_action) / 500
    radius = max_radius * step_factor
    if radius/max_radius < 0.1:
        radius = max_radius * 0.1
    if dist < radius:
        reward += 1
    else:
        reward += -1
        
    if action_mode == 'Up2U':
        reward += -1

    if crash:
        # terminal_state = True
        reward += -100

    #reach goal
    if dist<goal_radius:
        reward += 100
        terminal_state = True
    
    # calculate distance reward
    reward +=  5* (np.exp(-dist) - np.exp(-max_radius)) / (1 - np.exp(-max_radius))

    return (reward, terminal_state)

