#! /usr/bin/env python

import numpy as np
from math import *
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan

MAX_LIDAR_DISTANCE = 2.0
COLLISION_DISTANCE = 0.125 # LaserScan.range_min = 0.1199999
NEARBY_DISTANCE = 0.45

ZONE_0_LENGTH = .2
ZONE_1_LENGTH = .5

ANGLE_MAX = 360  #360  degree
ANGLE_MIN = 1 - 1   #0 degree
ANGLE_BACK = 180  #180 degree_
# HORIZON_WIDTH = 75  #original
# HORIZON_WIDTH = [9, 16, 65, 9] #9:x1, x2, x7   16:x3, x4   25:x5, x6 

# Convert LasecScan msg to array
def lidarScan(msgScan):
    distances = np.array([])
    angles = np.array([])

    for i in range(len(msgScan.ranges)):
        angle = degrees(i * msgScan.angle_increment)
        if ( msgScan.ranges[i] > MAX_LIDAR_DISTANCE ):
            distance = MAX_LIDAR_DISTANCE
        elif ( msgScan.ranges[i] < msgScan.range_min ):
            distance = msgScan.range_min
            # For real robot - protection
            if msgScan.ranges[i] < 0.01:
                distance = MAX_LIDAR_DISTANCE
        else:
            distance = msgScan.ranges[i]

        distances = np.append(distances, distance)
        angles = np.append(angles, angle)

    # distances in [m], angles in [degrees]
    return ( distances, angles )

def LidarGoGoPowerRanger(lidarVal):
    if lidarVal > ZONE_1_LENGTH:
        return 2
    elif ZONE_1_LENGTH >= lidarVal > ZONE_0_LENGTH:
        return 1
    else:
        return 0

    

# Discretization of lidar scan
def scanDiscretization(state_space, lidar, target_pos, robot_pose, robot_prev_pose, goal_radius):
    ### now --> 72*4 stage
    x1 = 2  # no obstacle
    x2 = 2
    x3 = 2
    x4 = 2
    x5 = 2 
    x6 = 2
    x7 = 1
    print()

    length_lidar = len(lidar) 
    ratio = length_lidar / 360 
    Angle_det = 24
    # Lidar discretization
    
    # Zone 1: 78: 102+1
    # lidar_x1 = min(lidar[round(ratio*(ANGLE_MAX  - HORIZON_WIDTH[0] ))
    lidar_x1 = min(lidar[round(ratio*(90- Angle_det/2)): round(ratio*(90+ Angle_det/2+1))])
    x1 = LidarGoGoPowerRanger(lidar_x1)

    # Zone 2: 12: 36+1
    lidar_x2 = min(lidar[round(ratio*( Angle_det/2 )): round(ratio*(Angle_det/2+ Angle_det+1))])
    x2 = LidarGoGoPowerRanger(lidar_x2)


    # Zone 3-1: 0: 12+1
    lidar_x3_1 = min(lidar[round(ratio*(0)): round(ratio*(Angle_det/2+1))])
    # Zone 3-2: 348: 360
    lidar_x3_2 = min(lidar[round(ratio*( 360-Angle_det/2 )): round(ratio*(360))])
    # Zone 3 result
    lidar_x3 = min(lidar_x3_1, lidar_x3_2)
    x3 = LidarGoGoPowerRanger(lidar_x3)


    # Zone 4: 324:348+1
    lidar_x4 = min(lidar[round(ratio*( 360-Angle_det/2-Angle_det )): round(ratio*(360-Angle_det/2 +1))])
    x4 = LidarGoGoPowerRanger(lidar_x4)    

    # Zone 5: 258: 282+1
    lidar_x5 = min(lidar[round(ratio*(270- Angle_det/2)): round(ratio*(270+ Angle_det/2+1))])
    x5 = LidarGoGoPowerRanger(lidar_x5)

    # Zone 6: 168: 192+1
    lidar_x6 = min(lidar[round(ratio*(180- Angle_det/2)): round(ratio*(180+ Angle_det/2+1))])
    x6 = LidarGoGoPowerRanger(lidar_x6)

    dist = np.linalg.norm(np.array(target_pos) - np.array(robot_pose))
    robot_pose = np.array([robot_pose[0], robot_pose[1]])
    robot_prev_pose = np.array([robot_prev_pose[0], robot_prev_pose[1]])
    target_pos = np.array([target_pos[0], target_pos[1]])
    #  vector from robot to target
    d_vec = target_pos - robot_pose
    #  vexor from robot to robot_prev
    v_vec = robot_pose - robot_prev_pose

    d_vec3d = np.array([d_vec[0], d_vec[1], 0])
    v_vec3d = np.array([v_vec[0], v_vec[1], 0])

    if np.dot(d_vec, v_vec) < 0:
        x7 = 0 # going back
    else:
        if np.arccos(np.dot(d_vec, v_vec) / (np.linalg.norm(d_vec) * np.linalg.norm(v_vec))) < np.arcsin(goal_radius)/dist:
            x7 = 1 # going to target
        elif np.cross(d_vec3d, v_vec3d)[2] < 0:
            x7 = 2  # too much right
        else:
            x7 = 3 # too much left
       


    ss = np.where(np.all(state_space == np.array([x1, x2, x3, x4, x5, x6, x7]), axis = 1))
    state_ind = int(ss[0])

    return ( state_ind, x1, x2, x3 , x4 , x5, x6, x7)

# Check - crash
def checkCrash(lidar):
    # lidar_horizon = np.concatenate((lidar[(ANGLE_MIN + sum(HORIZON_WIDTH)):(ANGLE_MIN):-1],lidar[(ANGLE_MAX):(ANGLE_MAX - sum(HORIZON_WIDTH)):-1]))
    # W = np.linspace(1.56, 1, len(lidar_horizon) // 2)
    # W = np.append(W, np.linspace(1, 1.56, len(lidar_horizon) // 2))
    # if np.min( W * lidar_horizon ) < COLLISION_DISTANCE:

    length_lidar = len(lidar) 
    ratio = length_lidar / 360 
    angle = 60

    lidar_front_left = min(lidar[round(ratio*(0)): round(ratio*(angle+1))])
    lidar_front_right = min(lidar[round(ratio*(360-angle)): round(ratio*(360))])
    lidar_front = min(lidar_front_left, lidar_front_right)

    lidar_back_left = min(lidar[round(ratio*(180-angle)): round(ratio*(180+1))])
    lidar_back_right = min(lidar[round(ratio*(180)), round(ratio*(180+angle+1))])
    lidar_back = min(lidar_back_left, lidar_back_right)


    if lidar_front <= 0.12:
        return True, lidar_back
    else:
        return False, lidar_back

# Check - object nearby
def checkObjectNearby(x3):
    # lidar_horizon = np.concatenate((lidar[(ANGLE_MIN + sum(HORIZON_WIDTH)):(ANGLE_MIN):-1],lidar[(ANGLE_MAX):(ANGLE_MAX - sum(HORIZON_WIDTH)):-1]))
    # W = np.linspace(1.56, 1, len(lidar_horizon) // 2)
    # W = np.append(W, np.linspace(1, 1.56, len(lidar_horizon) // 2))
    # if np.min( W * lidar_horizon ) < NEARBY_DISTANCE:
    if x3 != 2:
        return True
    else:
        return False

# Check - goal near
def checkGoalNear(x, y, x_goal, y_goal):

    ro = sqrt( pow( ( x_goal - x ) , 2 ) + pow( ( y_goal - y ) , 2) )
    if ro < 0.5: 
        return True
    else:
        return False
