#! /usr/bin/env python

import numpy as np
from math import *
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan

MAX_LIDAR_DISTANCE = 3.5
COLLISION_DISTANCE = 0.125 # LaserScan.range_min = 0.1199999
NEARBY_DISTANCE = 0.2


ANGLE_MAX = 360 - 1  #360  degree
ANGLE_MIN = 1 - 1   #0 degree
# HORIZON_WIDTH = 75  #original
HORIZON_WIDTH = [9, 16, 65, 9] #9:x1, x2, x7   16:x3, x4   25:x5, x6 

# Convert LasecScan msg to array
def lidarScan(msgScan):
    distances = np.array([])
    angles = np.array([])

    for i in range(len(msgScan.ranges)):
        angle = degrees(i * msgScan.angle_increment)
        if ( msgScan.ranges[i] > MAX_LIDAR_DISTANCE ):
            distance = MAX_LIDAR_DISTANCE
        if ( msgScan.ranges[i] < msgScan.range_min ):
            distance = msgScan.range_min
            # For real robot - protection
            if msgScan.ranges[i] < 0.01:
                distance = NEARBY_DISTANCE
        else:
            distance = msgScan.ranges[i]

        distances = np.append(distances, distance)
        angles = np.append(angles, angle)

    # distances in [m], angles in [degrees]
    return ( distances, angles )
    
def lidar_min(lidar):
    return min([ MAX_LIDAR_DISTANCE if isinstance(laser, str) or laser == float('inf') or laser == 'inf' else laser for laser in lidar])

def lidar_max(lidar):
    idx_max_dist = []
    length_lidar = len(lidar) 
    ratio = length_lidar / 360.0 
    for idx, laser in enumerate(lidar):
        if laser == np.max(lidar):
            idx_max_dist.append(idx)

    idx_max_dist = [ degree*ratio for degree in idx_max_dist] # type float range [0, 360]
    idx_max_dist = [ degree - 360.0  if degree > 180.0 else degree for degree in idx_max_dist ]
    idx_max_dist = [ degree if abs(degree) <= 90.0 else 360.0 for degree in idx_max_dist]
    close_0 = 0.0
    theta = 360.0
    for degree in idx_max_dist:
        if abs(degree) < theta:
            close_0 = degree
    print(f'idx_max_dist: {idx_max_dist}  {close_0} {close_0 * (np.pi/180.0)}')
    return close_0 * (np.pi/180.0)

# Discretization of lidar scan
def defineState(lidar, 
                target_pos, 
                robot_pose, 
                robot_prev_pose, 
                max_dist, 
                goal_radius):
    ### now --> 2304*3*4 stage

    if isinstance(lidar[0], str) or lidar[0] == float('inf') or lidar[0] == 'inf':
        x = MAX_LIDAR_DISTANCE
    else:
        x = lidar[0]
    
    length_lidar = len(lidar) 
    # print(f'length_lidar: {length_lidar}')
    ratio = length_lidar / 360 

    ###############################################################################
    ##HORIZON_WIDTH[0] --> 9 degree :x1, x8
    # lidar_x1 = min(lidar[81: 90])
    x1 = lidar_min(lidar[round(ratio*(ANGLE_MIN + HORIZON_WIDTH[1] + HORIZON_WIDTH[2])): round(ratio*(ANGLE_MIN + HORIZON_WIDTH[1] + HORIZON_WIDTH[2] + HORIZON_WIDTH[3])) ])
  

    x8 = lidar_min(lidar[round(ratio*(ANGLE_MAX - HORIZON_WIDTH[1] - HORIZON_WIDTH[2] - HORIZON_WIDTH[3])):round(ratio*(ANGLE_MAX - HORIZON_WIDTH[1] - HORIZON_WIDTH[2])) ])

    ###############################################################################
    ##HORIZON_WIDTH[2] --> 65 degree(25 to 90) :x2, x7
    # lidar_x2 = min(lidar[25: 90])
    x2 = lidar_min(lidar[round(ratio*(ANGLE_MIN + HORIZON_WIDTH[0] + HORIZON_WIDTH[1])): round(ratio*(ANGLE_MIN + HORIZON_WIDTH[0] + HORIZON_WIDTH[1] + HORIZON_WIDTH[2] ))])

    # lidar_x7 = min(lidar[270: 335])
    x7 = lidar_min(lidar[round(ratio*(ANGLE_MAX  - HORIZON_WIDTH[0] - HORIZON_WIDTH[1] - HORIZON_WIDTH[2] )):round(ratio*(ANGLE_MAX - HORIZON_WIDTH[0] - HORIZON_WIDTH[1])) ])

     ###############################################################################
    ##HORIZON_WIDTH[1] --> 16 degree (9 to 25):x3, x6
    # lidar_x3 = min(lidar[9: 25])
    x3 = lidar_min( lidar[round(ratio*(ANGLE_MIN + HORIZON_WIDTH[0])): round(ratio*(ANGLE_MIN + HORIZON_WIDTH[0] + HORIZON_WIDTH[1]))])

    # lidar_x6 = min(lidar[335: 351])
    x6 = lidar_min(lidar[round(ratio*(ANGLE_MAX  - HORIZON_WIDTH[0] - HORIZON_WIDTH[1])):round(ratio*(ANGLE_MAX - HORIZON_WIDTH[0]))])

    ###############################################################################
    ##HORIZON_WIDTH[0] --> 9 degree :x4, x5    
    x4 = lidar_min(lidar[ANGLE_MIN: round(ratio*(ANGLE_MIN + HORIZON_WIDTH[0]))])

    # from index 351 to 0
    x5 = lidar_min(lidar[round(ratio*(ANGLE_MAX  - HORIZON_WIDTH[0] )):round(ratio*(ANGLE_MAX))] + lidar[0])

    # distance
    target_pos = np.array(target_pos)
    robot_pose = np.array(robot_pose)

    # dist = np.linalg.norm(target_pos - robot_pose)
    dist = np.linalg.norm(target_pos - robot_pose)
    
    robot_pose = np.array([robot_pose[0], robot_pose[1]])
    robot_prev_pose = np.array([robot_prev_pose[0], robot_prev_pose[1]])
    target_pos = np.array([target_pos[0], target_pos[1]])

    
    #  vector from robot to target
    d_vec = target_pos - robot_pose
    #  vexor from robot to robot_prev
    v_vec = robot_pose - robot_prev_pose

    d_vec3d = np.array([d_vec[0], d_vec[1], 0])
    v_vec3d = np.array([v_vec[0], v_vec[1], 0])

    if np.cross(d_vec3d, v_vec3d)[2] < 0:
        angle_state = - np.arccos(np.dot(d_vec, v_vec) / (np.linalg.norm(d_vec) * np.linalg.norm(v_vec)))   # too much right return 0. to -pi
    else:
        angle_state = np.arccos(np.dot(d_vec, v_vec) / (np.linalg.norm(d_vec) * np.linalg.norm(v_vec)))  # too much left return 0. to pi

    angle_max_dist_state = lidar_max(lidar)

    return np.array([x, x1, x2, x3, x4, x5, x6, x7, x8, dist, angle_state, angle_max_dist_state], dtype = np.float32) 

# Check - crash
def checkCrash(lidar):
    if lidar_min(lidar) < COLLISION_DISTANCE:
        return True
    else:
        return False
    
# Check - object nearby
def checkObjectNearby(lidar):
    if lidar_min(lidar) < NEARBY_DISTANCE:
        return True
    else:
        return False
