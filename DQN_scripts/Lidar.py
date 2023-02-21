#! /usr/bin/env python

import numpy as np
from math import *
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan

MAX_LIDAR_DISTANCE = .8
COLLISION_DISTANCE = 0.125 # LaserScan.range_min = 0.1199999
NEARBY_DISTANCE = 0.45

ZONE_0_LENGTH = .25
ZONE_1_LENGTH = .5

ANGLE_MAX = 360  #360  degree
ANGLE_MIN = 1 - 1   #0 degree
ANGLE_BACK = 180  #180 degree
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

# Discretization of lidar scan
def defineState(lidar, 
                target_pos, 
                robot_pose, 
                robot_prev_pose, 
                max_dist, 
                goal_radius):
    ### now --> 2304*3*4 stage
    x1 = 0 #[0, 1]
    x2 = 0 #[0, 1]
    x3 = 0 #[0, 1, 2] -> [0, 0.5, 1]
    x4 = 0 #[0, 1, 2, 3] -> [0, 0.33, 0.67, 1]
    x5 = 0 #[0, 1, 2, 3] -> [0, 0.33, 0.67, 1]
    x6 = 0 #[0, 1, 2] -> [0, 0.5, 1]
    x7 = 0 #[0, 1]
    x8 = 0 #[0, 1]
    
    x9 = 0 #[0, 1, 2] -> [0, 0.5, 1]
    x10 = 1 #[0:Go Back, 2: too much left, 3:too much right, 1:G0 target] -> [0, 0.5, 0.5, 1]

    length_lidar = len(lidar) 
    # print(f'length_lidar: {length_lidar}')
    ratio = length_lidar / 360 
    
    ###############################################################################
    ##HORIZON_WIDTH[0] --> 9 degree :x1, x8
    # lidar_x1 = min(lidar[81: 90])
    lidar_x1 = min(lidar[round(ratio*(ANGLE_MIN + HORIZON_WIDTH[1] + HORIZON_WIDTH[2])): round(ratio*(ANGLE_MIN + HORIZON_WIDTH[1] + HORIZON_WIDTH[2] + HORIZON_WIDTH[3])) ])
    if ZONE_0_LENGTH < lidar_x1 < ZONE_1_LENGTH:
        x1 = 1
    else: 
        x1 = 0

    # lidar_x8 = min(lidar[270: 279])
    lidar_x8 = min(lidar[round(ratio*(ANGLE_MAX - HORIZON_WIDTH[1] - HORIZON_WIDTH[2] - HORIZON_WIDTH[3])):round(ratio*(ANGLE_MAX - HORIZON_WIDTH[1] - HORIZON_WIDTH[2])) ])
    if ZONE_0_LENGTH < lidar_x8 < ZONE_1_LENGTH:
        x8 = 1
    else: 
        x8 = 0

    ###############################################################################
    ##HORIZON_WIDTH[2] --> 65 degree(25 to 90) :x2, x7
    # lidar_x2 = min(lidar[25: 90])
    lidar_x2 = min(lidar[round(ratio*(ANGLE_MIN + HORIZON_WIDTH[0] + HORIZON_WIDTH[1])): round(ratio*(ANGLE_MIN + HORIZON_WIDTH[0] + HORIZON_WIDTH[1] + HORIZON_WIDTH[2] ))])
    if ZONE_0_LENGTH <= lidar_x2:
        x2 = 1
    else:
        x2 = 0

    # lidar_x7 = min(lidar[270: 335])
    lidar_x7 = min(lidar[round(ratio*(ANGLE_MAX  - HORIZON_WIDTH[0] - HORIZON_WIDTH[1] - HORIZON_WIDTH[2] )):round(ratio*(ANGLE_MAX - HORIZON_WIDTH[0] - HORIZON_WIDTH[1])) ])
    if ZONE_0_LENGTH <= lidar_x7:
        x7 = 1
    else:
        x7 = 0

    ###############################################################################
    ##HORIZON_WIDTH[1] --> 16 degree (9 to 25):x3, x6
    # lidar_x3 = min(lidar[9: 25])
    lidar_x3 = min( lidar[round(ratio*(ANGLE_MIN + HORIZON_WIDTH[0])): round(ratio*(ANGLE_MIN + HORIZON_WIDTH[0] + HORIZON_WIDTH[1]))])
    if ZONE_1_LENGTH < lidar_x3:
        x3 = 0
    elif ZONE_0_LENGTH < lidar_x3 < ZONE_1_LENGTH:
        x3 = 0.5
    elif lidar_x3 < ZONE_0_LENGTH:
        x3 = 1

    # lidar_x6 = min(lidar[335: 351])
    lidar_x6 = min(lidar[round(ratio*(ANGLE_MAX  - HORIZON_WIDTH[0] - HORIZON_WIDTH[1])):round(ratio*(ANGLE_MAX - HORIZON_WIDTH[0]))])
    if ZONE_1_LENGTH < lidar_x6:
        x6 = 0
    elif ZONE_0_LENGTH < lidar_x6 < ZONE_1_LENGTH:
        x6 = 0.5
    elif lidar_x6 < ZONE_0_LENGTH:
        x6 = 1

    ###############################################################################
    ##HORIZON_WIDTH[0] --> 9 degree :x4, x5    
    # lidar_x4 = min(lidar[0: 10])
    lidar_x4 = min(lidar[ANGLE_MIN: round(ratio*(ANGLE_MIN + HORIZON_WIDTH[0]))])
    if MAX_LIDAR_DISTANCE < lidar_x4:
        x4 = 0
    elif ZONE_1_LENGTH < lidar_x4:
        x4 = 0.33
    elif ZONE_0_LENGTH < lidar_x4 < ZONE_1_LENGTH:
        x4 = 0.67
    elif lidar_x4 < ZONE_0_LENGTH:
        x4 = 1

    # from index 351 to 0
    # lidar_x5 = min(lidar[350: 360] + lidar[0])
    lidar_x5 = min(lidar[round(ratio*(ANGLE_MAX  - HORIZON_WIDTH[0] )):round(ratio*(ANGLE_MAX))] + lidar[0])

    if MAX_LIDAR_DISTANCE < lidar_x5:
        x5 = 0
    elif ZONE_1_LENGTH < lidar_x5:
        x5 = 0.33
    elif ZONE_0_LENGTH < lidar_x5 < ZONE_1_LENGTH:
        x5 = 0.67
    elif lidar_x5 < ZONE_0_LENGTH:
        x5 = 1

    ###############################################################################
    # distance
    target_pos = np.array(target_pos)
    robot_pose = np.array(robot_pose)

    # dist = np.linalg.norm(target_pos - robot_pose)
    x9 = np.linalg.norm(target_pos - robot_pose)

    # if dist > .5 * max_dist:
    #     x9 = 0
    # elif dist > 2*goal_radius:
    #     x9 = 0.5
    # else:
    #     x9 = 1
    
    robot_pose = np.array([robot_pose[0], robot_pose[1]])
    robot_prev_pose = np.array([robot_prev_pose[0], robot_prev_pose[1]])
    target_pos = np.array([target_pos[0], target_pos[1]])

    # #  vector from robot to target
    d_vec = target_pos - robot_pose
    #  vexor from robot to robot_prev
    v_vec = robot_pose - robot_prev_pose

    x10 = np.arccos(np.dot(d_vec, v_vec) / (np.linalg.norm(d_vec) * np.linalg.norm(v_vec)))
    # d_vec3d = np.array([d_vec[0], d_vec[1], 0])
    # v_vec3d = np.array([v_vec[0], v_vec[1], 0])

    # if np.dot(d_vec, v_vec) < 0:
    #     x10 = 0 # going back
    # else:
    #     if np.arccos(np.dot(d_vec, v_vec) / (np.linalg.norm(d_vec) * np.linalg.norm(v_vec))) < np.arcsin(goal_radius)/dist:
    #         x10 = 1 # going to target
    #     elif np.cross(d_vec3d, v_vec3d)[2] < 0:
    #         x10 = 0.5  # too much right
    #     else:
    #         x10 = 0.5 # too much left

    return np.array([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10], dtype = np.float32)

# Check - crash
def checkCrash(lidar):
    if np.min(lidar) < COLLISION_DISTANCE:
        return True
    else:
        return False
