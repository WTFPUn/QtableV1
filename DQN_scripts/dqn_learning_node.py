#! /usr/bin/env python

import rclpy
from time import time, sleep
# import time
from datetime import datetime
import matplotlib.pyplot as plt
from rclpy.node import Node
from std_srvs.srv import Empty
import pandas as pd
from std_srvs.srv._empty import Empty_Request
import sys
DATA_PATH = '/mnt/c/Users/keera/Documents/Github/Basic_robot/QtableV1/Data'
MODULES_PATH = '/mnt/c/Users/keera/Documents/Github/Basic_robot/QtableV1/DQN_scripts'

sys.path.insert(0, MODULES_PATH)
from gazebo_msgs.msg._model_state import ModelState
from geometry_msgs.msg import Twist

from Utils import *
from DQNlearning import DQN
from Qlearning import *
from Lidar import *
from Control import *
from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerGuardCondition
from rclpy.utilities import timeout_sec_to_nsec

from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary

import argparse
import os
#################################################################################
# if gpu is to be used
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Episode parameters
MAX_EPISODES = 10
MAX_STEPS_PER_EPISODE = 500
MIN_TIME_BETWEEN_ACTIONS = 0.0
MAX_EPISODES_BEFORE_SAVE = 5

# Learning parameters
ALPHA = 0.5
GAMMA = 0.9

# Log file directory
CHECKPOINT_DIR = MODULES_PATH + '/Checkpoint/Log_learning'

# Q table source file
DQN_SOURCE_DIR = CHECKPOINT_DIR + '/DQN_beta.pth'

RADIUS_REDUCE_RATE = .5
REWARD_THRESHOLD =  -200
CUMULATIVE_REWARD = 0.0

GOAL_POSITION = (0., 2., .0)
GOAL_RADIUS = .1

# edit when chang order in def roboDoAction in Control.py  *****
ACTIONS_DESCRIPTION = { 0 : 'Forward',
                        1 : 'CW',
                        2 : 'CCW',
                        3 : 'Stop',
                        4 : 'SuperForward'}
MAX_WIDTH = 25
STEP_DONE = 0
########################################################################################################

parser = argparse.ArgumentParser(description='Qtable V1 ~~Branch: welcomeToV2')
# Log file directory
parser.add_argument('--log_file_dir', default = CHECKPOINT_DIR, type=str, help='/Checkpoint/Log_learning')
# Q table source file
parser.add_argument('--DQN_source_dir', default = DQN_SOURCE_DIR, type=str, help='/Checkpoint/Log_learning/DQN_beta.pth')

# Episode parameters
parser.add_argument('--max_episodes', default=MAX_EPISODES, type=int, help="MAX_EPISODES = 10 (default)", dest="max_episodes")
parser.add_argument('--max_step_per_episodes', default=MAX_STEPS_PER_EPISODE, type=int, help="MAX_STEPS_PER_EPISODE = 500 (default)")
parser.add_argument('--max_episodes_before_save', default=MAX_EPISODES_BEFORE_SAVE, type=int, help="MAX_EPISODES_BEFORE_SAVE = 5 (default)")

# need to use action='store true' to store boolean. True when type --resume. False otherwise
parser.add_argument('--resume', action='store_true', help ="continue learning with same DQN_beta.pth")
parser.add_argument('--n_actions_enable', default=5, type=int, help='default--> 0:forward, 1:CW, 2:CCW, 3:stop, 4:superForward')

parser.add_argument('--radiaus_reduce_rate', default=RADIUS_REDUCE_RATE, type=float)
parser.add_argument('--reward_threshold', default=REWARD_THRESHOLD, type=int)
parser.add_argument('--GOAL_POSITION', default=GOAL_POSITION, nargs='+', type=float)
parser.add_argument('--GOAL_RADIUS', default=GOAL_RADIUS, type=float)


args_parse = parser.parse_args()

(GOAL_X, GOAL_Y, GOAL_THETA) = tuple(args_parse.GOAL_POSITION)
WIN_COUNT = 0.0

########################################################################################################

class DQNLearningNode(Node):
    def __init__(self):
        super().__init__('learning_node')
        self.timer_period = .5 # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.reset = self.create_client(Empty, '/reset_simulation')
        self.setPosPub = self.create_publisher(ModelState, 'gazebo/set_model_state', 10)
        self.velPub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.dummy_req = Empty_Request()
        self.reset.call_async(self.dummy_req)
        self.actions = createActions(args_parse.n_actions_enable)
        self.state_space = createStateSpace()
        self.policy_net = DQN(n_observations = len(self.state_space), n_actions = len(self.actions)).to(DEVICE)
        self.target_net = DQN(n_observations = len(self.state_space), n_actions = len(self.actions)).to(DEVICE)

        if args_parse.resume:
                self.policy_net.load_state_dict(torch.load(args_parse.DQN_source_dir))

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=1e-4, amsgrad=True)

        print(f'\n {"start learning_node with":^{MAX_WIDTH*4}}')
        print('-'*100)
        for arg in vars(args_parse):
            print(f'{arg:<{MAX_WIDTH}}: {str(getattr(args_parse, arg)):<{MAX_WIDTH}}')

        print('-'*100)
        print(f'\n state_space shape:  {summary(self.policy_net, input_size = (len(self.state_space),))}')
        print(f'\n n_actions: {args_parse.n_actions_enable} --> {[ACTIONS_DESCRIPTION[i] for i in range(args_parse.n_actions_enable)]}')

        
        print()
        input("Press enter to continue...")
        # initial position
        self.robot_in_pos = False
        self.first_action_taken = False

        # init time
        self.t_0 = self.get_clock().now()
        self.t_start = self.get_clock().now()

        # init timer
        while not (self.t_start > self.t_0):
            self.t_start = self.get_clock().now()

        self.t_ep = self.t_start
        self.t_sim_start = self.t_start
        self.t_step = self.t_start

        self.CUMULATIVE_REWARD = CUMULATIVE_REWARD
        self.WIN_COUNT = WIN_COUNT
        self.terminal_state = False
        self.is_set_pos = False
    
    
    def wait_for_message(
        node,
        topic: str,
        msg_type,
        time_to_wait=-1
    ):
        """
        Wait for the next incoming message.
        :param msg_type: message type
        :param node: node to initialize the subscription on
        :param topic: topic name to wait for message
        :time_to_wait: seconds to wait before returning
        :return (True, msg) if a message was successfully received, (False, ()) if message
            could not be obtained or shutdown was triggered asynchronously on the context.
        """
        context = node.context
        wait_set = _rclpy.WaitSet(1, 1, 0, 0, 0, 0, context.handle)
        wait_set.clear_entities()

        sub = node.create_subscription(msg_type, topic, lambda _: None, 1)
        wait_set.add_subscription(sub.handle)
        sigint_gc = SignalHandlerGuardCondition(context=context)
        wait_set.add_guard_condition(sigint_gc.handle)

        timeout_nsec = timeout_sec_to_nsec(time_to_wait)
        wait_set.wait(timeout_nsec)

        subs_ready = wait_set.get_ready_entities('subscription')
        guards_ready = wait_set.get_ready_entities('guard_condition')

        if guards_ready:
            if sigint_gc.handle.pointer in guards_ready:
                return (False, None)

        if subs_ready:
            if sub.handle.pointer in subs_ready:
                msg_info = sub.handle.take_message(sub.msg_type, sub.raw)
                return (True, msg_info[0])

        return (False, None)
    
    def timer_callback(self):
            _, msgScan = self.wait_for_message('/scan', LaserScan)

            # find time taken betwwen 2 callbacks
            step_time = (self.get_clock().now() - self.t_step).nanoseconds / 1e9
            self.t_step = self.get_clock().now()
            if step_time > 2:
                text = '\r\nTOO BIG STEP TIME: %.2f s' % step_time
                print(text)
                
                raise SystemExit
            
            #training....
            if self.episode < args_parse.max_episodes :
                # simulation time
                _, odomMsg = self.wait_for_message('/odom', Odometry) 
                ( current_x , current_y ) = getPosition(odomMsg)
                ( lidar, angles ) = lidarScan(msgScan)
                self.is_set_pos = False
                # First acion
                if not self.first_action_taken:
                        self.prev_position = getPosition(odomMsg)
                        self.first_action_taken = True

                state  = defineState(lidar, 
                                    (GOAL_X, GOAL_Y), 
                                    (current_x, current_y), 
                                    self.prev_position, 
                                    self.MAX_RADIUS, 
                                    GOAL_RADIUS)
                
                state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

                for t in count():
                    action = select_action(state, STEP_DONE, self.policy_net)
                    status_rda = robotDoAction(self.velPub, action)

                    observation = defineState(lidar, 
                                    (GOAL_X, GOAL_Y), 
                                    (current_x, current_y), 
                                    self.prev_position, 
                                    self.MAX_RADIUS, 
                                    GOAL_RADIUS)
                    
                    ( reward, self.terminal_state, win_count) = getReward(  action = self.action, 
                                                                            prev_action = self.prev_action, 
                                                                            lidar = lidar, 
                                                                            prev_lidar = self.prev_lidar, 
                                                                            crash = self.crash,
                                                                            current_position = (current_x, current_y),
                                                                            goal_position = (GOAL_X, GOAL_Y), 
                                                                            max_radius = self.MAX_RADIUS, 
                                                                            radius_reduce_rate = args_parse.radiaus_reduce_rate, 
                                                                            nano_start_time = ep_time ,
                                                                            nano_current_time = self.get_clock().now().nanoseconds, 
                                                                            goal_radius = args_parse.GOAL_RADIUS, 
                                                                            angle_state = state[-1], 
                                                                            win_count = self.WIN_COUNT)
                    reward = torch.tensor([reward], device=DEVICE)
                    done = terminated or truncated

                    if terminated:
                        next_state = None
                    else:
                        next_state = torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    state = next_state
                self.prev_position = (current_x, current_y)
                
                
            
            else:

                raise SystemExit

                ep_time = (self.get_clock().now() - self.t_ep).nanoseconds / 1e9
                # End of en Episode
                print(f'\n episode {self.episode} of {args_parse.max_episodes}')
                
                if self.CUMULATIVE_REWARD < args_parse.reward_threshold or self.terminal_state:
                    robotStop(self.velPub)
                    print(f"\n End of episode. step: {self.ep_steps}")
                    print(f' CUMULATIVE_REWARD: {self.CUMULATIVE_REWARD}')
                    print(f' WIN_COUNT: {self.WIN_COUNT}')
                    # if self.crash:
                    #     # get crash position
                    #     _, odomMsg = self.wait_for_message('/odom', Odometry)
                    #     ( x_crash , y_crash ) = getPosition(odomMsg)
                    #     theta_crash = degrees(getRotation(odomMsg))

                    self.t_ep = self.get_clock().now()
                    self.reward_min = np.min(self.ep_reward_arr)
                    self.reward_max = np.max(self.ep_reward_arr)
                    self.reward_avg = np.mean(self.ep_reward_arr)
                    text = '---------------------------------------\r\n'
                    self.reset.call_async(self.dummy_req)
                    text = text + '\r\nepisode time: %.2f s (avg step: %.2f s) \r\n' % (ep_time, ep_time / (self.ep_steps))
                    text = text + 'episode steps: %d \r\n' % self.ep_steps
                    text = text + 'episode reward: %.2f \r\n' % self.ep_reward
                    text = text + 'episode max | avg | min reward: %.2f | %.2f | %.2f \r\n' % (self.reward_max, self.reward_avg, self.reward_min)
                    if args_parse.exploration_func == 1:
                        text = text + 'T = %f \r\n' % self.T
                    else:
                        text = text + 'EPSILON = %f \r\n' % self.EPSILON
                    print(text)

                    self.ep_steps = 0
                    self.ep_reward = 0
                    # cum reward reset
                    self.CUMULATIVE_REWARD = 0
                    self.crash = 0
                    self.robot_in_pos = False
                    self.first_action_taken = False
                    self.terminal_state = False

                    # save to csv every n episodes
                    if self.episode % args_parse.max_episodes_before_save == 0:
                        print(f"saving data to csv every {args_parse.max_episodes_before_save} episodes")
                        # self.save_info_csv()

                    self.episode = self.episode + 1
                    sleep(1)

                else:
                    self.ep_steps = self.ep_steps + 1

                    # Initial position
                    if not self.is_set_pos:
                        _, odomMsg = self.wait_for_message('/odom', Odometry)
                        robotStop(self.velPub)
                        self.ep_steps = self.ep_steps - 1
                        self.first_action_taken = False
                        # init pos
                        (x_set_init, y_set_init) = getPosition(odomMsg)
                        print('set pos')
                        ( x_init , y_init) = robotSetPos(self.setPosPub, x_set_init, y_set_init)

                        self.prev_position = getPosition(odomMsg)
                        self.MAX_RADIUS = np.linalg.norm([x_init - GOAL_X, y_init - GOAL_Y])
                        # check init pos
                        self.is_set_pos = True
                      
                    # First acion
                    elif not self.first_action_taken:
                        _, odomMsg = self.wait_for_message('/odom', Odometry)               #just added
                        ( current_x , current_y ) = getPosition(odomMsg)                    #just added
                        ( lidar, angles ) = lidarScan(msgScan)                              #just added
                        

                        states  = defineState(lidar, 
                                              (GOAL_X, GOAL_Y), 
                                              (current_x, current_y), 
                                              self.prev_position, 
                                              self.MAX_RADIUS, 
                                              GOAL_RADIUS)
                        
                        self.crash = checkCrash(lidar)

                        status_rda = robotDoAction(self.velPub, self.action)

                        self.prev_lidar = lidar
                        self.prev_action = self.action
                        self.prev_state_ind = state_ind

                        self.first_action_taken = True

                        if not status_rda == 'robotDoAction => OK':
                            print('\r\n', status_rda, '\r\n')

                    # Rest of the algorithm
                    else:
                        _, odomMsg = self.wait_for_message('/odom', Odometry)               #just added
                        ( current_x , current_y ) = getPosition(odomMsg)                    #just added
                        ( lidar, angles ) = lidarScan(msgScan)
                        # print(self.prev_position,( current_x , current_y ))
                        
                        # get position
                        _, odomMsg = self.wait_for_message('/odom', Odometry)
                        yaw = getRotation(odomMsg)


                        ( state_ind, x1, x2, x3 , x4 , x5, x6, x7, x8, x9, x10 ) = scanDiscretization(self.state_space, lidar, (GOAL_X, GOAL_Y), (current_x, current_y),self.prev_position, self.MAX_RADIUS, GOAL_RADIUS)
                        self.crash = checkCrash(lidar)
                        
                        # radius caculated by norm of  and goal position
                    
                        # ( reward, terminal_state ) = getReward(self.action, self.prev_action, lidar, self.prev_lidar, self.crash)
                        # getReward(action, prev_action,lidar, prev_lidar, crash, current_position, goal_position, max_radius, args_parse.radiaus_reduce_rate, nano_start_time, nano_current_time):
                        ( reward, self.terminal_state, win_count) = getReward(self.action, self.prev_action, lidar, self.prev_lidar, self.crash,
                                                                   (current_x, current_y),
                                                                    # self.prev_position,
                                                                     (GOAL_X, GOAL_Y), 
                                                                    self.MAX_RADIUS, args_parse.radiaus_reduce_rate, ep_time ,
                                                                    self.get_clock().now().nanoseconds, 
                                                                    args_parse.GOAL_RADIUS, x10, self.WIN_COUNT)
                        
                        self.CUMULATIVE_REWARD += reward
                        self.WIN_COUNT = win_count
                        print(f' CUMULATIVE_REWARD: {self.CUMULATIVE_REWARD}')
                        print(f' WIN_COUNT: {self.WIN_COUNT}')

                        status_rda = robotDoAction(self.velPub, self.action)

                        self.ep_reward = self.ep_reward + reward
                        self.ep_reward_arr = np.append(self.ep_reward_arr, reward)
                        self.prev_lidar = lidar
                        self.prev_action = self.action
                        self.prev_state_ind = state_ind

                        ( current_x , current_y ) = getPosition(odomMsg)
                        self.prev_position = (current_x, current_y)


def main(ep_count, args=None):
    rclpy.init(args=args)
    # print('args_parse: ', args_parse)
    movebase_publisher = DQNLearningNode()
    try:
        rclpy.spin(movebase_publisher)
    except SystemExit:                 # <--- process the exception 
        rclpy.logging.get_logger("End of learning").info('Done')
    
        
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    movebase_publisher.destroy_node()
    rclpy.shutdown()
    
    ep_count += movebase_publisher.episode
    if ep_count < args_parse.max_episodes:
        main(ep_count= ep_count, args=None)
    print("time skip error at episodes: ", movebase_publisher.episode, " total episodes: ", ep_count)


if __name__ == '__main__':
    main(ep_count = 0, args = None)