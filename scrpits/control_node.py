from time import time
from time import sleep
from datetime import datetime
import matplotlib.pyplot as plt
import os
from rclpy.node import Node
import rclpy
from std_msgs.msg import String
import math
from geometry_msgs.msg import Twist

import argparse

# Add args
parser = argparse.ArgumentParser(description='control node')
# Add goal position
parser.add_argument('--gpos', default = (0.0, 0.0), nargs=2, type=float, help='Enter x.x y.y pos for goal position')

args = parser.parse_args()

import sys
cwd = os.getcwd()
cwd = os.path.dirname(cwd)
print('PLEASE RUN ON QTABLE/SRCRIPTS FOLDER')
print('PATH:', cwd)

# cwd= 'Qtable/QtableV1'

DATA_PATH = os.path.join(cwd, 'Data')
MODULES_PATH = os.path.join(cwd, 'scrpits')
sys.path.insert(0, MODULES_PATH)

from Qlearning import *
from Lidar import *
from Control import *
from gazebo_msgs.msg._model_state import ModelState

from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerGuardCondition
from rclpy.utilities import timeout_sec_to_nsec

# Real robot
REAL_ROBOT = True

# Action parameter
MIN_TIME_BETWEEN_ACTIONS = 0.0

# wangwang
WANGWANG = .3

# Initial and goal positions
INIT_POSITIONS_X = [ 0] 
INIT_POSITIONS_Y = [ 0]
INIT_POSITIONS_THETA = [0]
GOAL_POSITIONS_X = [ 3]
GOAL_POSITIONS_Y = [ -2]
GOAL_POSITIONS_THETA = [ 90]
GOAL_RADIUS = .1

PATH_IND = 4

# Initial & Goal position
if REAL_ROBOT:
    X_INIT = 0.0
    Y_INIT = 0.0
    THETA_INIT = 0.0
    X_GOAL = args.gpos[0]
    Y_GOAL = args.gpos[1]
    THETA_GOAL = 90
else:
    RANDOM_INIT_POS = False

    X_INIT = INIT_POSITIONS_X[PATH_IND]
    Y_INIT = INIT_POSITIONS_Y[PATH_IND]
    THETA_INIT = INIT_POSITIONS_THETA[PATH_IND]

    X_GOAL = GOAL_POSITIONS_X[PATH_IND]
    Y_GOAL = GOAL_POSITIONS_Y[PATH_IND]
    THETA_GOAL = GOAL_POSITIONS_THETA[PATH_IND]

# Log file directory - Q table source
Q_TABLE_SOURCE = DATA_PATH + '/Log_learning'

class Ward():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = 0.5
    def isInside(self, x, y):
        return math.sqrt((x-self.x)**2 + (y-self.y)**2) < self.radius

class ControlNode(Node):
    def __init__(self):
        print('init...')
        super().__init__('control_node')
        self.setPosPub = self.create_publisher(ModelState, 'gazebo/set_model_state', 10)
        self.velPub = self.create_publisher(Twist, 'cmd_vel', 10)
        # self.actions = createActions(n_actions_enable)
        self.state_space = createStateSpace()
        self.Q_table = readQTable(Q_TABLE_SOURCE+'/Qtable.csv')
        # print(self.Q_table.shape)
        # n_actions_enable = Q_table.shape[1]
        self.actions = createActions(self.Q_table.shape[1])
        self.timer_period = .5 # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.t_step = self.get_clock().now()
        self.robot_in_pos = False
        self.prev_position = (X_INIT, Y_INIT)
        self.count = 0
        self.Angle_det = 24
        self.queuePos = []
        self.tempAction = 0
        self.tempCond = 0
        self.wards = []
        self.inTask = False

    def lidt(string):
        lidar = [0 if x == 20 else x for x in lidar]
        vertex_points = {}
        for i in range(1, len(lidar)-1):
            if lidar[i] != 0:
                if lidar[i-1] < lidar[i] and lidar[i+1] < lidar[i]:
                    vertex_points[i] = lidar[i]
                elif lidar[i-1] > lidar[i] and lidar[i+1] > lidar[i]:
                    vertex_points[i] = lidar[i]
        num_points = len(vertex_points)
        dic = {i*(360/len(lidar)): y for i, y in vertex_points.items()}
        start_key = list(dic.keys())[0]
        end_key = list(dic.keys())[-1]
        highest_point = None
        for key in dic.keys():
            if start_key <= key <= end_key:
                if highest_point is None or dic[key] > dic[highest_point]:
                    highest_point = key
        return highest_point

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
        print('running...')
        _, msgScan = self.wait_for_message('/scan', LaserScan)
        _, odomMsg = self.wait_for_message('/odom', Odometry)
        step_time = (self.get_clock().now() - self.t_step).nanoseconds / 1e9

        if step_time > MIN_TIME_BETWEEN_ACTIONS:
            self.t_step = self.get_clock().now()
            status = ""
            if not self.robot_in_pos:
                robotStop(self.velPub)
                if REAL_ROBOT:
                    ( x_init , y_init , theta_init ) = (0, 0, 0)
                    _, odomMsg = self.wait_for_message('/odom', Odometry)
                    ( x , y ) = getPosition(odomMsg)
                    theta = degrees(getRotation(odomMsg))
                    self.robot_in_pos = True
                    print('\r\nInitial position:')
                    print('x = %.2f [m]' % x)
                    print('y = %.2f [m]' % y)
                    print('theta = %.2f [degrees]' % theta)
                    print('')
                    print('\r\nGoal position:')
                    print('x = %.2f [m]' % X_GOAL)
                    print('y = %.2f [m]' % Y_GOAL)
                    input('Press Enter to start nong...')
                else:
                    if RANDOM_INIT_POS:
                        ( x_init , y_init , theta_init ) = robotSetRandomPos(self.setPosPub)
                    else:
                        ( x_init , y_init , theta_init ) = robotSetPos(self.setPosPub, X_INIT, Y_INIT, THETA_INIT)
                    # check init pos
                    _, odomMsg = self.wait_for_message('/odom', Odometry)
                    ( x , y ) = getPosition(odomMsg)
                    theta = degrees(getRotation(odomMsg))
                    print(theta, theta_init)
                    if abs(x-x_init) < 0.05 and abs(y-y_init) < 0.05 and abs(theta-theta_init) < 2:
                        self.robot_in_pos = True
                        print('\r\nInitial position:')
                        print('x = %.2f [m]' % x)
                        print('y = %.2f [m]' % y)
                        print('theta = %.2f [degrees]' % theta)
                        print('')
                        sleep(1)
                    else:
                        self.robot_in_pos = False
            else:
                self.count = self.count + 1
                text = '\r\nStep %d , Step time %.2f s' % (self.count, step_time)
                
                # Get robot position and orientation
                ( x , y ) = getPosition(odomMsg)
                theta = getRotation(odomMsg)

                # Get lidar scan
                ( lidar, angles ) = lidarScan(msgScan)
                # ( state_ind, x1, x2, x3 , x4 , x5, x6, x7 ) = scanDiscretization(self.state_space, lidar)
            
                length_lidar = len(lidar) 
                ratio = length_lidar / 360 
            
                ( state_ind, x1, x2, x3 , x4 , x5, x6, x7 ) = scanDiscretization(self.state_space, lidar, (X_GOAL, Y_GOAL), (x, y), self.prev_position, GOAL_RADIUS)
                
                # get lidar value from x1 and x5 Zone
                lidar_x1 = min(lidar[round(ratio*(90- self.Angle_det/2)): round(ratio*(90+ self.Angle_det/2+1))])
                lidar_x5 = min(lidar[round(ratio*(270- self.Angle_det/2)): round(ratio*(270+ self.Angle_det/2+1))])
                
                # Zone 2: 12: 36+1
                lidar_x2 = min(lidar[round(ratio*(self.Angle_det/2 )): round(ratio*(self.Angle_det/2+ self.Angle_det+1))])

                # Zone 3-1: 0: 12+1
                lidar_x3_1 = min(lidar[round(ratio*(0)): round(ratio*(self.Angle_det/2+1))])
                # Zone 3-2: 348: 360
                lidar_x3_2 = min(lidar[round(ratio*( 360-self.Angle_det/2 )): round(ratio*(360))])
                # Zone 3 result
                lidar_x3 = min(lidar_x3_1, lidar_x3_2)

                # Zone 4: 324:348+1
                lidar_x4 = min(lidar[round(ratio*( 360-self.Angle_det/2-self.Angle_det )): round(ratio*(360-self.Angle_det/2 +1))])


                            
                # Check for objects nearby
                crash, lidar_back = checkCrash(lidar)
                object_nearby = checkObjectNearby(x3)
                goal_near = checkGoalNear(x, y, X_GOAL, Y_GOAL)
                enable_feedback_control = True
                posIsSame = False
                self.queuePos.append((round(x, 3),round(y, 3)))
                print(self.queuePos)
                if len(self.queuePos) == 5:
                    posIsSame = all(cond == self.queuePos[0] for cond in self.queuePos)
                    self.queuePos.pop(0)

                allLidarHaveObs = all([x1 <= 1, x2 <= 1, x3 < 1, x4 < 1, x5 < 1])

                IsInWard = [self.wards[i].isInside(x, y) for i in range(len(self.wards))]
                # Find the nearest ward
                nearestWard = 0
                for i in range(len(self.wards)):
                    if IsInWard[i]:
                        nearestWard = i
                        break

                # if crash. go backward
                if (crash or self.tempCond == 1) and not (self.inTask ) :
                    # robotStop(self.velPub)
                    if lidar_back >= 0.15 and crash:
                        velMsg = createVelMsg(-0.15,0.0)
                        self.velPub.publish(velMsg)
                        text = text + ' ==> Crash!  backward'
                        self.tempCond = 1
                    elif self.tempCond == 1 or crash :
                        if lidar_x1 >= lidar_x5:
                            velMsg = createVelMsg(0.0, -math.pi/2)
                            self.velPub.publish(velMsg)
                        else:
                            velMsg = createVelMsg(0.0, math.pi/2)
                            self.velPub.publish(velMsg)
                        self.tempCond = 0
                        
                    status = 'Crash! backward'
                # check if robot is in ward then avoid it
                elif any(IsInWard) and not (self.inTask):
                    if lidar_x1 >= lidar_x5:
                        velMsg = createVelMsg(0.0, -math.pi)
                        self.velPub.publish(velMsg)
                    else:
                        velMsg = createVelMsg(0.0, math.pi)
                        self.velPub.publish(velMsg)
                    text = text + ' ==> Avoid ward'
                    status = 'Avoid ward'
                # U turn algorithm
                # if last n position is the same coordinate.
                elif (posIsSame or self.tempCond == 2)  or (allLidarHaveObs) or (self.inTask) or (self.tempCond == 4):
                    if not self.inTask :
                        if self.tempCond == 2:
                            velMsg = createVelMsg(0.0, -math.pi)
                            self.velPub.publish(velMsg)
                            text = text + ' ==> U turn algorithm'
                            self.tempCond = 0
                        else:
                            velMsg = createVelMsg(0.0, -math.pi)    
                            self.velPub.publish(velMsg)
                
                    else :
                        if self.tempCond == 4:
                            robotGoSuperForward(self.velPub)
                            self.tempCond = 0
                            self.inTask = False
                        elif x1 > 1 or x5 > 1:
                            if lidar_x1 >= lidar_x5:
                                velMsg = createVelMsg(0.0, -math.pi)
                                self.velPub.publish(velMsg)
                                self.wards.append(Ward(x, y))
                                print("Ward added at: ", x, y)
                                self.inTask = True
                            else:
                                velMsg = createVelMsg(0.0, math.pi)
                                self.velPub.publish(velMsg)
                                self.wards.append(Ward(x, y))
                                print("Ward added at: ", x, y)
                                self.inTask = True
                        else:
                            robotGoForward(self.velPub)
                            self.inTask = True
                            
                    posIsSame = False
                
                # opportunity algorithm  if robot not in goal zone(2m) do this
                elif (( np.min(lidar) > WANGWANG) and np.linalg.norm(np.array([x,y]) - np.array([X_GOAL, Y_GOAL])) > 2 ) or self.tempCond == 3:
                        # max = np.argmax([lidar_x1, lidar_x2, lidar_x3, lidar_x4, lidar_x5])
                        # Angle = [-math.pi, -math.pi/4, "F", math.pi/4, math.pi]
                        angle_dir=self.lidt(lidar)
                        
                        if self.tempCond == 3:
                            velMsg = createVelMsg(0.15,0.0)
                            self.velPub.publish(velMsg)
                            text = text + ' ==> T๋ee algorithm(forward)'
                            self.tempCond = 0
                        #/180*math.pi 
                        else :
                            if angle_dir >180 : 
                                angle_new = angle_dir-360 
                            else :
                                angle_new = angle_dir
                            text = text + ' ==> T๋ee algorithm'
                            velMsg = createVelMsg(0.0, angle_new/180*math.pi*2)
                            self.velPub.publish(velMsg)


                        # elif max == 2:
                        #     velMsg = createVelMsg(0.15,0.0)
                        #     self.velPub.publish(velMsg)
                        #     text = text + ' ==> Opportunity algorithm(rotate)'
                        #     self.tempCond = 3
                        # else:
                        #     velMsg = createVelMsg(0.0, Angle[max])
                        #     self.velPub.publish(velMsg)
                        #     text = text + ' ==> Opportunity algorithm(rotate)'
                        #     self.tempCond = 3
                        

                        
                        
                    
                # Feedback control algorithm
                elif ( not object_nearby and goal_near ) or ( np.min(lidar) > WANGWANG):
                    status = robotFeedbackControl(self.velPub, x, y, theta, X_GOAL, Y_GOAL, radians(THETA_GOAL))
                    text = text + ' ==> Feedback control algorithm '
                    if goal_near:
                        text = text + '(goal near)'
                
                # go go powerranger algorithm
                elif x1 == x5 and x3 == 2  :
                    robotGoSuperForward(self.velPub)
                    text = text + ' ==> go go powerranger algorithm '
                
                # Q-learning algorithm
                else:
                    ( action, status ) = getBestAction(self.Q_table, state_ind, self.actions)
                    if not status == 'getBestAction => OK':
                        print('\r\n', status, '\r\n')

                    status = robotDoAction(self.velPub, action)
                    if not status == 'robotDoAction => OK':
                        print('\r\n', status, '\r\n')
                    text = text + ' ==> Q-learning algorithm'
                
                # set prev_position
                self.prev_position = ( x , y )
                
                text = text + '\r\nx :       %.2f -> %.2f [m]' % (x, X_GOAL)
                text = text + '\r\ny :       %.2f -> %.2f [m]' % (y, Y_GOAL)
                text = text + '\r\ntheta :   %.2f -> %.2f [degrees]' % (degrees(theta), THETA_GOAL)

                if status == 'Goal position reached!':
                    robotStop(self.velPub)
                    text = text + '\r\n\r\nGoal position reached! End of simulation!'
                    
                    raise SystemExit

                print(text)

def main(args=None):
    rclpy.init(args=args)

    movebase_publisher = ControlNode()
    try:
        rclpy.spin(movebase_publisher)
    except SystemExit:                 # <--- process the exception 
        rclpy.logging.get_logger("Quitting").info('Done')
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    movebase_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()