from time import time
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelState
from math import *
import numpy as np
from tf_transformations import euler_from_quaternion, quaternion_from_euler

# Q-learning speed parameters
CONST_LINEAR_SPEED_FORWARD = 0.42
CONST_ANGULAR_SPEED_FORWARD = 0.0

CONST_LINEAR_SPEED_TURN = 0.05

CONST_ANGULAR_SPEED_TURN = 0.5

ALPHA = 0.8

# Get theta in [radians]
def getRotation(odomMsg):
    orientation_q = odomMsg.pose.pose.orientation
    orientation_list = [ orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
    (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
    return yaw

# Get (x,y) coordinates in [m]
def getPosition(odomMsg):
    x = odomMsg.pose.pose.position.x
    y = odomMsg.pose.pose.position.y
    return ( x , y)

# Get linear speed in [m/s]
def getLinVel(odomMsg):
    return odomMsg.twist.twist.linear.x

# Get angular speed in [rad/s] - z axis
def getAngVel(odomMsg):
    return odomMsg.twist.twist.angular.z

# Create rosmsg Twist()
def createVelMsg(v,w):
    velMsg = Twist()
    velMsg.linear.x = v
    velMsg.linear.y = 0.
    velMsg.linear.z = 0.
    velMsg.angular.x = 0.
    velMsg.angular.y = 0.
    velMsg.angular.z = w
    return velMsg

#######################################################
def robotUp2U(velPub, LINEAR_SPEED, ANGULAR_SPEED):   #
    velMsg = createVelMsg(LINEAR_SPEED, ANGULAR_SPEED)#
    velPub.publish(velMsg)                            #


# Go GO command
def robotGG(velPub, LINEAR_SPEED, ANGULAR_SPEED):
    velMsg = createVelMsg(LINEAR_SPEED, 0.0)#
    velPub.publish(velMsg)

# Full Turn command
def robotFullTurn(velPub, LINEAR_SPEED, ANGULAR_SPEED):
    velMsg = createVelMsg(0.0, ANGULAR_SPEED)#
    velPub.publish(velMsg)
#######################################################


# Go forward command
def robotGoForward(velPub):
    velMsg = createVelMsg(CONST_LINEAR_SPEED_FORWARD, CONST_ANGULAR_SPEED_FORWARD)
    velPub.publish(velMsg)

# Go 2 x forward command
def robotGoSuperForward(velPub):
    velMsg = createVelMsg(2*CONST_LINEAR_SPEED_FORWARD, CONST_ANGULAR_SPEED_FORWARD)
    velPub.publish(velMsg)

# Go backward command
def robotGoBackward(velPub):
    velMsg = createVelMsg(-CONST_LINEAR_SPEED_FORWARD, CONST_ANGULAR_SPEED_FORWARD)
    velPub.publish(velMsg)

# Turn left command
def robotTurnLeft(velPub):
    velMsg = createVelMsg(CONST_LINEAR_SPEED_TURN, CONST_ANGULAR_SPEED_TURN)
    velPub.publish(velMsg)

# Turn right command
def robotTurnRight(velPub):
    velMsg = createVelMsg(CONST_LINEAR_SPEED_TURN,-CONST_ANGULAR_SPEED_TURN)
    velPub.publish(velMsg)

# Stop command
def robotStop(velPub):
    velMsg = createVelMsg(0.0,0.0)
    velPub.publish(velMsg)
# CW command
def robotCW(velPub):
    velMsg = createVelMsg(0.0, -CONST_ANGULAR_SPEED_TURN)
    velPub.publish(velMsg)

# CCW command
def robotCCW(velPub):
    velMsg = createVelMsg(0.0, CONST_ANGULAR_SPEED_TURN)
    velPub.publish(velMsg)    

# Set robot position and orientation
def robotSetPos(setPosPub, x, y, theta = 0.0):
    checkpoint = ModelState()

    checkpoint.model_name = 'turtlebot3_burger'

    checkpoint.pose.position.x = float(x)
    checkpoint.pose.position.y = float(y)
    checkpoint.pose.position.z = 0.0

    [x_q,y_q,z_q,w_q] = quaternion_from_euler(0.0,0.0,radians(theta))

    checkpoint.pose.orientation.x = x_q
    checkpoint.pose.orientation.y = y_q
    checkpoint.pose.orientation.z = z_q
    checkpoint.pose.orientation.w = w_q

    checkpoint.twist.linear.x = 0.0
    checkpoint.twist.linear.y = 0.0
    checkpoint.twist.linear.z = 0.0

    checkpoint.twist.angular.x = 0.0
    checkpoint.twist.angular.y = 0.0
    checkpoint.twist.angular.z = 0.0

    setPosPub.publish(checkpoint)
    return ( x, y)


# Perform an action
def robotDoAction(velPub, action, LINEAR_SPEED, ANGULAR_SPEED):
    status = 'robotDoAction => OK'
    match action:
        case 0: robotGG(velPub, LINEAR_SPEED, ANGULAR_SPEED)
        case 1: robotFullTurn(velPub, LINEAR_SPEED, ANGULAR_SPEED)
        case _: 
            status = 'robotDoAction => INVALID ACTION'
            robotUp2U(velPub, LINEAR_SPEED, ANGULAR_SPEED)

    return status

