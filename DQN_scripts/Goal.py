import os
import random
import sys

from gazebo_msgs.srv import DeleteEntity
from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Pose
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_srvs.srv import Empty

class Goal(Node):

    """
    to reset goal when succeed or crash, create client in other nodes

    self.task_succeed_client = self.create_client(Empty, 'task_succeed')
    self.task_fail_client = self.create_client(Empty, 'task_fail')

    req = Empty.Request()
    while not self.task_fail_client.wait_for_service(timeout_sec=1.0):
        self.get_logger().info('service not available, waiting again...')
    self.task_fail_client.call_async(req)

    or 

    req = Empty.Request()
    while not self.task_succeed_client.wait_for_service(timeout_sec=1.0):
        self.get_logger().info('service not available, waiting again...')
    self.task_succeed_client.call_async(req)
    """
    
    def __init__(self):
        super().__init__('dqn_gazebo')

        # Entity 'goal'
        self.entity_path = '/opt/ros/humble/share/turtlebot3_gazebo/models/turtlebot3_dqn_world/goal_box/model.sdf'
        self.entity = open(self.entity_path, 'r').read()
        self.entity_name = 'goal'

        # manually change these to set goal (create publisher to be automatic)
        self.goal_pose_x = 3.0
        self.goal_pose_y = -2.0

        self.init_state = False

        qos = QoSProfile(depth=10)

        # Initialise publishers to publish goal position
        self.goal_pose_pub = self.create_publisher(Pose, 'goal_pose', qos)

        # Initialise client to interact with gazebo
        self.delete_entity_client = self.create_client(DeleteEntity, 'delete_entity')
        self.spawn_entity_client = self.create_client(SpawnEntity, 'spawn_entity')
        self.reset_simulation_client = self.create_client(Empty, 'reset_simulation')

        # Initialise servers to listen for robot's crash or success
        self.task_succeed_server = self.create_service(Empty, 'task_succeed', self.task_succeed_callback)
        self.task_fail_server = self.create_service(Empty, 'task_fail', self.task_fail_callback)

        # Timer
        self.publish_timer = self.create_timer(0.010, self.publish_callback)
    


    """*******************************************************************************
    ** Callback functions and relevant functions
    *******************************************************************************"""
    def publish_callback(self):
        # First time
        if self.init_state is False:
            self.delete_entity()
            self.reset_simulation()
            self.init_state = True
            print("init!!!")
            print("Goal pose: ", self.goal_pose_x, self.goal_pose_y)

        # Publish goal pose
        goal_pose = Pose()
        goal_pose.position.x = self.goal_pose_x
        goal_pose.position.y = self.goal_pose_y
        self.goal_pose_pub.publish(goal_pose)
        self.spawn_entity()

    def task_succeed_callback(self, request, response):
        self.delete_entity()
        # uncomment to random goal
        # self.generate_goal_pose()
        print("GOAL! generate a new goal :)")
        return response

    def task_fail_callback(self, request, response):
        self.delete_entity()
        self.reset_simulation()
        # uncomment to random goal
        # self.generate_goal_pose()
        print("FAIL! reset the gazebo environment :(")
        return response

    def generate_goal_pose(self):
        self.goal_pose_x = random.randrange(-15, 16) / 10.0
        self.goal_pose_y = random.randrange(-15, 16) / 10.0
        print("Goal pose: ", self.goal_pose_x, self.goal_pose_y)

    def reset_simulation(self):
        req = Empty.Request()
        while not self.reset_simulation_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        self.reset_simulation_client.call_async(req)

    def delete_entity(self):
        req = DeleteEntity.Request()
        req.name = self.entity_name
        while not self.delete_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        self.delete_entity_client.call_async(req)

    def spawn_entity(self):
        goal_pose = Pose()
        goal_pose.position.x = self.goal_pose_x
        goal_pose.position.y = self.goal_pose_y
        req = SpawnEntity.Request()
        req.name = self.entity_name
        req.xml = self.entity
        req.initial_pose = goal_pose
        while not self.spawn_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        self.spawn_entity_client.call_async(req)


def main(args=sys.argv):
    rclpy.init(args=None)
    dqn_gazebo = Goal()
    rclpy.spin(dqn_gazebo)
    dqn_gazebo.destroy()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
