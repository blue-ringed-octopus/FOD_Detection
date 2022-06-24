#!/usr/bin/env python

import rospy
import actionlib

#move_base_msgs
from move_base_msgs.msg import *
from geometry_msgs.msg import *

def simple_move(x,y,w,z):


    sac = actionlib.SimpleActionClient('move_base', MoveBaseAction )

    #create goal
    goal = MoveBaseGoal()
    goal.target_pose.pose.position.x = x
    goal.target_pose.pose.position.y = y
    goal.target_pose.pose.orientation.w = w
    goal.target_pose.pose.orientation.z = z
    goal.target_pose.header.frame_id = 'map'
    goal.target_pose.header.stamp = rospy.Time.now()

    #start listner
    sac.wait_for_server()
    #send goal
    sac.send_goal(goal)
    print("Sending goal:",x,y,w,z)
    #finish
    sac.wait_for_result()
    #print result
    print (sac.get_result())

def talker(coordinates):
    array = PoseArray()
    array.header.frame_id = 'map'
    array.header.stamp = rospy.Time.now()
    pose = Pose()
    pose.position.x = float(coordinates[0])
    pose.position.y = float(coordinates[1])
    pose.orientation.w = float(coordinates[2])
    pose.orientation.z = float(coordinates[3])
    array.poses.append(pose)

    pub = rospy.Publisher('simpleNavPoses', PoseArray, queue_size=100)
    rate = rospy.Rate(1) # 1hz

        #To not have to deal with threading, Im gonna publish just a couple times in the begging, and then continue with telling the robot to go to the points
    count = 0
    while count<1:
        rate.sleep()	
        print("sending rviz arrow")
        pub.publish(array)
        count +=1
def navigate2point(coordinates):
    try:
        simple_move((coordinates[0]),(coordinates[1]),(coordinates[2]),(coordinates[3]))
        talker(coordinates)
        print("goal reached")
    except rospy.ROSInterruptException:
        print ("Keyboard Interrupt")
		

if __name__ == '__main__':
	rospy.init_node('simple_move')
	navigate2point([1.6912122257962636, -1.0575576183401214, 0.7393174353359425, -0.673357059670636])


