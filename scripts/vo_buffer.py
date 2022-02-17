#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 13:58:36 2022

@author: navsea-jetson
"""
import rospy
from nav_msgs.msg import Odometry

def callback(vo):

    vo.header.stamp=rospy.Time.now()
    pub.publish(vo)


def listener():
    rospy.init_node('buffer', anonymous=True)
    global pub
    pub = rospy.Publisher('vo_buff', Odometry, queue_size=10)
    rospy.Subscriber("rtabmap/odom", Odometry, callback)
    rospy.spin()
    
if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass