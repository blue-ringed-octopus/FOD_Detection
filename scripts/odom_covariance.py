#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Mon Jan  18 23:13:00 2021

@author: Benjamin
"""

import rospy
from nav_msgs.msg import Odometry

def callback(data):
    data.pose.covariance= [0.001, 0, 0, 0, 0, 0,0, 0.001, 0, 0, 0, 0,0, 0, 99999, 0, 0, 0,0, 0, 0, 99999, 0, 0,
                           0, 0, 0, 0, 99999, 0,0, 0, 0, 0, 0, 0.5]
    pub.publish(data)
    
def listener():
    rospy.init_node('odom_appender', anonymous=True)
    global pub
    pub = rospy.Publisher('odom_cov', Odometry, queue_size=10)
    rospy.Subscriber("odom", Odometry, callback)
    rospy.spin()
    
if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass