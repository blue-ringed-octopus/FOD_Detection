#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Fri Feb  05 23:13:00 2021

@author: Benjamin
"""
import numpy as np
import ros_numpy
import rospy 
import rospkg

# def Get_current_location():
# 	listener = tf.TransformListener()
# 	
# 	trans=[]
# 	while trans==[]:
# 		try:
# 			(trans,rot)= listener.lookupTransform('/map', '/base_link', rospy.Time(0))
# 		except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
# 			continue
# 		return trans

def find_closest_point(loc, objectives):
	dist=np.sum((np.asarray(loc)-(np.asarray(objectives))[0:2])**2,axis=1)
	idx=np.argmin(dist)
	return (idx,objectives[idx])

def greedy_scheduler(objectives, curr_loc):
	#trans=Get_current_location()
	#(idx, closest_point)=find_closest_point(trans[0:2], objectives)
    (idx, closest_point)=find_closest_point(curr_loc[0:2], objectives)
    return (idx, closest_point)

if __name__ == "__main__":
	rospy.init_node('tf_listener')
	dummy_obj=[[-4,0, 1],[98,6, 1],[-3,0.2, 1]]
	(idx, closest_point)=greedy_scheduler(dummy_obj)
	print(idx)
	print(closest_point)

	
