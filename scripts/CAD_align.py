#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Mon Jan  18 23:13:00 2021

@author: Benjamin
"""
#requires pip install open3d-python==0.3.0.0
import FOD_Detection as fd
import save_cloud
import numpy as np
import ros_numpy
import rospy 
import rospkg
import threading
import time
import open3d as o3d
import copy
import colorsys as cs
import random
from scipy.io import loadmat
from rtabmap_ros.srv import GetMap
from rtabmap_ros.srv import PublishMap
from std_srvs.srv import Empty
from sensor_msgs.msg import PointCloud2

rospack=rospkg.RosPack()
navsea=rospack.get_path('navsea')

#--------------Load variables-------------------------
x = loadmat(navsea+'/scripts/FOD_detection_var.mat')
cov_global=np.array(x['cov_global'])
CAD=np.array(x['CAD_points'])
Sparse_CAD=np.array(x['CAD_points_Sparse'])

#----------------------Global variables----------------
done=False
map_data=None
icp_thres=5

#-----------------------Functions----------------------
def get_map_thread():
	global map_data
	global done
	
	map_data=rospy.wait_for_message('rtabmap/cloud_map',PointCloud2)
	done = True


def get_map_client():
	thread = threading.Thread(target=get_map_thread)
	print("Getting map")
	try:
		thread.start()
		time.sleep(0.5)
		publish_map=rospy.ServiceProxy('rtabmap/publish_map',PublishMap)
		publish_map(1,1,0)
		print("requested map")
		while not done:
			time.sleep(0.5)
		#return data
	except rospy.ServiceException as e:
		print("Service all failed: %s"%e)

def Cloud2CADtf(tf_init):
	print("FOD detection module started")
# get map from rtabmap
	get_map_client()
	print("Got Map")
	cloud=save_cloud.msg2pc(map_data)
	CAD_cloud=o3d.PointCloud()
	CAD_cloud.points=o3d.Vector3dVector(CAD)
	print("cloud size:"+str(len(np.asarray(cloud.points))))
# Process map
	#----------------------------ICP align---------------- 	
	cloud_icp_ds=save_cloud.random_downsample(cloud,percentage=0.1)
	print("Align map with CAD model...")
	tf1=o3d.registration_icp(cloud_icp_ds, CAD_cloud,icp_thres,tf_init,o3d.TransformationEstimationPointToPoint())
    	cloud.transform(tf1.transformation)
	croped_points=fd.crop_cloud(cloud,[-0.01, 6],[-2.75, 0.1],[-0.01, 1.5])
	cloud.points=o3d.Vector3dVector(croped_points)
	tf2=o3d.registration_icp(cloud, CAD_cloud,icp_thres,np.asarray([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]),o3d.TransformationEstimationPointToPoint())
	tf=np.matmul(tf2.transformation,tf1.transformation)
	return tf

def CAD2Cloudtf(tf_init=np.asarray([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])):
	tf=Cloud2CADtf(tf_init)
	tf_inv=np.linalg.inv(tf)
	return tf, tf_inv
if __name__ == "__main__":
	rospy.init_node('FOD_Detection',anonymous=True)
	tf=CAD2Cloudtf()
	print(tf)

