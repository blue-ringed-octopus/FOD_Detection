#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Mon Jan  18 23:13:00 2021

@author: Benjamin
"""
#requires o3d 0.3.0.0
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
import scipy as sp
import os
import tank_loop_nav as tl
import FOD_Detection as fd
rospack=rospkg.RosPack()
navsea=rospack.get_path('navsea')

#--------------Load variables-------------------------
x = loadmat(navsea+'/scripts/FOD_detection_var.mat')
cov_global=np.array(x['cov_global'])
CAD=np.array(x['CAD_points'])
Sparse_CAD=np.array(x['CAD_points_Sparse'])
Mdist_table=np.array(x['Mdist_table'])
xgrid=np.reshape(np.array(x['x']),(-1))
ygrid=np.reshape(np.array(x['y']),(-1))
zgrid=np.reshape(np.array(x['z']),(-1))

#----------------------Global variables----------------
done=False
map_data=None
icp_thres=5

#-----------------------Functions----------------------
def get_file_name():
	i=0
	while os.path.exists(navsea+"/output/training_set/training_map_"+str(i)+".mat"):
		i=i+1
	return (navsea+"/output/training_set/training_map_"+str(i)+".mat")

def Save_clouds():
	print("Map generation started")
# get map from rtabmap
	fd.get_map_client()
	print("Got Map")
	cloud=fd.msg2pc(map_data)
	CAD_cloud=o3d.PointCloud()
	CAD_cloud.points=o3d.Vector3dVector(CAD)
	print("raw cloud size:"+str(len(np.asarray(cloud.points))))
# Process map
	#----------------------------ICP align---------------- 	
	cloud_icp_ds=fd.random_downsample(cloud,percentage=0.1)
	print("Align map with CAD model...")
	tf_init=np.asarray([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]])
	#tf_init=np.asarray([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
	tf=o3d.registration_icp(cloud_icp_ds, CAD_cloud,icp_thres,tf_init,o3d.TransformationEstimationPointToPoint())
    	cloud.transform(tf.transformation)
	croped_points=fd.crop_cloud(cloud,[-0.01, 6],[-2.75, 0.1],[-0.01, 1.5])
	cloud.points=o3d.Vector3dVector(croped_points)
	tf=o3d.registration_icp(cloud, CAD_cloud,icp_thres,np.asarray([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]),o3d.TransformationEstimationPointToPoint())
    	cloud.transform(tf.transformation)
	croped_points=fd.crop_cloud(cloud,[-0.01, 6],[-2.75, 0.1],[-0.01, 1.5])
	cloud.points=o3d.Vector3dVector(croped_points)
	mdic={"cloud":np.asarray(cloud.points)}
	filename=get_file_name()
	sp.io.savemat(filename, mdic)
	print("saved file to "+ str(filename))

if __name__ == "__main__":
	rospy.init_node('Map_generation',anonymous=True)
	tl.tank_loop_routine(4)
	Save_clouds()

