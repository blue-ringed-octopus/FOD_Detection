#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Mon Jan  18 23:13:00 2021

@author: Benjamin
"""
#requires pip install open3d-python==0.3.0.0

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
import scipy as sp
import os
from scipy.io import loadmat
from rtabmap_ros.srv import GetMap
from rtabmap_ros.srv import PublishMap
from std_srvs.srv import Empty
from sensor_msgs.msg import PointCloud2
from scipy.interpolate import RegularGridInterpolator 
from scipy.cluster import hierarchy

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
def loop_input():
	inp=""
	while inp.lower()!='y' and inp.lower()!='n':
		inp=raw_input("Save result?(y/n): \n")
	save=inp.lower()=='y'
	return save

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.draw_geometries([source_temp, target_temp])

def get_map_thread():
	global map_data
	global done
	
	#map_data=rospy.wait_for_message('rtabmap/cloud_map',PointCloud2)
	map_data=rospy.wait_for_message('rtabmap/map_assembler/cloud_map',PointCloud2)

	done = True


def get_map_client():
	thread = threading.Thread(target=get_map_thread)
	print("Getting map")
	try:
		thread.start()
		time.sleep(0.5)
		#publish_map=rospy.ServiceProxy('rtabmap/publish_map',PublishMap)
		#publish_map=rospy.ServiceProxy('rtabmap/map_assembler/publish_map',PublishMap)
		#publish_map(1,1,0)
		print("requested map")
		while not done:
			time.sleep(0.5)
		#return data
	except rospy.ServiceException as e:
		print("Service all failed: %s"%e)

def msg2pc(data):
	pc=ros_numpy.numpify(data)
	points=np.zeros((pc.shape[0],3))
	points[:,0]=pc['x']
	points[:,1]=pc['y']
	points[:,2]=pc['z']
	p=o3d.PointCloud()
	p.points=o3d.Vector3dVector(points)
	return p
def drawcloud(clouds, size):
	vis=o3d.VisualizerWithEditing()
	vis.create_window()
	ro=o3d.RenderOption()
	ro=vis.get_render_option()
	ro.point_size=size
	ro.show_coordinate_frame=True
	for cloud in clouds:
		vis.add_geometry(cloud)
	vis.run()
	vis.destroy_window()
def crop_cloud(cloud,xlim,ylim,zlim):
	croped_points=[]
	pointlist=np.asarray(cloud.points)
	for point in pointlist:
		if (point[0]>xlim[0]) & (point[0]<xlim[1]) & (point[1]>ylim[0]) & (point[1]<ylim[1]) & (point[2]>zlim[0]) & (point[2]<zlim[1]):
			croped_points.append(point)		
	return croped_points

def M_dist_interp(cloudPoints):
	interpreter=RegularGridInterpolator((xgrid,ygrid,zgrid),values=Mdist_table, method="linear", bounds_error=False)	
	mdist=interpreter(cloudPoints)
	return mdist

def cloud_from_points(points):
	tempcloud=o3d.PointCloud()
	tempcloud.points=o3d.Vector3dVector(np.asarray(points))
	return tempcloud

def Isolate_fod(cloud,mdist, cutoff):
	fod_points=cloud[mdist>=cutoff]
	tank_points=cloud[mdist<cutoff]
	return (fod_points, tank_points)
def random_downsample(cloud, percentage):
	og_size=len(np.asarray(cloud.points))
	ds_size=int(np.floor(len(np.asarray(cloud.points))*percentage))
	print("Downsampling from "+ str(og_size)+" to " +str(ds_size))
	ds_points=random.sample(np.asarray(cloud.points),ds_size)
	cloud_ds=o3d.PointCloud()
	cloud_ds.points=o3d.Vector3dVector(ds_points)
	return cloud_ds

def get_file_name(path, file_name):
	i=0
	while os.path.exists(path+file_name+"_"+str(i)+".mat"):
		i=i+1
	return (path+file_name+"_"+str(i)+".mat")

def save_mat(path, file_name, cloud):
	mdic={"cloud":np.asarray(cloud.points)}
	filename_num=get_file_name(path, file_name)
	sp.io.savemat(filename_num, mdic)
	print("saved to: "+filename_num)
def save_cloud():
	print("FOD detection module started")
# get map from rtabmap
	raw_cloud=None
	while raw_cloud==None or len(raw_cloud.points)==0:
		get_map_client()
		raw_cloud=msg2pc(map_data)
	print("Got Map")

  	#drawcloud([cloud], size=0.1)
	CAD_cloud=o3d.PointCloud()

	CAD_cloud.points=o3d.Vector3dVector(CAD)
	print("raw cloud size:"+str(len(np.asarray(raw_cloud.points))))

# Process map
	#----------------------------ICP align---------------- 	
	cloud_icp_ds=o3d.voxel_down_sample(raw_cloud,0.01)
	cloud_icp_ds=random_downsample(cloud_icp_ds,percentage=0.2)
	print("Align map with CAD model...")
	#tf_init=np.asarray([[-1,0,0,0],[0,-1,0,0],[0,0,1,],[0,0,0,1]])
	tf_init=np.asarray([[1,0,0,0.5],[0,1,0,-1],[0,0,1,0],[0,0,0,1]])
	#draw_registration_result(cloud_icp_ds,CAD_cloud,tf_init)
	
	tf1=o3d.registration_icp(cloud_icp_ds, CAD_cloud,icp_thres,tf_init,o3d.TransformationEstimationPointToPoint())
	
	#draw_registration_result(cloud_icp_ds,CAD_cloud,tf1.transformation)

    	#raw_cloud.transform(tf.transformation)
	cloud_icp_ds.transform(tf1.transformation)

	croped_points=crop_cloud(cloud_icp_ds,[-0.01, 6],[-3, 0.1],[-0.01, 1.5])
	cloud_icp_ds.points=o3d.Vector3dVector(croped_points)

	tf2=o3d.registration_icp(cloud_icp_ds, CAD_cloud,icp_thres,np.asarray([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]),o3d.TransformationEstimationPointToPoint())

	tf=np.matmul(tf2.transformation,tf1.transformation)
    	raw_cloud.transform(tf)
	croped_points=crop_cloud(raw_cloud,[-0.01, 6],[-3, 0.1],[-0.01, 1.5])
	cropped_cloud=o3d.PointCloud()
	cropped_cloud.points=o3d.Vector3dVector(croped_points)
	
	#drawcloud([raw_cloud], size=5)	
	

	cloud=random_downsample(cropped_cloud,percentage=0.05)
	cloud=o3d.voxel_down_sample(cropped_cloud,0.0075)

	#------------------calculate m-distance------------------------
	mdist=M_dist_interp(np.asarray(cloud.points))
	cloud.paint_uniform_color([0.2,0.2,0.2])
	np.asarray(cloud.colors)[np.where(np.isnan(mdist))[0],:]=[0,1,1]
	np.asarray(cloud.colors)[np.where(np.isinf(mdist))[0],:]=[0,1,1]
	mdist[np.where(np.isnan(mdist))]=0
	mdist[np.where(np.isinf(mdist))]=np.max(mdist)
	np.asarray(cloud.colors)[np.where(mdist>2.5)[0],:]=[1,0,0] #2.5
	
  	drawcloud([cloud], size=0.1)
	#Save Cloud for ML
	if (loop_input()):
		save_mat(path=navsea+"/output/Trial_PC/", file_name="Trial_PC", cloud=raw_cloud)

if __name__ == "__main__":
	rospy.init_node('FOD_Detection',anonymous=False)
	save_cloud()

