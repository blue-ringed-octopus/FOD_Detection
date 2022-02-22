#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Jan  18 23:13:00 2021

@author: Benjamin
"""
import numpy as np
import ros_numpy
import rospy 
import rospkg
import threading
import time
import open3d as o3d
import copy
import random
import scipy as sp
import os
from scipy.io import loadmat
from sensor_msgs.msg import PointCloud2

rospack=rospkg.RosPack()
navsea=rospack.get_path('navsea')

#--------------Load variables-------------------------
x = loadmat(navsea+'/scripts/FOD_detection_var.mat')
cov_global=np.array(x['cov_global'])
CAD=np.array(x['CAD_points'])


#----------------------Global variables----------------
done=False
map_data=None
icp_thres=5

#-----------------------Functions----------------------
def loop_input(prompt):
	inp=""
	while inp.lower()!='y' and inp.lower()!='n':
		inp=input(prompt+"(y/n): \n")
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
    pc=ros_numpy.point_cloud2.split_rgb_field(pc)
    rgb=np.zeros((pc.shape[0],3))
    rgb[:,0]=pc['r']
    rgb[:,1]=pc['g']
    rgb[:,2]=pc['b']
    print(rgb)
    p=o3d.geometry.PointCloud()
    p.points=o3d.utility.Vector3dVector(points)
    p.colors=o3d.utility.Vector3dVector(np.asarray(rgb/255))
    return p
def drawcloud(clouds, size):
	vis=o3d.visualization.VisualizerWithEditing()
	vis.create_window()
	ro=o3d.visualization.RenderOption()
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


def cloud_from_points(points):
	tempcloud=o3d.geometry.PointCloud()
	tempcloud.points=o3d.utility.Vector3dVector(np.asarray(points))
	return tempcloud

def random_downsample(cloud, percentage):
	og_size=len(np.asarray(cloud.points))
	ds_size=int(np.floor(len(np.asarray(cloud.points))*percentage))
	print("Downsampling from "+ str(og_size)+" to " +str(ds_size))
	ds_points=random.sample(list(cloud.points),ds_size)
	cloud_ds=o3d.geometry.PointCloud()
	cloud_ds.points=o3d.utility.Vector3dVector(ds_points)
	return cloud_ds

def get_file_name(path, file_name):
	i=0
	while os.path.exists(path+file_name+"_"+str(i)+".mat"):
		i=i+1
	return (path+file_name+"_"+str(i)+".mat")

def save_mat(path, file_name, cloud):
	mdic={"cloud":np.asarray(cloud.points), "rgb":np.asarray(cloud.colors)}
	filename_num=get_file_name(path, file_name)
	sp.io.savemat(filename_num, mdic)
	print("saved to: "+filename_num)

def save_raw_cloud():
	# get map from rtabmap
	raw_cloud=None
	try:
		while raw_cloud==None or len(raw_cloud.points)==0:
			get_map_client()
			raw_cloud=msg2pc(map_data)
	except KeyboardInterrupt:
		print("Terminating")
	print("Got Map")


	print("raw cloud size:"+str(len(np.asarray(raw_cloud.points))))
	
	if (loop_input("Plot raw cloud?")):
		drawcloud([raw_cloud], size=5)	

	if (loop_input("Save raw cloud?")):
		save_mat(path=os.path.expanduser("~/output/"), file_name="Raw_Cloud", cloud=raw_cloud)

	return raw_cloud

def icp_cloud(raw_cloud):
    CAD_cloud=o3d.geometry.PointCloud()
    CAD_cloud.points=o3d.utility.Vector3dVector(CAD)
    cloud_icp_ds=raw_cloud.voxel_down_sample(0.01)
    cloud_icp_ds=random_downsample(cloud_icp_ds,percentage=0.2)
    print("Aligning map with CAD model...")
	#tf_init=np.asarray([[-1,0,0,0],[0,-1,0,0],[0,0,1,],[0,0,0,1]])
    tf_init=np.asarray([[1,0,0,0.5],[0,1,0,-1],[0,0,1,0],[0,0,0,1]])
    tf1=o3d.pipelines.registration.registration_icp(cloud_icp_ds, CAD_cloud,icp_thres,
                                                    tf_init,o3d.pipelines.registration.TransformationEstimationPointToPoint())
	
    cloud_icp_ds.transform(tf1.transformation)

    croped_points=crop_cloud(cloud_icp_ds,[-0.01, 6],[-3, 0.1],[-0.01, 1.5])
    cloud_icp_ds.points=o3d.utility.Vector3dVector(croped_points)

    tf2=o3d.registration_icp(cloud_icp_ds, CAD_cloud,icp_thres,np.asarray([[1,0,0,0],[0,1,0,0],[0,0,1,0],		 		 						[0,0,0,1]]),o3d.TransformationEstimationPointToPoint())

    tf=np.matmul(tf2.transformation,tf1.transformation)
    raw_cloud.transform(tf)
    croped_points=crop_cloud(raw_cloud,[-0.01, 6],[-3, 0.1],[-0.01, 1.5])
    cropped_cloud=o3d.geometry.PointCloud()
    cropped_cloud.points=o3d.utility.Vector3dVector(croped_points)
    cloud=random_downsample(cropped_cloud,percentage=0.05)
    cloud=o3d.geometry.voxel_down_sample(cropped_cloud,0.0075)
    return cloud, CAD_cloud

def save_cloud():
    raw_cloud=save_raw_cloud()
    aligned_cloud, CAD_cloud=icp_cloud(raw_cloud)
    return aligned_cloud, CAD_cloud

if __name__ == "__main__":
	try:
		rospy.init_node('FOD_Detection',anonymous=False)
		save_raw_cloud()	
	except KeyboardInterrupt:
		print("Terminating")



