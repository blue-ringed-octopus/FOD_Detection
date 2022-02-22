#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Jan  18 23:13:00 2021

@author: Benjamin
"""
import pickle
import numpy as np
import ros_numpy
import rospy 
import rospkg
import threading
import time
import open3d as o3d
import copy
import scipy as sp
import os
from sensor_msgs.msg import PointCloud2
import Local_Covariance_Trainer as pclib

def loop_input(prompt):
       inp=""
       while inp.lower()!='y' and inp.lower()!='n':
           inp=input(prompt+"(y/n): \n")
           save=inp.lower()=='y'
       return save
    
class Pointcloud_fetcher:
    '''
    Contructor
    '''
    def __init__(self,icp_thres, reference_cloud_uri):
        self.reference_cloud=o3d.io.read_point_cloud(reference_cloud_uri)
        self.done=False
        self.map_data=None
        self.icp_thres=icp_thres
              
    def draw_registration_result(source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.draw_geometries([source_temp, target_temp])
    
    def get_map_thread(self):  	
    	#map_data=rospy.wait_for_message('rtabmap/cloud_map',PointCloud2)
    	self.map_data=rospy.wait_for_message('rtabmap/map_assembler/cloud_map',PointCloud2)
    	self.done = True
    
    def get_map_client(self):
    	thread = threading.Thread(target=self.get_map_thread)
    	print("Getting map")
    	try:
    		thread.start()
    		time.sleep(0.5)
    		#publish_map=rospy.ServiceProxy('rtabmap/publish_map',PublishMap)
    		#publish_map=rospy.ServiceProxy('rtabmap/map_assembler/publish_map',PublishMap)
    		#publish_map(1,1,0)
    		print("requested map")
    		while not self.done:
    			time.sleep(0.5)
    		#return data
    	except rospy.ServiceException as e:
    		print("Service all failed: %s"%e)
    
    def msg2pc(self):
        pc=ros_numpy.numpify(self.map_data)
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
        self.raw_cloud=p
        
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
    
    def get_file_name(path, file_name):
    	i=0
    	while os.path.exists(path+file_name+"_"+str(i)+".mat"):
    		i=i+1
    	return (path+file_name+"_"+str(i)+".mat")
    
    def save_mat(self, path, file_name, cloud):
    	mdic={"cloud":np.asarray(cloud.points), "rgb":np.asarray(cloud.colors)}
    	filename_num=self.get_file_name(path, file_name)
    	sp.io.savemat(filename_num, mdic)
    	print("saved to: "+filename_num)
    
    def save_raw_cloud(self):
    	# get map from rtabmap
    	self.raw_cloud=None
    	try:
    		while self.raw_cloud==None or len(self.raw_cloud.points)==0:
    			self.get_map_client()
    			self.msg2pc()
    	except KeyboardInterrupt:
    		print("Terminating")
    	print("Got Map")
    
    
    	print("raw cloud size:"+str(len(np.asarray(self.raw_cloud.points))))
    	
    	if (loop_input("Plot raw cloud?")):
            test=self.raw_cloud
            self.drawcloud(clouds=[test], size=5)	
    
    	if (loop_input("Save raw cloud?")):
    		self.save_mat(path=os.path.expanduser("~/output/"), file_name="Raw_Cloud", cloud=self.raw_cloud)
        
    def process_raw_cloud(self):
        cloud_icp_ds=self.raw_cloud.voxel_down_sample(0.01)
        cloud_icp_ds=pclib.random_downsample(cloud_icp_ds,percentage=0.2)
        print("Aligning map with reference model...")
    	#tf_init=np.asarray([[-1,0,0,0],[0,-1,0,0],[0,0,1,],[0,0,0,1]])
        tf_init=np.asarray([[1,0,0,0.5],[0,1,0,-1],[0,0,1,0],[0,0,0,1]])
        tf=o3d.pipelines.registration.registration_icp(cloud_icp_ds, self.reference_cloud,icp_thres,
                                                        tf_init,
                                                        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    	
       # cloud_icp_ds.transform(tf1.transformation)
    
       # croped_points=crop_cloud(cloud_icp_ds,[-0.01, 6],[-3, 0.1],[-0.01, 1.5])
     #   cloud_icp_ds.points=o3d.utility.Vector3dVector(croped_points)
    
       # tf2=o3d.registration_icp(cloud_icp_ds, CAD_cloud,icp_thres,np.asarray([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]),o3d.TransformationEstimationPointToPoint())
    
   #     tf=np.matmul(tf2.transformation,tf1.transformation)
        self.raw_cloud.transform(tf)
      #  croped_points=crop_cloud(raw_cloud,[-0.01, 6],[-3, 0.1],[-0.01, 1.5])
        self.processed_cloud=self.raw_cloud
        self.tf=tf
        self.tf_inv=np.linalg.inv(tf)
        print(tf)
        print("Done")
        # cloud=pclib.random_downsample(cropped_cloud,percentage=0.05)
        # cloud=o3d.geometry.voxel_down_sample(cropped_cloud,0.0075)
    
if __name__ == "__main__":
    try:
        rospy.init_node('FOD_Detection',anonymous=False)
        rospack=rospkg.RosPack()
        navsea=rospack.get_path('navsea')
        reference_cloud_uri=navsea+"/resource/mean_cloud.pcd"
        icp_thres=5
        fetcher=Pointcloud_fetcher(icp_thres,reference_cloud_uri)
        fetcher.save_raw_cloud()
        fetcher.process_raw_cloud()
    except KeyboardInterrupt:
        print("Terminating")



