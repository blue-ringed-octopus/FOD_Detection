#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Jan  18 23:13:00 2021

@author: Benjamin
"""
#requires pip install open3d-python==0.3.0.0

import numpy as np
import rospy 
import rospkg
import open3d as o3d
import copy
import os
import save_cloud
import colorsys as cs
from scipy.io import loadmat
from scipy.interpolate import RegularGridInterpolator 
from scipy.cluster import hierarchy

rospack=rospkg.RosPack()
navsea=rospack.get_path('navsea')

#--------------Load variables-------------------------
x = loadmat(navsea+'/scripts/FOD_detection_var.mat')
cov_global=np.array(x['cov_global'])
Mdist_table=np.array(x['Mdist_table'])
xgrid=np.reshape(np.array(x['x']),(-1))
ygrid=np.reshape(np.array(x['y']),(-1))
zgrid=np.reshape(np.array(x['z']),(-1))

#-----------------------Functions----------------------
def Loop_input(prompt):
	inp=""
	while inp.lower()!='y' and inp.lower()!='n':
		inp=raw_input(prompt+"(y/n): \n")
	save=inp.lower()=='y'
	return save


def Drawcloud(clouds, size):
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
def Crop_cloud(cloud,xlim,ylim,zlim):
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


def Fod_clustering(points, minsize, cutoff):
	labels=hierarchy.fclusterdata(points, criterion='distance',t=cutoff)
	#labels=hierarchy.fclusterdata(points, criterion='inconsistent',t=1.25)
	num_point=np.bincount(labels)
	print(num_point)
	clouds=[]
	for i in range(max(labels)):
		if num_point[i+1]>=minsize:
			pointlist=[]
			for j in range(len(points)):
				if i+1==labels[j]:
					pointlist.append(points[j])
			clouds.append(Cloud_from_points(pointlist))
	for i in range(len(clouds)):
		rgb=cs.hsv_to_rgb(float(i)/len(clouds),1,1)
		clouds[i].paint_uniform_color(rgb)
	return clouds

def Cloud_from_points(points):
	tempcloud=o3d.PointCloud()
	tempcloud.points=o3d.Vector3dVector(np.asarray(points))
	return tempcloud

def Mean_m_dist(points, cloud, mdist):
	region_mdist=[]
	sigma=0.05		
	tree=o3d.KDTreeFlann(cloud)
	j=1
	cloudpoints=np.asarray(cloud.points)
	for point in points:
		print("Convoluting point: "+str(j)+'/'+str(len(points)))
		j=j+1
		[k, idx,_]=tree.search_radius_vector_3d(point, sigma*4)

		dist=np.linalg.norm(point-cloudpoints[idx],axis=1)
		w=np.exp((-1)*((dist/sigma)**2)/2)
		wMDist=np.dot(mdist[idx],w)
		sumW=sum(w)
		region_mdist.append(wMDist/sumW)
	return np.asarray(region_mdist)


def Cluster_centroid(clusters):
#todo: weighted centroid with Mdist
	centroids=[]
	for cluster in clusters:
		points=np.asarray(cluster.points)
		centroids.append(np.mean(points,axis=0))
	return np.asarray(centroids)

def Get_file_name(path, file_name):
	i=0
	while os.path.exists(path+file_name+"_"+str(i)+".mat"):
		i=i+1
	return (path+file_name+"_"+str(i)+".mat")

def Save_mat(path, file_name, cloud):
	mdic={"cloud":np.asarray(cloud.points)}
	filename_num=Get_file_name(path, file_name)
	sp.io.savemat(filename_num, mdic)
	print("saved to: "+filename_num)

def Cloud_m_dist(cloud):
	mdist=M_dist_interp(np.asarray(cloud.points))
	cloud.paint_uniform_color([0.2,0.2,0.2])
	np.asarray(cloud.colors)[np.where(np.isnan(mdist))[0],:]=[0,1,1]
	np.asarray(cloud.colors)[np.where(np.isinf(mdist))[0],:]=[0,1,1]
	mdist[np.where(np.isnan(mdist))]=0
	mdist[np.where(np.isinf(mdist))]=np.max(mdist)
	np.asarray(cloud.colors)[np.where(mdist>2.5)[0],:]=[1,0,0] #2.5
	if (Loop_input("Plot FOD?")):
  		Drawcloud([cloud], size=0.1)
	return mdist

def Isolate_FOD(cloud,CAD_cloud, mdist,convolution,cad_project=False):
	if(convolution):
		if cad_project:
			cloud_ds=Cloud_from_points(Crop_cloud(CAD_cloud,[-0.01, 6],[-3, 0.1],[-0.01, 1.5]))
			cloud_ds=o3d.voxel_down_sample(cloud_ds,0.1)
			cloudPointSize=20
			clusterPointMinNum=0 
			cluster_cutoff=0.275
		else:
			cloud_ds=o3d.voxel_down_sample(cloud,0.075)
			print("downsampled from "+str(len(cloud.points))+"to"+str(len(cloud_ds.points)))
			cluster_cutoff=0.275
			cloudPointSize=20
			clusterPointMinNum=5 

		points=cloud_ds.points
		m_dist=Mean_m_dist(np.asarray(cloud_ds.points), cloud, mdist)
		cutoff=1.75

	else:
		cloudPointSize=0.1
		clusterPointMinNum=10
		cluster_cutoff=0.275

	tankpoints=np.asarray(cloud.points)[mdist<cutoff]
	fodpoints=np.asarray(cloud.points)[mdist>=cutoff]


	FODS=Cloud_from_points(fodpoints)
	Tank=Cloud_from_points(fodpoints)

	FODS.paint_uniform_color([1,0,0])
	Tank.paint_uniform_color([0.2, 0.2 ,0.2])
	#Drawcloud([Tank], size=cloudPointSize)
	return FODS, Tank, clusterPointMinNum, cluster_cutoff

def Project_obsticles(cloud):
 	obsticle_cloud=o3d.voxel_down_sample(cloud,0.05)
	obsticle_points=np.asarray(Crop_cloud(obsticle_cloud,[-np.inf, np.inf],[-np.inf, np.inf],[0.05,1]))
 	obsticle_points=np.delete(obsticle_points,2, axis=1)
	return obsticle_points

def FOD_Detection_routine(convolution=False):
	print("FOD detection module started")
	cloud, CAD_cloud=save_cloud.save_cloud()  # get cloud	
	mdist=Cloud_m_dist(cloud)	#Point-wise m-distance

	#ToDo: denoise point cloud

	FOD, Tank, clusterPointMinNum, cluster_cutoff= Isolate_FOD(cloud,CAD_cloud, mdist,convolution) #separate fod and tank cloud points

	FOD_clusters=Fod_clustering(FOD.points,clusterPointMinNum, cluster_cutoff) #Cluster FOD points to FOD clouds 

	
	if (Loop_input("Plot Clusters?")):
		recombined=copy.deepcopy(FOD_clusters)
		recombined.append(Tank)
		o3d.draw_geometries(recombined)

	if not (convolution):
		if (Loop_input("Save FOD Clusters?")):
			#Save Fods for ML
			for fod in FOD_clusters:
				save_mat(path=navsea+"/output/FOD_pointcloud/", file_name="FODpc", cloud=fod)

	centroid=Cluster_centroid(FOD_clusters) #find FOD centroids 

	# project obsticles 
 	obsticle_points=Project_obsticles(cloud)
	
	return np.asarray(centroid),obsticle_points

if __name__ == "__main__":
	rospy.init_node('FOD_Detection',anonymous=False)
	FOD_Detection_routine(True)

