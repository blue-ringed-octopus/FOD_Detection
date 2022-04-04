#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Jan  18 23:13:00 2021

@author: Benjamin
"""
import pickle
import yaml
import numpy as np
import rospy 
import rospkg
import open3d as o3d
import copy
import os
import save_cloud
import colorsys as cs
import Local_Covariance_Trainer as pclib
from scipy.io import loadmat
from scipy.interpolate import RegularGridInterpolator 
from scipy.cluster import hierarchy
from scipy.spatial import KDTree

def Loop_input(self, prompt):
    	inp=""
    	while inp.lower()!='y' and inp.lower()!='n':
    		inp=input(prompt+"(y/n): \n")
    	save=inp.lower()=='y'
    	return save
    
    
class FOD_Detector:
    
    '''
    Contructor
    '''
    def __init__(self):
        rospack=rospkg.RosPack()
        self.navsea=rospack.get_path('navsea')
        self.icp_thres=rospy.get_param("icp_threshold")
        trained_param=pickle.load( open(self.navsea+"param/detection params.p", "rb" ) )
        with open(self.navsea+"param/fod_detection_params.yaml", 'r') as file:
            params= yaml.safe_load(file)
        params['fod_detection'].update(trained_param)    
        self.params=params
     
        

    #-----------------------Functions----------------------
    def fetch_cloud(self):
        reference_cloud_uri=self.params["preprocess"]["reference_cloud_uri"]
        self.ref_cloud=o3d.io.read_point_cloud(reference_cloud_uri)
        cloud_info_uri=self.params["fod_detection"]["cloud_info_uri"]
        self.reference_cloud_info = pickle.load( open(cloud_info_uri, "rb" ) )
        
        icp_thres=self.params["preprocess"]["icp_threshold"]
        reference_cloud_uri=self.params["preprocess"]["reference_cloud_uri"]
        self.fetcher=save_cloud.Pointcloud_fetcher(icp_thres,reference_cloud_uri)
        self.fetcher.get_raw_cloud()
        self.raw_cloud=self.fetcher.raw_cloud
        
    def process_cloud(self):
            rospy.loginfo("Denoising PointCloud")
            params=self.params["preprocess"]
            denosie_neightbor=params["denosie_neightbor"]
            denoise_std=params["denoise_std"]
            cloud=self.raw_cloud.remove_statistical_outlier(nb_neighbors=denosie_neightbor,
                                                                std_ratio=denoise_std)
            
            rospy.loginfo("Registering Pointcloud to reference")
            icp_thres=params["icp_threshold"]
            icp_tf_init=np.asarray(params["icp_tf_init"])
            bound=self.reference_cloud_info['bound']
            cloud_sparse=cloud.voxel_down_sample(0.05)
            tf=(o3d.pipelines.registration.registration_icp(
                cloud_sparse, self.ref_cloud, icp_thres, icp_tf_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)))      
            
            cloud=cloud.transform(tf.transformation)
            in_bound=pclib.crop_cloud_par(np.asarray(cloud.points), bound)
            self.cloud=cloud.select_by_index(in_bound)
           
            rospy.loginfo("Creating KD-tree")
            self.cloud_ds=pclib.random_downsample(cloud, params["downsample_rate"])
            self.cloud_ds_trees.append(KDTree(np.asarray(self.cloud_ds.points)))
            
   
    def calculate_discrep(self):
        reference_cloud=self.ref_cloud
        metric=self.params["fod_detection"]["metric"]
        if metric=="l2":
            dist=pclib.calculate_discrep(target_tree=self.reference_cloud_info["kd-tree"],
                                         cloud=np.asarray((self.cloud_ds).points))
        else:
            dist=pclib.calculate_discrep(target=np.asarray(reference_cloud.points), 
                                         target_tree=self.reference_cloud_info["kd-tree"],
                                         cloud=np.asarray((self.cloud_ds).points, 
                                         cov_inv=self.reference_cloud_info["inverse covariance"]))
        self.dist=dist    
        
    
    def blur_dist(self):
        params=self.params["smoothing"]
        self.blur_dist=pclib.multi_blur(self.cloud_ds,self.cloud_ds, 
                                          self.dist,cloud_tree=self.cloud_ds_tree
                                          ,k=int(params['k']),
                                          num_iter=int(params['iteration']))
    
    def Segment_FOD(self):
        params=self.params['fod_detection']
        cloud_ds_ds, idx=pclib.random_downsample(self.cloud_ds, params["downsample_rate"])
        dist=self.blur_dist[idx]
        self.metric=params['metric']
        if self.metric=="l2":
            cutoff=params['l2-dist cutoff']

        else:
            cutoff=params['m-dist cutoff']

        fods=cloud_ds_ds.select_by_index(np.where(dist>=cutoff)[0])
        tank=cloud_ds_ds.select_by_index(np.where(dist<cutoff)[0])
        
        self.fod_dist=dist[dist>=cutoff]
        fods.paint_uniform_color([1,0,0])
        tank.paint_uniform_color([0.2, 0.2 ,0.2])
        
        self.fods=fods
        self.tank=tank
    
    def Fod_clustering(self):
        params=self.params['clustering']
        cloud=self.fods
        minsize=params['minimum_fod_point_count']
        dist=self.fod_dist
        if self.metric=="l2":
            cutoff=self.params['fod_detection']["m-dist cluster cutoff"]
        else:
            cutoff=self.params['fod_detection']["l2-dist cluster cutoff"]
            
        points=np.asarray(cloud.points)
        if len(points)<=minsize:
            print("no fod")
            return
        labels=hierarchy.fclusterdata(points, criterion='distance',t=cutoff)-1
        num_point=np.bincount(labels)
        print(num_point)
        clouds=[]
        dists=[]
        for i in range(max(labels)+1):
            if num_point[i]>=minsize:
                pointlist=[points[j] for j in range(len(points)) if i==labels[j]]
                if len(dist)!=0:
                    dists+=[[dist[j] for j in range(len(points)) if i==labels[j]]]
                clouds.append(pclib.Cloud_from_points(pointlist))
        for i in range(len(clouds)):
            rgb=cs.hsv_to_rgb(float(i)/len(clouds),1,1)
            clouds[i].paint_uniform_color(rgb)
            
        self.fods=clouds
        self.fod_dist=dists
        
    def Cluster_centroid(self):
        '''
        documentation
        '''
        clusters=self.fods
        weights=self.fod_dist
        centroids=[]
        for i, cluster in enumerate(clusters):
            points=np.asarray(cluster.points)
            centroids.append(np.average(points,axis=0, weights=weights[i]))
        
        self.fod_centroids=np.asarray(centroids)
    
    
    def plot_fod_centroid(base_pc, centroids):
        spheres=[]
        for points in centroids:
            spheres.append(o3d.geometry.TriangleMesh.create_sphere(radius=0.05))
            tf=np.eye(4)
            tf[0:3,3]=points
            spheres[-1]=spheres[-1].transform(tf)
            spheres[-1].paint_uniform_color([1,0,0])
        o3d.visualization.draw_geometries([base_pc]+spheres)
        

    
         

    
    # def Get_file_name(path, file_name):
    # 	i=0
    # 	while os.path.exists(path+file_name+"_"+str(i)+".mat"):
    # 		i=i+1
    # 	return (path+file_name+"_"+str(i)+".mat")

    # def Cloud_m_dist(cloud):
    # 	mdist=M_dist_interp(np.asarray(cloud.points))
    # 	cloud.paint_uniform_color([0.2,0.2,0.2])
    # 	np.asarray(cloud.colors)[np.where(np.isnan(mdist))[0],:]=[0,1,1]
    # 	np.asarray(cloud.colors)[np.where(np.isinf(mdist))[0],:]=[0,1,1]
    # 	mdist[np.where(np.isnan(mdist))]=0
    # 	mdist[np.where(np.isinf(mdist))]=np.max(mdist)
    # 	np.asarray(cloud.colors)[np.where(mdist>2.5)[0],:]=[1,0,0] #2.5
    # 	if (Loop_input("Plot FOD?")):
    #   		Drawcloud([cloud], size=0.1)
    # 	return mdist
    
    def Isolate_FOD(cloud, dist, cutoff):
        FODS=cloud.select_by_index(np.where(dist>=cutoff)[0])
        Tank=cloud.select_by_index(np.where(dist<cutoff)[0])
        
        FOD_dist=dist[dist>=cutoff]
        FODS.paint_uniform_color([1,0,0])
        Tank.paint_uniform_color([0.2, 0.2 ,0.2])
        return FODS, Tank, FOD_dist

    
def FOD_Detection_routine():
    print("FOD detection module started")
    fod_detector=FOD_Detector()
    
    #get fod cloud
    fod_detector.fetch_cloud()
    
    #FOD cloud ICP
    fod_detector.process_cloud()
    
   	 #Calculate dist
    fod_detector.calculate_discrep()
    # blur
    fod_detector.blur_dist()
    
    #segmentation 
    fod_detector.Segment_FOD()
    
    #cluster 
    fod_detector.Fod_clustering()
    fod_detector.Cluster_centroid()
    
    if (Loop_input("Plot FOD centroids?")):
        fod_detector.plot_fod_centroid
    
    # 	centroid=Cluster_centroid(FOD_clusters) #find FOD centroids 
    
    # project obsticles 
    obsticle_points=Project_obsticles()
    	
    return fod_detector

if __name__ == "__main__":
    rospy.init_node('FOD_Detection',anonymous=False)
    FOD_Detection_routine()

