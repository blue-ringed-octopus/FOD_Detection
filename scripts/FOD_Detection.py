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
import save_cloud
import colorsys as cs
import Local_Covariance_Trainer as pclib
from scipy.cluster import hierarchy
from scipy.spatial import KDTree

def Loop_input(prompt):
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
        trained_param=pickle.load( open(self.navsea+"/param/detection params.p", "rb" ) )
        with open(self.navsea+"/param/fod_detection_params.yaml", 'r') as file:
            params= yaml.safe_load(file)
        params['fod_detection'].update(trained_param)    
        self.params=params
     
        

    #-----------------------Functions----------------------
    def fetch_cloud(self):
        '''
        read reference pointcloud and it's info from given uri
        request and get point cloud from rtabmap through ros services 
        '''
        reference_cloud_uri=self.params["preprocess"]["reference_cloud_uri"]
        self.ref_cloud=o3d.io.read_point_cloud(reference_cloud_uri)
        cloud_info_uri=self.params["fod_detection"]["cloud_info_uri"]
        self.reference_cloud_info = pickle.load( open(cloud_info_uri, "rb" ) )
        
        icp_thres=self.params["preprocess"]["icp_threshold"]
        reference_cloud_uri=self.params["preprocess"]["reference_cloud_uri"]
        self.fetcher=save_cloud.Pointcloud_fetcher(icp_thres,reference_cloud_uri)
        self.fetcher.get_raw_cloud()
        self.raw_cloud=self.fetcher.raw_cloud
        self.fetcher.save_raw_cloud()
        
    def process_cloud(self):
        '''
        Denoise pointcloud, then icp with reference pointcloud, and random downsample
        '''
        rospy.loginfo("Denoising PointCloud")
        params=self.params["preprocess"]
        denosie_neightbor=params["denosie_neightbor"]
        denoise_std=params["denoise_std"]
        cloud, idx=self.raw_cloud.remove_statistical_outlier(nb_neighbors=denosie_neightbor,
                                                            std_ratio=denoise_std)
        
        rospy.loginfo("Registering Pointcloud to reference")
        icp_thres=params["icp_threshold"]
        icp_tf_init=np.asarray(params["icp_tf_init"])
        bound=self.reference_cloud_info['bound']
        bound[2,1]=0.6

        cloud_sparse=cloud.voxel_down_sample(0.05)
        tf=(o3d.pipelines.registration.registration_icp(
            cloud_sparse, self.ref_cloud, icp_thres, icp_tf_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)))      
        
        cloud=cloud.transform(tf.transformation)
        in_bound=pclib.crop_cloud_par(np.asarray(cloud.points), bound)
        self.tf=tf.transformation
        self.cloud=cloud.select_by_index(in_bound)
       
        rospy.loginfo("Creating KD-tree")
        self.cloud_ds,_=pclib.random_downsample(self.cloud, params["downsample_rate"])
        self.cloud_ds_tree=KDTree(np.asarray(self.cloud_ds.points))
            
   
    def calculate_discrep(self):
        '''
        calculate discrepency from map to reference pointcloud
        '''
        reference_cloud=self.ref_cloud
        metric=self.params["fod_detection"]["metric"]
        if metric=="l2":
            dist=pclib.calculate_discrep(target_tree=self.reference_cloud_info["kd-tree"],
                                         cloud=np.asarray((self.cloud_ds).points))
        else:
            dist=pclib.calculate_discrep(target=np.asarray(reference_cloud.points), 
                                         target_tree=self.reference_cloud_info["kd-tree"],
                                         cloud=np.asarray((self.cloud_ds).points), 
                                         cov_inv=self.reference_cloud_info["inverse covariance"])
        self.dist=dist    
        
    
    def blur_dist(self):
        '''
        smoothing discrepiency using neighborhood
        '''
        params=self.params["smoothing"]
        self.blur_dist=pclib.multi_blur(self.cloud_ds,self.cloud_ds, 
                                          self.dist,cloud_tree=self.cloud_ds_tree
                                          ,k=int(params['k']),
                                          num_iter=int(params['iteration']))
    
    def Segment_FOD(self):
        '''
        seperate points with high discrepency from the map 
        '''
        params=self.params['fod_detection']
       # cloud_ds_ds, idx=pclib.random_downsample(self.cloud_ds, params["downsample_rate"])
       # dist=self.blur_dist[idx]
        cloud_vox,_, idx=self.cloud_ds.voxel_down_sample_and_trace(self.params["clustering"]["cluster_voxel_size"], 
                                                                   self.cloud_ds.get_min_bound(), 
                                                                   self.cloud_ds.get_max_bound(), 
                                                                   False)
        dist=np.asarray([np.average(self.blur_dist[i]) for i in idx])
        self.metric=params['metric']
        if self.metric=="l2":
            cutoff=params['l2-dist cutoff']

        else:
            cutoff=params['m-dist cutoff']

        # fods=cloud_vox.select_by_index(np.where(dist>=cutoff)[0])
        # tank=cloud_vox.select_by_index(np.where(dist<cutoff)[0])
        fods, tank, fod_mdist =pclib.Isolate_FOD(cloud_vox, dist, cutoff)

        self.fod_dist=dist[dist>=cutoff]
        fods.paint_uniform_color([1,0,0])
        tank.paint_uniform_color([0.2, 0.2 ,0.2])
        
        self.fods=fods
        self.tank=tank
    
    def Fod_clustering(self):
        '''
        cluster high discrepency points into fod candidates
        '''
        params=self.params['clustering']
        cloud=self.fods
        minsize=params['minimum_fod_point_count']
        dist=self.fod_dist
        if self.metric=="l2":
            cutoff=self.params['fod_detection']["L2-dist cluster cutoff"]
        else:
            cutoff=self.params['fod_detection']["m-dist cluster cutoff"]

        # points=np.asarray(cloud.points)
        # if len(points)<=minsize:
        #     print("no fod")
        #     return
        # labels=hierarchy.fclusterdata(points, criterion='distance',t=cutoff)-1
        # num_point=np.bincount(labels)
        # print(num_point)
        # clouds=[]
        # dists=[]
        # for i in range(max(labels)+1):
        #     if num_point[i]>=minsize:
        #         pointlist=[points[j] for j in range(len(points)) if i==labels[j]]
        #         if len(dist)!=0:
        #             dists+=[[dist[j] for j in range(len(points)) if i==labels[j]]]
        #         clouds.append(pclib.Cloud_from_points(pointlist))
        # for i in range(len(clouds)):
        #     rgb=cs.hsv_to_rgb(float(i)/len(clouds),1,1)
        #     clouds[i].paint_uniform_color(rgb)
            
        FOD_clusters, cluster_dists=pclib.Fod_clustering(cloud,minsize, cutoff,dist)

        self.fods_list=FOD_clusters
        self.fod_dist=cluster_dists
        
    def Cluster_centroid(self):
        '''
        calculate centroids of clusters, weighted by the dicrepency 
        '''
        clusters=self.fods_list
        weights=self.fod_dist
        centroids=[]
        # for i, cluster in enumerate(clusters):
        #     points=np.asarray(cluster.points)
        #     centroids.append(np.average(points,axis=0, weights=weights[i]))
        centroids=pclib.Cluster_centroid(clusters, weights=weights)

        self.fod_centroids=np.asarray(centroids)
    
    
    # def plot_fod_centroid(self):
    #     centroids=self.fod_centroids
    #     base_pc=self.cloud
    #     spheres=[]
    #     for points in centroids:
    #         spheres.append(o3d.geometry.TriangleMesh.create_sphere(radius=0.05))
    #         tf=np.eye(4)
    #         tf[0:3,3]=points
    #         spheres[-1]=spheres[-1].transform(tf)
    #         spheres[-1].paint_uniform_color([1,0,0])
    #     o3d.visualization.draw_geometries([base_pc]+spheres)
    
    def Project_obsticles(self):
        '''
        project potential obsticles from the map to the ground plane for waypoint generation
        '''
        obsticle_cloud=self.fods.voxel_down_sample(0.05)
        idx=pclib.crop_cloud_par(np.asarray(obsticle_cloud.points), 
                                 [[-np.inf, np.inf],[-np.inf, np.inf],[0.05,1]])
        obsticle_cloud=obsticle_cloud.select_by_index(idx)
        obsticle_points=np.asarray(obsticle_cloud.points)
        obsticle_points=np.delete(obsticle_points,2, axis=1)
        return obsticle_points
        
    def FOD_Detection_routine(self):
        print("FOD detection module started")
        
        #get fod cloud
        self.fetch_cloud()
        
        #FOD cloud ICP
        self.process_cloud()
        
       	 #Calculate dist
        self.calculate_discrep()
        # blur
        self.blur_dist()
        
        #segmentation 
        self.Segment_FOD()
        
        #cluster 
        self.Fod_clustering()
        self.Cluster_centroid()
        if (Loop_input("Plot Clustes?")):
            o3d.visualization.draw_geometries(self.fods_list+[self.tank], window_name="Clustering")

        if (Loop_input("Plot FOD centroids?")):
            pclib.plot_fod_centroid(self.cloud, self.fod_centroids, window_name="FOD candidates")
        
        # project obsticles 
        self.Project_obsticles()
        
        rospy.loginfo("fod detection finished! ")
        	
if __name__ == "__main__":
    rospy.init_node('FOD_Detection',anonymous=False)
    fod_detector=FOD_Detector()
    fod_detector.FOD_Detection_routine()

