#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 17:19:22 2021

@author: BW
"""
import numpy as np 
from numba import cuda
from scipy.spatial import KDTree
import open3d as o3d
import colorsys as cs
import copy
from scipy.cluster import hierarchy
import random
def Isolate_FOD(cloud, dist, cutoff):
    FODS=cloud.select_by_index(np.where(dist>=cutoff)[0])
    Tank=cloud.select_by_index(np.where(dist<cutoff)[0])
    
    FOD_dist=dist[dist>=cutoff]
    FODS.paint_uniform_color([1,0,0])
    Tank.paint_uniform_color([0.2, 0.2 ,0.2])
    return FODS, Tank, FOD_dist

def Fod_clustering(cloud, minsize, cutoff, dist=[]):
    points=np.asarray(cloud.points)
    if len(points)<=minsize:
        print("no fod")
        return [],[]
    labels=hierarchy.fclusterdata(points, criterion='distance',t=cutoff)-1
    num_point=np.bincount(labels)
    clouds=[]
    dists=[]
    for i in range(max(labels)+1):
        if num_point[i]>=minsize:
            pointlist=[points[j] for j in range(len(points)) if i==labels[j]]
            if len(dist)!=0:
                dists+=[[dist[j] for j in range(len(points)) if i==labels[j]]]
            clouds.append(Cloud_from_points(pointlist))
    for i in range(len(clouds)):
        rgb=cs.hsv_to_rgb(float(i)/len(clouds),1,1)
        clouds[i].paint_uniform_color(rgb)
    return clouds, dists    

def Cluster_centroid(clusters, weights=None):
    '''
    documentation
    '''
    centroids=[]
    for i, cluster in enumerate(clusters):
        points=np.asarray(cluster.points)
        centroids.append(np.average(points,axis=0, weights=weights[i]))
    return np.asarray(centroids)

def plot_fod_centroid(base_pc, centroids, window_name="centroid"):
    spheres=[]
    for points in centroids:
        spheres.append(o3d.geometry.TriangleMesh.create_sphere(radius=0.05))
        tf=np.eye(4)
        tf[0:3,3]=points
        spheres[-1]=spheres[-1].transform(tf)
        spheres[-1].paint_uniform_color([1,0,0])
    o3d.visualization.draw_geometries([base_pc]+spheres,window_name=window_name)
    
def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.draw_geometries([source_temp, target_temp])

def drawcloud(clouds, size,window_name="Open3D"):
	vis=o3d.visualization.VisualizerWithEditing()
	vis.create_window(window_name=window_name)
	ro=o3d.visualization.RenderOption()
	ro=vis.get_render_option()
	ro.point_size=size
	ro.show_coordinate_frame=True
	for cloud in clouds:
		vis.add_geometry(cloud)
	vis.run()
	vis.destroy_window()

def random_downsample(cloud, percentage):
    og_size=len(np.asarray(cloud.points))
    ds_size=int(np.floor(len(np.asarray(cloud.points))*percentage))
    print("Downsampling from "+ str(og_size)+" to " +str(ds_size))
    idx=random.sample(range(len(cloud.points)),ds_size)
    idx.sort()
    cloud_ds=cloud.select_by_index(idx)
    return cloud_ds, idx

def Cloud_from_points(points, rgb=None):
    '''
    creat point cloud object from list of points
    '''
    cloud=o3d.geometry.PointCloud()
    cloud.points=o3d.utility.Vector3dVector(np.asarray(points))
    if not rgb is None:
        cloud.colors=o3d.utility.Vector3dVector(rgb)
    return cloud
    
TPB=128
@cuda.jit
def crop_cloud_kernel(d_out, d_cloud, d_lim):
    i=cuda.grid(1)
    nx=d_cloud.shape[0]
    if i<nx:
        in_bound=True
        for j in range(3):
            in_bound=in_bound and d_cloud[i,j]>d_lim[j,0] and d_cloud[i,j]<d_lim[j,1] 
        d_out[i]=in_bound
    
def crop_cloud_par(cloud,lim):
    #lim=[[x_min, x_max],[y_min, y_max],[z_min, z_max]]
    nx=cloud.shape[0]
    d_cloud=cuda.to_device(cloud)
    d_lim=cuda.to_device(lim)
    d_out=cuda.device_array(nx,dtype=bool)
    thread=(TPB)
    blocks=(nx+TPB-1)//TPB
    crop_cloud_kernel[blocks, thread](d_out, d_cloud, d_lim)
    in_bound=d_out.copy_to_host()
    return np.where(in_bound)[0]
    
@cuda.jit()
def mean_blur_kernel(d_out, d_cloud, d_neighbors,d_dists_mat, d_weights):
    '''
    kernel to calculate mean intensity from nearest neighbors
    input: d_out: device array to store mean intensity for each point
           d_cloud: device array of points
           d_neighbors: device array of nearest neighbors for each point
           d_dists_mat: intensity values for each point
    '''
    
    nx=d_cloud.shape[0]
    i=cuda.grid(1)
    if i<nx:
        d_out[i]=0
        weight_sum=0
        for j in d_neighbors[i]:
            d_out[i]+=d_weights[j]*d_dists_mat[j]
            weight_sum+=d_weights[j]
    #    d_out[i]= d_out[i]/d_neighbors[i].shape[0]
        d_out[i]= d_out[i]/weight_sum
    
def mean_blur_par(cloud, dists_mat, tree, num_neighbors, weights):
    '''
    Wrapper for mean blur kernel
    '''
    _,neighbors=tree.query(cloud, k=num_neighbors)
    neighbors=neighbors.astype(np.int32)    
    nx=cloud.shape[0]
    d_cloud=cuda.to_device(cloud)
    d_neighbors=cuda.to_device(neighbors)
    d_dists_mat=cuda.to_device(dists_mat)
    d_weights=cuda.to_device(weights)
    d_out=cuda.device_array(nx,dtype=np.float32)
    thread=(TPB)
    blocks=(nx+TPB-1)//TPB
    mean_blur_kernel[blocks, thread](d_out,d_cloud, d_neighbors, d_dists_mat,d_weights)
    
    return d_out.copy_to_host()
    
@cuda.jit(device=True)
def dev_h_to_rgb(h):
    h=h%360
    x=(1-abs((h/60)%2-1))
    if(h<60):
        r,g,b=1,x,0
    elif(h<120):
        r,g,b=x,1,0
    elif(h<180):
        r,g,b=0,1,x
    elif(h<240):
        r,g,b=0,x,1
    elif(h<300):
        r,g,b=x,0,1
    else:
        r,g,b=1,0,x
        
    return r,g,b

@cuda.jit()
def color_map_kernel(d_color, d_dists, min_dist, max_dist):
    i=cuda.grid(1)
    nx=d_dists.shape[0]
    if i<nx:
        if d_dists[i]>max_dist:
            val=max_dist
        elif d_dists[i]<min_dist:
            val=min_dist
        else:
            val=d_dists[i]
        r,g,b=dev_h_to_rgb(((-240)/(max_dist-min_dist)*(val-min_dist)+240))
        d_color[i,0]=r
        d_color[i,1]=g
        d_color[i,2]=b
        
def color_map_par(dists, min_dist, max_dist):
    # valid_d=dists[np.isfinite(dists)]
    # min_dist, max_dist=np.percentile(valid_d,percentiles)
    nx=dists.shape[0]
    d_dists=cuda.to_device(dists)
    d_color=cuda.device_array((nx,3),dtype=np.float64)
    thread=(TPB)
    blocks=(nx+TPB-1)//TPB
    color_map_kernel[blocks, thread](d_color, d_dists, min_dist, max_dist)
    
    return d_color.copy_to_host()

def color_cloud(cloud, distance, min_dist, max_dist):
    new_cloud=copy.deepcopy(cloud)
    color=color_map_par(distance, min_dist, max_dist) 
    new_cloud.colors=o3d.utility.Vector3dVector(np.asarray(color))
    return new_cloud    

def calculate_discrep(cloud, target_tree ,target=[], cov_inv=[]):
    d ,closest_index_kd_cad=target_tree.query(cloud, k=1)
    if len(cov_inv)==0 or len(target)==0:
        print("using Euclidean dist")
        return d
    else:
        print("using Mahalanobis dist")
        md=np.zeros(cloud.shape[0])
        for i, pt in enumerate(cloud):
            error=np.asarray([pt-target[closest_index_kd_cad[i]]])
            md[i]=np.sqrt(error@cov_inv[closest_index_kd_cad[i]]@error.T)
            if np.isnan(md[i]):
                md[i]=np.inf
        return md

def calculate_dist(target_tree, cloud):
    d,_=target_tree.query(cloud, k=1)
    return d

def gaussian_blur(points, cloud, mdist):
	region_mdist=[]
	sigma=0.05		
	tree=o3d.geometry.KDTreeFlann(cloud)
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
    
def multi_blur(cloud,sample_points, distance, cloud_tree, num_iter, weights=[],k=150, method='mean'):
    if len(weights)==0:
        weights=np.ones(len(np.asarray(cloud.points)))
        
    if method=='mean':
        for i in range(num_iter):
            blur_dist=mean_blur_par(np.asarray(cloud.points), distance, cloud_tree, k, weights)
            distance=blur_dist
    else:
        for i in range(num_iter):
            blur_dist=gaussian_blur(sample_points,cloud ,distance)
    return blur_dist

class Local_Covariance_Trainer:
   
    @staticmethod
    def count_points_par(num_points, num_sample):
        return
    
    
    @staticmethod
    def remove_low_sample_points(samples,target, percentile, min_sample_num=2, k=1):
        target_tree=KDTree(target)
        _,closest_index_kd=target_tree.query(samples, k=k)
        closest_index_kd=closest_index_kd.flatten()
        num_target_pt=target.shape[0]
        #num_sample=np.asarray([np.count_nonzero(closest_index_kd == i) for i in range(num_target_pt)])
        num_sample=np.zeros(num_target_pt) 
        for i, index in enumerate(closest_index_kd):
            num_sample[index]+=1
        
        if percentile:
            min_sample_num=np.percentile(num_sample, percentile) 
            
        return np.where(num_sample>=min_sample_num)[0]
    
    @staticmethod
    def local_covariance(samples,target):
        target_tree=KDTree(target)
        _,closest_index_kd=target_tree.query(samples, k=1)
    
        num_target_pt=target.shape[0]
        num_sample=np.zeros(num_target_pt) 
        mean=np.zeros(target.shape)
        samples_pts= [ [] for _ in range(num_target_pt)]
        for i, index in enumerate(closest_index_kd):
            samples_pts[index]+=[i]
            num_sample[index]+=1
            
        covariance=np.zeros((num_target_pt,3,3))    
        for i, point in enumerate(samples_pts):
            if point:
                mean[i,:]=np.average(samples[point], axis=0)
            for sample in point:
                error=np.asarray([mean[i]-samples[sample]])
                covariance[i]+=error.T@error
                
        return mean, covariance,num_sample
    
    @staticmethod
    def covariance_smoothing(target,covariance, num_sample,voxel_ds, method, neighborhood, k=250,sigma=0.075):
        if not voxel_ds:
            sample=target
        else:
            sample=Cloud_from_points(target)
            sample=sample.voxel_down_sample(voxel_ds)
            sample=np.asarray(sample.points)
            
        if neighborhood=='nn':
            target_tree=KDTree(target)
            dist,closest_index_kd=target_tree.query(sample, k=250)
        else:
            tree=o3d.geometry.KDTreeFlann(Cloud_from_points(target))
            closest_index_kd=[]
            dist=[]
            for pt in sample:
                [_, idx ,d]=tree.search_radius_vector_3d(pt, sigma*4)
                closest_index_kd.append(np.asarray(idx, dtype=np.int32))
                dist.append(np.asarray(d))
            dist=np.asarray(dist)
        if method=='mean':
            for j in range(1):
                    cov_smooth=([(np.sum(covariance[closest_index_kd[i]],0))/(np.sum(num_sample[closest_index_kd[i]]))
                               if (np.sum(num_sample[closest_index_kd[i]]))!=0 else np.ones((3,3))*np.nan for i, pt in enumerate(sample)])
                    covariance=np.asarray(cov_smooth)
        else:
            for j in range(1):
                cov_smooth=[]
                for i in range(len(sample)):
                    n=np.asarray(num_sample[closest_index_kd[i]])
                    w=np.exp(-(dist[i]**2/sigma**2)/2)
                    v1=np.sum(w*n)
                    v2=np.sum(w**2*n)
                    if (v1-v2/v1)!=0:
                        mats=np.asarray([w[k]*n[k]*cov for k, cov in enumerate(covariance[closest_index_kd[i]])])
                        cov_smooth.append(1/(v1-v2/v1)*np.sum(mats,axis=0))
                    else:
                        cov_smooth.append(np.ones((3,3))*np.nan)
                covariance=cov_smooth
        
    
        if voxel_ds:
            sample_tree=KDTree(sample)
            _,closest_index_kd=sample_tree.query(target, k=1)
            covariance=[covariance[idx] for idx in closest_index_kd]
                    
        return covariance
    
    @staticmethod
    def learn_local_covariance(sample_clouds, target_cloud, k_smoothing=250, voxel_ds=0,regularize=0.1, method='mean',neighborhood='nn'):
        mean,covariance, num_sample=Local_Covariance_Trainer.local_covariance(sample_clouds,target_cloud)
        cov_smooth=Local_Covariance_Trainer.covariance_smoothing(target_cloud,covariance,num_sample,voxel_ds, method, neighborhood, k=k_smoothing)
        
        cov_smooth=np.asarray([cov+np.eye(3)*regularize for cov in cov_smooth])
        cov_inv=np.asarray([np.linalg.inv(cov) if np.linalg.det(cov)!=0 else np.matrix(np.ones((3,3)) * np.inf) for cov in cov_smooth])
        return mean, cov_inv, num_sample, cov_smooth