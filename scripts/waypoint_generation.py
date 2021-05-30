#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 10:13:17 2021

@author: wade
"""
import rospkg
import rospy 
import pickle
import numpy as np
import random
from numpy import sqrt, sin, cos, nan
from math import atan2
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from nav_msgs.msg import OccupancyGrid
from sklearn.neighbors import NearestNeighbors

rospack=rospkg.RosPack()
navsea=rospack.get_path('navsea')
resolution=0.05

robot_radius = 0.25 # meters from object
min_distance = 0.75 # meters from object
max_distance = 1.5 # meters from object
cost_threshold = 70 # waypoint will not go to cost >= this
ray_sim_step = 0.01 # raytrace ray step

##==================== Map Topics ====================#:
    
def topic_to_array(map_topic, object_waypoint, resolution=resolution):

    w = map_topic.info.width
    map_array = []
    row_buffer = []
    first_row = False
    for i in range(len(map_topic.data)):
        row_buffer += [map_topic.data[i]]
        if i%(w-1) == 0 and i != 0 and first_row == False:
            map_array += [row_buffer]
            row_buffer = []  
            first_row = True
        elif (i+1)%w == 0 and first_row == True:
            map_array += [row_buffer]
            row_buffer = []
    map_array = np.array(map_array).transpose()  
    object_indicies = np.array([abs(round((map_topic.info.origin.position.x-object_waypoint[0])/resolution)),
                                abs(round((map_topic.info.origin.position.y-object_waypoint[1])/resolution))])
        
 
    return map_array, object_indicies.astype(np.int)
def project_tf(tf, points):
    #print(np.asarray(points).shape[1])
    if np.asarray(points).shape[1]==2:
    	newPoints=[np.append(point, [0,1]) for point in points]
    elif np.asarray(points).shape[1]==3:
        newPoints=[np.append(point, [1]) for point in points]
    elif np.asarray(points).shape[1]==4:
        newPoints=points
    newPoints=np.matmul(tf,np.transpose(newPoints))
    return np.transpose(newPoints)

def index2point(map_topic, waypoint_indicies, resolution): 
    points = [ np.append(index[0]*resolution + map_topic.info.origin.position.x, 
		index[1]*resolution + map_topic.info.origin.position.y) for index in waypoint_indicies]
    return np.asarray(points)

def point2index(map_topic, points, resolution):
    indices=[np.array([abs(round((map_topic.info.origin.position.x-point[0])/resolution)),
                                abs(round((map_topic.info.origin.position.y-point[1])/resolution))]) for point in points]
    return np.asarray(indices).astype(np.int)

def waypoint_indicies_to_topic(map_topic, object_point, waypoint_indicies, tf, resolution):
    point=index2point(map_topic, waypoint_indicies, resolution)
    tf_init=np.asarray([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    p=project_tf(tf, point)[0]
    x=p[0]
    y=p[1]

    p_obj=project_tf(tf, [object_point])[0]
    x_obj=p_obj[0]
    y_obj=p_obj[1]

    theta = atan2(y_obj-y, x_obj-x)
    z = np.sin(theta/2)
    w = np.cos(theta/2)
    return [x, y, w, z]

##==================== Waypoint Topics ====================#:

def generate_tree(costmap):
    costmap_points = []
    for i in range(costmap.shape[0]):
        for j in range(costmap.shape[1]):
            costmap_points += [[i,j]]
    costmap_points = np.array(costmap_points)
    costmap_tree = cKDTree(costmap_points)
    return costmap_tree

def generate_candidates(costmap, costmap_tree, point, resolution, min_dist, max_dist):
    min_dist = int(abs(round(min_dist/resolution)))
    max_dist = int(abs(round(max_dist/resolution)))
    neighborhood = (costmap_tree.data[costmap_tree.query_ball_point(point, max_dist)]).astype(np.int)
    candidates = []
    for neighbor in neighborhood:
        if sqrt((neighbor[0]-point[0])**2 + (neighbor[1]-point[1])**2) > min_dist:
            candidates += [[neighbor[0], neighbor[1]]]
    return np.array(candidates)

def filter_collision(costmap, costmap_tree, candidates, radius=robot_radius, cost_threshold=cost_threshold):
    radius = int(abs(round(radius/resolution)))
    new_candidates = []
    for candidate in candidates: 
        save_candidate = 1
        neighborhood = (costmap_tree.data[costmap_tree.query_ball_point(candidate, radius)]).astype(np.int)
        for neighbor in neighborhood:
            if costmap[neighbor[0], neighbor[1]] >= cost_threshold:
                save_candidate = 0
                break
        if save_candidate == 1:
            new_candidates += [[candidate[0], candidate[1]]]
    return np.array(new_candidates)

def filter_obsticles(candidates, obsticle_tree):
    distances, indices =obsticle_tree.kneighbors(candidates)
    #print(candidates)
    idx=np.where(distances>(robot_radius/resolution))
    candidates=candidates[idx[0]]
    return candidates

def filter_raytrace(costmap, candidates, point, step=ray_sim_step, costmap_resolution=resolution):
    new_candidates = []
    ray_costs = []
    for candidate in candidates:
        save_candidate = True
        r = sqrt((candidate[0] - point[0])**2 + (candidate[1] - point[1])**2)
        theta = atan2((point[1] - candidate[1]), (point[0] - candidate[0]))
        x_pos = candidate[0]
        y_pos = candidate[1]
        distance = 0
        ray_cost = 0
        while distance < r:
            x_pos += step*cos(theta)
            y_pos += step*sin(theta)
	    if  int(round(x_pos))>=costmap.shape[0] or int(round(y_pos))>=costmap.shape[1]:
		continue
  	    else:
            	ray_cost += costmap[int(round(x_pos)), int(round(y_pos))]
            	distance = sqrt((x_pos-candidate[0])**2 + (y_pos-candidate[1])**2)
            if costmap[int(round(x_pos)), int(round(y_pos))] >= np.inf or int(round(x_pos))>=costmap.shape[0] or int(round(y_pos))>=costmap.shape[1]:
                save_candidate = False
                break
        if save_candidate:
            new_candidates += [[candidate[0], candidate[1], theta]] # also store theta for later
            ray_costs += [ray_cost]
    return np.array(new_candidates), np.array(ray_costs)
    
def minimize_ray_cost(candidates, ray_costs):
    min_cost_index = ray_costs.argmin()
    return candidates[min_cost_index]

def generate(obsticle_points, cad_costmap_topic, object_point,tf_cad2cloud, tf_cloud2cad,max_dist, min_dist, robot_radius=robot_radius,plot=False):
    flag =  True # flag if failed to find waypoints

    cad_costmap, object_indicies = topic_to_array(cad_costmap_topic, object_point)
    obsticle_indicies=point2index(cad_costmap_topic, obsticle_points, resolution)
    costmap_tree = generate_tree(cad_costmap)
    obsticle_tree = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(obsticle_indicies)
    
    if plot:
        plt.imshow(cad_costmap)
        plt.plot(object_indicies[1],object_indicies[0],'o', color='black')
        plt.show()

    candidates = generate_candidates(cad_costmap, costmap_tree,object_indicies, resolution,max_dist=max_dist, min_dist=min_dist)
    if plot:
        plt.imshow(cad_costmap)
        plt.plot(object_indicies[1],object_indicies[0],'o', color='black')
        plt.plot(candidates[:,1], candidates[:,0], 'o', color='orange')
        plt.show()
    
    candidates = filter_collision(cad_costmap, costmap_tree, candidates)
	
    max_temp=max_dist
    while candidates.size == 0:
	print("No valid candidate, retrying with increased tolerance")
	max_temp=max_temp*1.1
   	candidates = generate_candidates(cad_costmap, costmap_tree,object_indicies, resolution, max_dist=max_temp)

    	if plot:
     	   plt.imshow(cad_costmap)
     	   plt.plot(object_indicies[1],object_indicies[0],'o', color='black')
    	   plt.plot(candidates[:,1], candidates[:,0], 'o', color='orange')
    	   plt.show()
    
        candidates = filter_collision(cad_costmap, costmap_tree, candidates)

 

    if plot:
        plt.imshow(cad_costmap)
        plt.plot(object_indicies[1],object_indicies[0],'o', color='black')
	plt.plot(obsticle_indicies[:,1],obsticle_indicies[:,0],'o', color='red')
        plt.plot(candidates[:,1], candidates[:,0], 'o', color='orange')
        plt.show()    

    candidates = filter_obsticles(candidates, obsticle_tree)
 
    if candidates.size == 0:
        print('waypoint_generation: Error: could not generate point, retry with increased tolerance')
        return nan, flag
    #print(len(candidates))
    if len(candidates)>250:
	idx=random.sample(range(0,len(candidates)), 250)
	candidates=candidates[idx]


    #print(candidates)
    if plot:
        plt.imshow(cad_costmap)
        plt.plot(object_indicies[1],object_indicies[0],'o', color='black')
	plt.plot(obsticle_indicies[:,1],obsticle_indicies[:,0],'o', color='red')
        plt.plot(candidates[:,1], candidates[:,0], 'o', color='orange')
        plt.show()

    candidates, ray_costs = filter_raytrace(cad_costmap, candidates, object_indicies)  
    if candidates.size == 0:
        print('waypoint_generation: Error: could not generate point, retry with increased tolerance')
        return nan, flag

   # print(len(candidates))
    candidates=np.delete(candidates, np.where((candidates[:,0]>cad_costmap.shape[0]) | (candidates[:,1]>cad_costmap.shape[1])), axis=0)
 
  # print(candidates)    
    if candidates.size == 0:
        print('waypoint_generation: Error: could not generate point, retry with increased tolerance')
        return nan, flag
	
    if False:
        plt.imshow(cad_costmap)
        plt.plot(object_indicies[1],object_indicies[0],'o', color='black')
	plt.plot(obsticle_indicies[:,1],obsticle_indicies[:,0],'o', color='red')
        plt.plot(candidates[:,1], candidates[:,0], 'o', color='orange')
        plt.show()
    
    waypoint = minimize_ray_cost(candidates, ray_costs)

    if plot:
        plt.imshow(cad_costmap)
	plt.plot(obsticle_indicies[:,1],obsticle_indicies[:,0],'o', color='red')
        plt.plot(object_indicies[1],object_indicies[0],'o', color='black')
        plt.plot(waypoint[1], waypoint[0], 'o', color='orange')
        plt.show()
    flag=False
    return waypoint, flag

def generate_waypoint(object_point,obsticle_points, tf_cad2cloud, tf_cloud2cad):
    pkl_filename = "cad_costmap.pkl"
    with open(navsea+"/scripts/"+pkl_filename, 'rb') as file:
        cad_costmap_topic = pickle.load(file) # replace to be recieved from topic   
    max_dist=max_distance
    min_dist=min_distance
    flag=True
    plot=False
    while flag:
    	waypoint_indicies, flag = generate(obsticle_points, cad_costmap_topic, object_point, tf_cad2cloud, tf_cloud2cad, max_dist=max_dist,min_dist=min_dist, plot=plot)
	max_dist=max_dist*1.1
	min_dist=min_dist*0.9

	plot=True
    
    if not flag:
        waypoint = waypoint_indicies_to_topic(cad_costmap_topic,object_point, [waypoint_indicies], tf_cad2cloud, resolution)
        print(waypoint)
        return waypoint
    else:
        print('waypoint_generation: returning nan for failed point')
        return waypoint_indicies
    
if __name__ == "__main__": 
    test_points=[[ 0.26146433, -2.37086167,  0.49230829],
 	[ 0.60216947, -0.2426275,   0.34959391],
 	[ 1.32999525, -1.53629,     0.16368696],
	 [ 2.4402498,  -1.32006528,  0.06624348],
	 [ 5.20872064, -0.20594317,  0.41271756]]
    for test_point in test_points:
         test_point = np.array(test_point)
    
         tf_cloud2cad=[[-9.99973519e-01, -7.20862060e-03, -9.98112829e-04,  2.60570437e-02],
 [ 7.20890875e-03, -9.99973975e-01, -2.85397093e-04,  1.40439794e-02],
 [-9.96029533e-04, -2.92584840e-04,  9.99999461e-01, -5.38154708e-03],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]



         tf_cad2cloud=[[-9.99973519e-01,  7.20890875e-03, -9.96029533e-04, 2.59497517e-02],
 [-7.20862060e-03, -9.99973975e-01, -2.92584840e-04,  1.42298747e-02],
 [-9.98112829e-04, -2.85397093e-04,  9.99999461e-01,  5.41156016e-03],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]


         generate_waypoint(test_point,tf_cad2cloud,tf_cloud2cad )
