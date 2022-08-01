#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 10:13:17 2021

@author: wade
"""
import numpy as np
import random
from numpy import sqrt, sin, cos, nan
from math import atan2
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
import yaml
from numba import cuda
import math

def parse_map_msg(map_msg):
    w = map_msg.info.width
    map_array = []
    row_buffer = []
    first_row = False
    for i in range(len(map_msg.data)):
        row_buffer += [map_msg.data[i]]
        if i%(w-1) == 0 and i != 0 and first_row == False:
            map_array += [row_buffer]
            row_buffer = []  
            first_row = True
        elif (i+1)%w == 0 and first_row == True:
            map_array += [row_buffer]
            row_buffer = []
    map_array = np.array(map_array).transpose()  
    
    origin=np.asarray([map_msg.info.origin.position.x, map_msg.info.origin.position.y])
    costmap=map_array
    
    return origin, costmap
    
class Waypoint_Generator: 
    '''
    Contructor
    '''
    def __init__(self,costmap, origin, tf,params, obstacle_points=[], verbose=False):
        self.tf_m2r=tf
        self.tf_r2m=np.linalg.inv(tf)
        self.robot_radius=params["robot_radius"]
        self.resolution=params["resolution"]
        self.min_distance_base=params["min_distance"]
        self.max_distance_base=params["max_distance"]
        self.obstacle_points=obstacle_points
        self.cost_threshold=params["cost_threshold"]
        self.ray_sim_step=params["ray_sim_step"]
        self.costmap=costmap
        self.origin=origin
        self.generate_tree()
        self.verbose=verbose
        self.plot=verbose
    
    
    
    def project_tf(self, tf, points):
        #print(np.asarray(points).shape[1])
        if np.asarray(points).shape[1]==2:
        	newPoints=[np.append(point, [0,1]) for point in points]
        elif np.asarray(points).shape[1]==3:
            newPoints=[np.append(point, [1]) for point in points]
        elif np.asarray(points).shape[1]==4:
            newPoints=points
        newPoints=np.matmul(tf,np.transpose(newPoints))
        return np.transpose(newPoints)
    
    def index2point(self, waypoint_indicies): 
        points = [ np.append(index[0]*self.resolution + self.origin[0], 
    		index[1]*self.resolution + self.origin[1]) for index in waypoint_indicies]
        return np.asarray(points)
    
    def point2index(self, point):
        index=np.array([abs(round((self.origin[0]-point[0])/self.resolution)),
                                    abs(round((self.origin[1]-point[1])/self.resolution))])
        return np.asarray(index).astype(int)
    
    def waypoint_indicies_to_msg(self, object_point, waypoint_indicies):
        point=self.index2point(waypoint_indicies)
        p=self.project_tf(self.tf_r2m, point)[0]
        x=p[0]
        y=p[1]
    
        p_obj=self.project_tf(self.tf_r2m, [object_point])[0]
        x_obj=p_obj[0]
        y_obj=p_obj[1]
    
        theta = atan2(y_obj-y, x_obj-x)
        w = np.cos(theta/2)
        z = np.sin(theta/2)
        
        return [x, y, w, z]
    
    ##==================== Waypoint msgs ====================#:
    
    def generate_tree(self):
        costmap=self.costmap
        costmap_points = []
        for i in range(costmap.shape[0]):
            for j in range(costmap.shape[1]):
                costmap_points += [[i,j]]
        costmap_points = np.array(costmap_points)
        self.costmap_tree = cKDTree(costmap_points)
    
    def generate_candidates(self, point):
        min_dist = int(abs(round(self.min_distance/self.resolution)))
        max_dist = int(abs(round(self.max_distance/self.resolution)))
        neighborhood = (self.costmap_tree.data[self.costmap_tree.query_ball_point(point, max_dist)]).astype(int)
        candidates = []
        for neighbor in neighborhood:
            if sqrt((neighbor[0]-point[0])**2 + (neighbor[1]-point[1])**2) > min_dist:
                candidates += [[neighbor[0], neighbor[1]]]
        idx=random.sample(range(0,len(candidates)), int(len(candidates)/5))
        candidates=np.array(candidates)[idx]
        return candidates
    
    def filter_collision(self, candidates):
        costmap=self.costmap
        costmap_tree=self.costmap_tree
        cost_threshold=self.cost_threshold
        radius = int(abs(round(self.robot_radius/self.resolution)))
        new_candidates = []
        for candidate in candidates: 
            save_candidate = 1
            neighborhood = (costmap_tree.data[costmap_tree.query_ball_point(candidate, radius)]).astype(int)
            for neighbor in neighborhood:
                if costmap[neighbor[0], neighbor[1]] >= cost_threshold:
                    save_candidate = 0
                    break
            if save_candidate == 1:
                new_candidates += [[candidate[0], candidate[1]]]
        return np.array(new_candidates)
    
    def filter_obstacles (self, candidates, obstacle_tree):
        distances, indices =obstacle_tree.kneighbors(candidates)
        #print(candidates)
        idx=np.where(distances>(self.robot_radius/self.resolution))
        candidates=candidates[idx[0]]
        return candidates
    
    @staticmethod
    @cuda.jit
    def filter_ray_trace_kernel(d_out, d_costmap, d_candidates, d_point, step_size):
        i=cuda.grid(1)
        save_candidate=True
        if i < d_out.shape[0]:
            d_out[i]=0
            candidate=d_candidates[i]
            r = math.sqrt((candidate[0] - d_point[0])**2 + (candidate[1] - d_point[1])**2)
            theta = atan2((d_point[1] - candidate[1]), (d_point[0] - candidate[0]))
            x_pos = candidate[0]
            y_pos = candidate[1]
            distance = 0
            ray_cost = 0
            while distance < r:
                x_pos += step_size*math.cos(theta)
                y_pos += step_size*math.sin(theta)
                if int(round(x_pos))>=d_costmap.shape[0] or int(round(y_pos))>=d_costmap.shape[1]:
                    continue
                else:
                    ray_cost += d_costmap[int(round(x_pos)), int(round(y_pos))]
                    distance = math.sqrt((x_pos-candidate[0])**2 + (y_pos-candidate[1])**2)
                    if d_costmap[int(round(x_pos)), int(round(y_pos))] >= np.inf or int(round(x_pos))>=d_costmap.shape[0] or int(round(y_pos))>=d_costmap.shape[1]:
                        save_candidate = False
                        break
            if save_candidate:
                d_out[i] += ray_cost
            else:
                d_out[i] =-1
        
    def filter_raytrace_par(self, candidates, point):
        nx=candidates.shape[0]
        d_candidates=cuda.to_device(candidates)
        d_point=cuda.to_device(point)
        d_out=cuda.device_array(nx,dtype=np.float32)
        d_costmap=cuda.to_device(self.costmap)
        TPB=128
        threads = (TPB)
        blocks=(nx+TPB-1)//TPB
        self.filter_ray_trace_kernel[blocks, threads](d_out,d_costmap, d_candidates, d_point,self.ray_sim_step)
        cost=d_out.copy_to_host()
        return candidates[cost>=0], cost[cost>=0]
    
    def filter_raytrace(self, candidates, point):
        costmap=self.costmap
        step=self.ray_sim_step
        new_candidates = []
        ray_costs = []
        for candidate in candidates:
            if self.verbose:
                print("candidate", candidate)
            save_candidate=True
            r = sqrt((candidate[0] - point[0])**2 + (candidate[1] - point[1])**2)
            theta = atan2((point[1] - candidate[1]), (point[0] - candidate[0]))
            x_pos = candidate[0]
            y_pos = candidate[1]
            distance = 0
            ray_cost = 0
            while distance < r:
                x_pos += step*cos(theta)
                y_pos += step*sin(theta)
                if int(round(x_pos))>=costmap.shape[0] or int(round(y_pos))>=costmap.shape[1]:
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
        
    def minimize_ray_cost(self, candidates, ray_costs):
        min_cost_index = ray_costs.argmin()
        return candidates[min_cost_index]
    
    def generate(self, object_point):
        plot=self.plot
        flag =  True # flag if failed to find waypoints
        verbose=self.verbose
        cad_costmap=self.costmap
        if verbose:
            print("object point",object_point)
        object_index=self.point2index(object_point)
        costmap_tree=self.costmap_tree
        obstacle_points=self.obstacle_points
        if not len(obstacle_points)==0:
            obstacle_indicies=[self.point2index(pt) for pt in obstacle_points]
            obstacle_tree = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(obstacle_indicies)
            
        if plot:
            plt.imshow(cad_costmap)
            plt.plot(object_index[1],object_index[0],'o', color='black')
            plt.show()
        if verbose:
            print("generating candidates")
        candidates = self.generate_candidates(object_index)
        if plot:
            plt.imshow(cad_costmap)
            plt.plot(object_index[1],object_index[0],'o', color='black')
            plt.plot(candidates[:,1], candidates[:,0], 'o', color='orange')
            plt.show()
        if verbose:
            print("filtering collision candidates")
        candidates = self.filter_collision(candidates)
    	
        while candidates.size == 0:
          print("No valid candidate, retrying with increased tolerance")
          self.max_distance_og=self.max_distance
          self.max_distance=self.max_distance*1.1
          candidates = self.generate_candidates(object_index)
    
          if plot:
             plt.imshow(cad_costmap)
             plt.plot(object_index[1],object_index[0],'o', color='black')
             plt.plot(candidates[:,1], candidates[:,0], 'o', color='orange')
             plt.show()
        
             candidates = self.filter_collision(candidates)
    
     
    
        if plot:
            plt.imshow(cad_costmap)
            plt.plot(object_index[1],object_index[0],'o', color='black')
            plt.plot(candidates[:,1], candidates[:,0], 'o', color='orange')
            plt.show()    
        if not len(obstacle_points)==0:
            candidates = self.filter_obstacles (candidates)
     
        if candidates.size == 0:
            print('waypoint_generation: Error: could not generate point, retry with increased tolerance')
            return nan, flag
        #print(len(candidates))
        if verbose:
            print("reduce candidates counts")
        if len(candidates)>100:
            idx=random.sample(range(0,len(candidates)), 100)
            candidates=candidates[idx]
    
    
        #print(candidates)
        if plot:
            plt.imshow(cad_costmap)
            plt.plot(object_index[1],object_index[0],'o', color='black')
        #    plt.plot(obstacle_indicies[:,1],obstacle_indicies[:,0],'o', color='red')
            plt.plot(candidates[:,1], candidates[:,0], 'o', color='orange')
            plt.show()
        if verbose:
            print("raytracing candidates")
        candidates, ray_costs = self.filter_raytrace_par(candidates, object_index)  
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
            plt.plot(object_index[1],object_index[0],'o', color='black')
          #  plt.plot(obstacle_indicies[:,1],obstacle_indicies[:,0],'o', color='red')
            plt.plot(candidates[:,1], candidates[:,0], 'o', color='orange')
            plt.show()
        if verbose:
            print("optimize ray cost")
        waypoint = self.minimize_ray_cost(candidates, ray_costs)
    
        if plot:
            plt.imshow(cad_costmap)
     #       plt.plot(obstacle_indicies[:,1],obstacle_indicies[:,0],'o', color='red')
            plt.plot(object_index[1],object_index[0],'o', color='black')
            plt.plot(waypoint[1], waypoint[0], 'o', color='orange')
            plt.show()
        flag=False
        return waypoint, flag
    
    def generate_waypoint(self, object_point):
        self.max_distance=self.max_distance_base
        self.min_distance=self.min_distance_base

        flag=True
        while flag:
            waypoint_indicies, flag = self.generate(object_point)
            self.max_distance=self.max_distance*1.1
            self.min_distance=self.min_distance*0.9
            
        if not flag:
            waypoint = self.waypoint_indicies_to_msg(object_point, [waypoint_indicies])
            print(waypoint)
            return waypoint
        else:
            print('waypoint_generation: returning nan for failed point')
            return waypoint_indicies
    
if __name__ == "__main__": 
    import rospy 
    import rospkg
    from nav_msgs.msg import OccupancyGrid

    topic="/move_base/global_costmap/costmap"
    rospy.init_node('FOD_Detection',anonymous=False)
    map_data = rospy.wait_for_message(topic, OccupancyGrid, timeout=5)
    origin, map_array=parse_map_msg(map_data)

    tf=np.eye(4)
    rospack=rospkg.RosPack()
    navsea=rospack.get_path('navsea')
    with open(navsea+"/param/waypoint_generation_params.yaml", 'r') as file:
        params= yaml.safe_load(file)
    test_points=[[ 1.5, -0.5,  0]]
    waypoint_generator=Waypoint_Generator(map_array , origin, tf, params,verbose=True)

    for test_point in test_points:
        test_point = np.array(test_point)
        waypoint=waypoint_generator.generate_waypoint(test_point)
    import navigate_to_point as n2p
    n2p.navigate2point(waypoint)