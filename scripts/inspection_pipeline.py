#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Feb  05 23:13:00 2021

@author: Benjamin
"""
import rospkg
import rospy 
from FOD_Detection import FOD_Detector
import tank_loop_nav as tl
from waypoint_generation import Waypoint_Generator
import greedy_scheduler as gs
import navigate_to_point as n2p
from std_srvs.srv import Empty
from nav_msgs.msg import OccupancyGrid
from waypoint_generation import parse_map_msg
import yaml

rospack=rospkg.RosPack()
navsea=rospack.get_path('navsea')

def loop_input():
	inp=""
	while inp.lower()!='done':
		inp=input("inspection pipeline started, enter \"done \" after initial teleoperation: \n")

if __name__ == "__main__":
    try:
        rospy.init_node('FOD_Detection',anonymous=True)
        #wait for teleoperation to be done
        loop_input()

		#Autonoumous SLAM phase
        print("Entering autonomous navigation module, please exit teleop")
		
        while True:
            try:
                num_loop = int(input("Please enter number of loop: "))
                break
            except ValueError:
                print("Not a valid number.  Try again...")
        if num_loop!=0:
            tl.tank_loop_routine(num_loop=num_loop)
        else:
            print("skip autonoumous phase")


        set_localize=rospy.ServiceProxy('rtabmap/set_mode_localization',Empty)
        set_localize()
        print("Rtabmap set to localization mode")

		#FOD Processing phase 
        print("Autonomous inspection complete, Entering processing module" )
        fod_detector=FOD_Detector()
        fod_detector.FOD_Detection_routine()

        tf=fod_detector.tf        
        obsticle_points=fod_detector.Project_obsticles()
        objectPoints=fod_detector.fod_centroids
        print(tf)
        #Picture generation 
        print("Number of FOD candidate found: "+str(len(objectPoints)))

        waypoints=[]
        print("Calculating waypoint location...")
        with open(navsea+"/param/waypoint_generation_params.yaml", 'r') as file:
            waypoint_params= yaml.safe_load(file)
        topic="/move_base/global_costmap/costmap"
        map_data = rospy.wait_for_message(topic, OccupancyGrid, timeout=5)
        origin, map_array=parse_map_msg(map_data)
        waypoint_generator=Waypoint_Generator(map_array, origin, tf,params=waypoint_params, verbose=False)
        
        for point in objectPoints:
            waypoints.append(waypoint_generator.generate_waypoint(point))
            print("waypoints: "+str(waypoints))
		
        numFOD=len(waypoints)
        waypoint_prev=[0,0,0,0]
        for i in range(numFOD):
            print("Navigating to FOD Candidate "+str(i+1)+"/"+str(numFOD))
            (idx,cloesetPoint)=gs.greedy_scheduler(waypoints, waypoint_prev)
            cloesetPoint=waypoints[idx]
            print(cloesetPoint)
            waypoint_prev=cloesetPoint
            del waypoints[idx]
            n2p.navigate2point(cloesetPoint)
            #snapshot.snapshot("FOD_candidate_"+str(i+1))
            save_im=rospy.ServiceProxy('image_saver/save',Empty)
            save_im()
            print("image saved")
    except KeyboardInterrupt:
        print("Terminating")
		
	
	
