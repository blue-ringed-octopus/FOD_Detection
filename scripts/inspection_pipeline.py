#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Feb  05 23:13:00 2021

@author: Benjamin
"""
import rospy 
from FOD_Detection import FOD_Detector
import numpy as np
import tank_loop_nav as tl
import waypoint_generation as wg
import greedy_scheduler as gs
import snapshot
import navigate_to_point as n2p
from std_srvs.srv import Empty

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
        tf_inv=np.linalg.inv(tf)
        obsticle_points=fod_detector.Project_obsticles()
        objectPoints=fod_detector.fod_centroids
        print(tf)
        #Picture generation 
        print("Number of FOD candidate found: "+str(len(fod_detector.objectPoints)))

        waypoints=[]
        print("Calculating waypoint location...")
        for point in objectPoints:
            waypoints.append(wg.generate_waypoint(point,obsticle_points,tf_inv, tf))
            print("waypoints: "+str(waypoints))
		
        numFOD=len(waypoints)

        for i in range(numFOD):
            print("Navigating to FOD Candidate "+str(i+1)+"/"+str(numFOD))
            (idx,cloesetPoint)=gs.greedy_scheduler(waypoints)
            print(cloesetPoint)
            print(waypoints[idx])
            del waypoints[idx]
            n2p.navigate2point(cloesetPoint)
            snapshot.snapshot("FOD_candidate_"+str(i+1))
    except KeyboardInterrupt:
        print("Terminating")
		
	
	
