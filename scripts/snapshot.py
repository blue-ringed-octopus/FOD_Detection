#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Feb  05 23:13:00 2021

@author: Benjamin
"""
import ros_numpy
import rospy 
import rospkg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

topic="/camera/color/image_raw"
#'/stereo_camera/left/image_rect_color'
rospack=rospkg.RosPack()
navsea=rospack.get_path('navsea')
def get_message():
	try:

		data=rospy.wait_for_message(topic,Image)
		return data 
	except rospy.ServiceException as e:
		print("Service all failed: %s"%e)

def snapshot(name):
	data=get_message()
	bridge=CvBridge()
	cv_image=bridge.imgmsg_to_cv2(data,"bgr8")
	#cv2.imshow("image window", cv_image)
	#cv2.waitKey()
	filename=navsea+"/output/candidate_picture/"+str(name)+".png"
	cv2.imwrite(filename,cv_image)
	print("Image saved to: "+str(filename))
if __name__ == "__main__":
	rospy.init_node('Snapshot',anonymous=True)
	snapshot("test")

	
