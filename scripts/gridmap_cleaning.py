#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Note: unfixed bug swaps x and y in parts of code
"""


import pickle
import numpy as np
from time import time
#from numba import cuda
from math import pi, cos, sin, floor

import cv2
from scipy import ndimage

import rospy 
from nav_msgs.msg import OccupancyGrid

resolution = 0.01
kernel_size=(5,5)
# Initialize messages to publish:
map_pub = rospy.Publisher('/map', OccupancyGrid, queue_size=1)

##==================== Map messages ====================#:
    
def message_to_array(message):
    global resolution
    w = message.info.width
    h = message.info.height
    map_array = []
    row_buffer = []
    first_row = False
    for i in range(len(message.data)):
        row_buffer += [message.data[i]]
        if i%(w-1) == 0 and i != 0 and first_row == False:
            map_array += [row_buffer]
            row_buffer = []  
            first_row = True
        elif (i+1)%w == 0 and first_row == True:
            map_array += [row_buffer]
            row_buffer = []
    map_array = np.array(map_array).transpose()
    origin = [abs(round(message.info.origin.position.x/resolution)), abs(round(message.info.origin.position.y/resolution))]    
    wh = [w,h]
    return map_array, origin, wh

def array_to_image(map_array, prob_threshold=100):
    m = map_array.shape[0]
    n = map_array.shape[1]
    img_array = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            if map_array[i,j] >= prob_threshold:
                img_array[i,j] = map_array[i,j]            
            else:
                img_array[i,j] = 0
    img_array = (-(img_array-100) * 255/100).astype('uint8')
    img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    return np.array(img_array[:,:,1])


##==================== Map Publish ====================#:
    
def publish_map_img(map_img, map_origin, prev_message):
    #reuse parts of old message (e.g. header) - replace later
    global resolution
    
    message = prev_message
    message.info.width = int(round(map_img.shape[0]))
    message.info.height = int(round(map_img.shape[1]))
   # message.data= np.ndarray.flatten(np.transpose((((map_img/255*100)-100)*-1).astype(int)))
    message.data= (np.ndarray.flatten(np.transpose((((map_img/255*100)-100)*-1))).astype(np.int8)).tolist()
 #   print(arr.shape)
    #message.data=arr.tolist()
 #   print(message.data.shape)
#    message.info.origin.position.x = -map_origin[1]*resolution
#    message.info.origin.position.y = -map_origin[0]*resolution
    map_pub.publish(message)
    return

##==================== Parallel Search ====================#:

def map_merge_parallel(rmap_message):
    # turn to image
    rmap_array, rmap_origin, rmap_wh = message_to_array(rmap_message)
    rmap = array_to_image(rmap_array)
    # clean rtabmap map
    noise_ksize = kernel_size
    rmap = cv2.blur(rmap, noise_ksize)
    _, rmap = cv2.threshold(rmap,127,255,cv2.THRESH_BINARY)
    publish_map_img(rmap, rmap_origin, rmap_message)
        
    return

if __name__ == "__main__": 
    print('Cleaning map. Press Ctrl+C to exit.')
    try:
        rospy.init_node('map_convolution', anonymous=True)
        sub = rospy.Subscriber('/rtabmap_map', OccupancyGrid, map_merge_parallel)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
