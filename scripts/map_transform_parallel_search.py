#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Note: unfixed bug swaps x and y in parts of code
"""


import pickle
import numpy as np
from time import time
from numba import cuda
from math import pi, cos, sin, floor
import  matplotlib.pyplot as plt
import cv2
from scipy import ndimage

import rospy 
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Float64



print('map_transform_parallel_search: Starting')

resolution = 0.05

# Initialize topics to publish:

map_transform_pub = rospy.Publisher('/map', OccupancyGrid, queue_size=1)

##==================== Map Topics ====================#:
    
def topic_to_array(topic):
    global resolution
    w = topic.info.width
    h = topic.info.height
    map_array = []
    row_buffer = []
    first_row = False
    for i in range(len(topic.data)):
        row_buffer += [topic.data[i]]
        if i%(w-1) == 0 and i != 0 and first_row == False:
            map_array += [row_buffer]
            row_buffer = []  
            first_row = True
        elif (i+1)%w == 0 and first_row == True:
            map_array += [row_buffer]
            row_buffer = []
    map_array = np.array(map_array).transpose()
    origin = [abs(round(topic.info.origin.position.x/resolution)), abs(round(topic.info.origin.position.y/resolution))]    
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

#@cuda.jit('void(uint8[:,:], uint8[:,:], int32, int32, int32, int32, int64, int64, float64, float64[:,:])')
@cuda.jit
def transform_kernel(default_map, rtabmap_map, nx, ny, rbx, rby, tx, ty, theta, output_map):
    i,j = cuda.grid(2) 

    if i < nx and j < ny:
        rx =  int(round(i*cos(theta) - j*sin(theta) + tx))
        ry  = int(round(i*sin(theta) + j*cos(theta) + ty))
        if rx >= 0 and ry >= 0 and rx < rbx and ry < rby:
            output_map[i,j] = rtabmap_map[rx,ry]
        else:
            output_map[i,j] = -1


def transform_parallel(default_map, rtabmap_map, rtabmap_origin, tx, ty, theta):
    TPBX, TPBY = 32, 32
    nx = default_map.shape[0]
    ny = default_map.shape[1]
    rbx = rtabmap_map.shape[0]
    rby = rtabmap_map.shape[1]
    d_default_map = cuda.to_device(default_map)
    d_rtabmap_map = cuda.to_device(rtabmap_map)
    d_output_map = cuda.device_array(shape = [nx, ny], dtype = np.float64)
    grid_dims = (nx+TPBX-1)//TPBX, (ny+TPBY-1)//TPBY
    block_dims = TPBX, TPBY
    transform_kernel[grid_dims, block_dims](d_default_map, d_rtabmap_map, nx, ny, rbx, rby, tx, ty, theta, d_output_map)
    output_map = d_output_map.copy_to_host()
    output_map = ndimage.rotate(output_map.astype(np.uint8), theta*180/pi, reshape=True, cval=255)

    # transform origin:
    xy = np.array([(rtabmap_origin[0]-tx)*cos(theta) + (rtabmap_origin[1]-ty)*sin(theta),
                   -(rtabmap_origin[0]-tx)*sin(theta) + (rtabmap_origin[1]-ty)*cos(theta)])
    org_center = np.array([default_map.shape[0]/2, default_map.shape[1]/2])
    rot_center = np.array([output_map.shape[0]/2, output_map.shape[1]/2])
    org = xy-org_center
    new = np.array([org[0]*np.cos(theta) - org[1]*np.sin(theta),
                    org[0]*np.sin(theta) + org[1]*np.cos(theta) ])
    output_origin = new+rot_center
    return output_map, [int(output_origin[1]), int(output_origin[0])]

def publish_map_img(map_img, map_origin, prev_topic):
    #reuse parts of old topic (e.g. header) - replace later
    global resolution
    
    topic = prev_topic
    topic.info.width = int(round(map_img.shape[0]))
    topic.info.height = int(round(map_img.shape[1]))
    topic.data = np.ndarray.flatten(np.transpose((((map_img/255*100)-100)*-1).astype(np.int)))
    topic.info.origin.position.x = -map_origin[1]*resolution
    topic.info.origin.position.y = -map_origin[0]*resolution
    map_transform_pub.publish(topic)
    return

##==================== Parallel Search ====================#:

def reward_template(map_img, downsample_factor):
    map_occupied_i = []
    map_occupied_j = []
    for i in range(floor(map_img.shape[0]/downsample_factor)):
        for j in range(floor(map_img.shape[1]/downsample_factor)):            
                if map_img[downsample_factor*i,downsample_factor*j] == 0:
                    map_occupied_i += [downsample_factor*i]
                    map_occupied_j += [downsample_factor*j]
    return np.array([map_occupied_i,map_occupied_j])

#@cuda.jit('void(uint8[:,:], uint8[:,:], uint16[:,:], int32, int32, int32, int32, int32, int32, int32, int32, float64[:,:,:])')
@cuda.jit
def reward_kernel_rough(default_map, rtabmap_map, template, dx, dy, rx, ry, nx, ny, nz, downsample_factor, rewards):
    i,j,k = cuda.grid(3)

    reward = 0
    if i < nx and j < ny and k < nz:
        theta = (k/nz)*2*pi
        for s in range(template.shape[1]):
            p = template[0,s]  
            q = template[1,s]
            p_new = int(round(p*cos(theta) - q*sin(theta) + (i-dx-rx)*downsample_factor))
            q_new = int(round(p*sin(theta) + q*cos(theta) + (j-dy-ry)*downsample_factor))
            # check positive rtabmap index within rtabmap image
            if p_new > 0 and q_new > 0 and p_new < downsample_factor*rx and q_new < downsample_factor*ry:
                if default_map[p,q] == rtabmap_map[p_new,q_new]:
                    reward += 1
        rewards[i,j,k] = reward

def parallel_search_rough(default_map, rtabmap_map, template, nz, downsample_factor):
    dx = int(round(default_map.shape[0] / downsample_factor))
    dy = int(round(default_map.shape[1] / downsample_factor))
    rx = int(round(rtabmap_map.shape[0] / downsample_factor))
    ry = int(round(rtabmap_map.shape[1] / downsample_factor))
    nx = 2*(dx+rx)
    ny = 2*(dy+ry)
    
    TPB = 8
    threads = TPB, TPB, TPB
    blocks = (nx+TPB-1)//TPB, (ny+TPB-1)//TPB, (nz+TPB-1)//TPB
    
    default_map = cuda.to_device(default_map)
    rtabmap_map = cuda.to_device(rtabmap_map)
    template = cuda.to_device(template)
    rewards = cuda.device_array(shape=[nx, ny, nz], dtype=np.float64)
    
    reward_kernel_rough[blocks, threads](default_map, rtabmap_map, template, dx, dy, rx, ry, nx, ny, nz, downsample_factor, rewards)

    rewards = rewards.copy_to_host()
    transformation = np.unravel_index(rewards.argmax(),rewards.shape)

    tx = (transformation[0] - dx-rx)*downsample_factor
    ty = (transformation[1] - dy-ry)*downsample_factor
    theta = (transformation[2]/nz)*2*pi
    reward = rewards[transformation[0], transformation[1], transformation[2]]
    return tx, ty, theta, reward

#@cuda.jit('void(uint8[:,:], uint8[:,:], uint16[:,:], int32, int32, int32, int32, int32, int32, int32, int32, int32, int32, float64, float64[:,:,:])')
@cuda.jit
def reward_kernel_fine(default_map, rtabmap_map, template, dx, dy, dz, rx, ry, nx, ny, nz, tx, ty, theta, rewards):
    i,j,k = cuda.grid(3)
    
    reward = 0
    if i < nx and j < ny and k < nz:
        theta = (theta + (k-dz)*pi/180)
        for s in range(template.shape[1]):
            p = template[0,s]  
            q = template[1,s]
            p_new = int(round(p*cos(theta) - q*sin(theta) + (tx+i-dx)))
            q_new = int(round(p*sin(theta) + q*cos(theta) + (ty+j-dy)))
            # check positive rtabmap index within rtabmap image
            if p_new > 0 and q_new > 0 and p_new < rx and q_new < ry:
                if default_map[p,q] == rtabmap_map[p_new,q_new]:
                    reward += 1
        rewards[i,j,k] = reward

def parallel_search_fine(default_map, rtabmap_map, template, dx, dy, dz, tx, ty, theta):
    
    nx = 2*dx
    ny = 2*dy
    nz = 2*dz
    
    rx = rtabmap_map.shape[0]
    ry = rtabmap_map.shape[1]
    
    default_map = cuda.to_device(default_map)
    rtabmap_map = cuda.to_device(rtabmap_map)
    rewards = cuda.device_array(shape=[nx, ny, nz], dtype=np.float64)

    TPB = 8
    threads = TPB, TPB, TPB
    blocks = (nx+TPB-1)//TPB, (ny+TPB-1)//TPB, (nz+TPB-1)//TPB

    reward_kernel_fine[blocks, threads](default_map, rtabmap_map, template, dx, dy, dz, rx, ry, nx, ny, nz, tx, ty, theta, rewards)

    rewards = rewards.copy_to_host()
    transform_delta = np.unravel_index(rewards.argmax(),rewards.shape)
    
    tx += transform_delta[0]-dx
    ty += transform_delta[1]-dy
    theta += (transform_delta[2]-dz)*pi/180
    reward = rewards[transform_delta[0], transform_delta[1], transform_delta[2]]
    return tx, ty, theta, reward 

def map_merge_parallel(rmap_topic):
    global dmap
    
    t0 = time()
    rmap_array, rmap_origin, rmap_wh = topic_to_array(rmap_topic)
    rmap = array_to_image(rmap_array)
    # compute reward template
    rough_template = reward_template(dmap, downsample_factor=2) # downsampling template proportional to computation time
    fine_template = reward_template(dmap, downsample_factor=1)
    # clean rtabmap map
    noise_ksize = (1,1)
    rmap = cv2.blur(rmap, noise_ksize)
    _, rmap = cv2.threshold(rmap,127,255,cv2.THRESH_BINARY)
    # calculate coarse transformation 
    nz = 72
    tx, ty, theta, reward = parallel_search_rough(dmap, rmap, rough_template, nz, downsample_factor=5) # 3
    # calculate fine transformation
    dx = dy = 5
    dz = 5
    tx, ty, theta, reward = parallel_search_fine(dmap, rmap, fine_template, dx, dy, dz, tx, ty, theta)
    # apply transformation and publish

    print('gridmap_parallel_search: Transformation found in ', round((time()-t0)*1000),' ms')
    return

def map_transform_parallel(rmap_topic):
    global dmap
    global initialized_transform_flag
    global tx_transform
    global ty_transform
    global theta_transform
    
    if not initialized_transform_flag:
        t0 = time()
        rmap_array, rmap_origin, rmap_wh = topic_to_array(rmap_topic)
        rmap = array_to_image(rmap_array)
        # compute reward template
        rough_template = reward_template(dmap, downsample_factor=2) # downsampling template proportional to computation time
        fine_template = reward_template(dmap, downsample_factor=1)
        # clean rtabmap map
        noise_ksize = (1,1)
        rmap = cv2.blur(rmap, noise_ksize)
        _, rmap = cv2.threshold(rmap,127,255,cv2.THRESH_BINARY)
        # calculate coarse transformation 
        nz = 72
        tx, ty, theta, reward = parallel_search_rough(dmap, rmap, rough_template, nz, downsample_factor=5) # 3
        # calculate fine transformation
        dx = dy = 5
        dz = 5
        tx, ty, theta, reward = parallel_search_fine(dmap, rmap, fine_template, dx, dy, dz, tx, ty, theta)
        output_map, output_origin = transform_parallel(dmap, rmap, rmap_origin, tx, ty, theta)
        print('map_transform_parallel_search: Transformation found in ', round((time()-t0)*1000),' ms')
        print('map_transform_parallel_search: Publishing transformed map topic /map for navigation')
        print('map_transform_parallel_search: Restart to find new transformation')
        publish_map_img(output_map, output_origin, rmap_topic)
        tx_transform = tx
        ty_transform = ty
        theta_transform = theta
        initialized_transform_flag = True
    else:
        rmap_array, rmap_origin, rmap_wh = topic_to_array(rmap_topic)
        rmap = array_to_image(rmap_array)
        # compute reward template
        rough_template = reward_template(dmap, downsample_factor=2) # downsampling template proportional to computation time
        fine_template = reward_template(dmap, downsample_factor=1)
        # clean rtabmap map
        noise_ksize = (1,1)
        rmap = cv2.blur(rmap, noise_ksize)
        _, rmap = cv2.threshold(rmap,127,255,cv2.THRESH_BINARY)
        
        output_map, output_origin = transform_parallel(dmap, rmap, rmap_origin, tx_transform, ty_transform, theta_transform)
        publish_map_img(output_map, output_origin, rmap_topic)
    return

if __name__ == "__main__":

    tx_transform = 0
    ty_transform = 0
    theta_transform = 0
    # Load default map:
    pkl_filename = "default_map.pkl"
    with open(pkl_filename, 'rb') as file:
        dmap_topic = pickle.load(file)
    dmap_array, dmap_origin, dmap_wh = topic_to_array(dmap_topic)
    dmap = array_to_image(dmap_array)
    initialized_transform_flag = False # indicates whether transform has been found
    tx_transform = 0
    ty_transform = 0
    theta_transform = 0
 
    try:
        rospy.init_node('map_transform_parallel_search', anonymous=True)
        sub = rospy.Subscriber('/rtabmap_map', OccupancyGrid, map_transform_parallel)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
