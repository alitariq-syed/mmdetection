#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 16:13:51 2021

@author: ali
"""


import numpy as np
import json

detection_dir = './tutorial_swin/'
#detection_dir = './tutorial_swin_C1/'

f = open(detection_dir+'my_results.bbox.json')
detect_dict = json.load(f)
f.close()

#class_result = np.loadtxt(detection_dir+'output_results_formatted_C1.txt')
# class_result = np.loadtxt(detection_dir+'output_results_formatted_swin.txt')
#class_result = np.loadtxt(detection_dir+'output_results_formatted_swin_remaining_exps_3.txt')
class_result = np.loadtxt(detection_dir+'output_results_formatted_swin_from_scratch.txt')



#image_id ---> 1---3302
#remove detection results for images whos predicted class is not TB

pred_result = np.argmax(class_result,1)

#index of pred_result is equivalent to image_id

#list if image_ids not classified as TB:

tb_image_ids = np.where(pred_result==2)[0]+1
#+1 to bring range to 1:3302

refined_list = []
for i in range(len(detect_dict)):
    if detect_dict[i]['image_id'] in tb_image_ids:
        refined_list.append(detect_dict[i])
        
with open(detection_dir+'refined_my_results_C1_swin_class_and_det_14.bbox.json', 'w') as f:
    json.dump(refined_list, f)