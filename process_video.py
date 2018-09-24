#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 11:59:40 2018

@author: jack.lingheng.meng
"""
import os
os.system('module load nixpkgs/16.09  gcc/5.4.0  cuda/8.0.44  cudnn/7.0 opencv/3.3.0  boost/1.65.1 openblas/0.2.20 hdf5/1.8.18 leveldb/1.18 mkl-dnn/0.14 python/3.5.2')
os.system('cd ~')
os.system('source openposeEnv_Python3/bin/activate')
os.system('export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/openpose_python_lib/lib:$HOME/openpose_python_lib/python/openpose:$HOME/caffe/build/lib:/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/Compiler/gcc5.4/boost/1.65.1/lib')

# From Python
# It requires OpenCV installed for Python
import sys
import csv
import cv2
import os
from sys import platform
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import math
from scipy.stats import mode

import pdb
from IPython.core.debugger import Tracer

# Remember to add your installation path here
# Option b
# If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
sys.path.insert(0,r'/home/lingheng/openpose_python_lib/python/openpose') 

# Parameters for OpenPose. Take a look at C++ OpenPose example for meaning of components. Ensure all below are filled
try:
    from openpose import *
except:
    raise Exception('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
params = dict()
params["logging_level"] = 3
params["output_resolution"] = "-1x-1"
params["net_resolution"] = "-1x368" # if crop video, this should be changged and must be mutplies of 16.
params["model_pose"] = "BODY_25"
params["alpha_pose"] = 0.6
params["scale_gap"] = 0.3
params["scale_number"] = 1
params["render_threshold"] = 0.05
# If GPU version is built, and multiple GPUs are available, set the ID here
params["num_gpu_start"] = 0
params["disable_blending"] = False
# Ensure you point to the correct path where models are located
params["default_model_folder"] = "/home/lingheng/openpose/models/"
# Construct OpenPose object allocates GPU memory
openpose = OpenPose(params)

def subplot_estimated_occupancy(occupancy_whole, occupancy_core, occupancy_margin, fig_filename):
    """
    Plot and save estimated occupancy in Three Interest Area.
    Args:
        occupancy_whole (pd.DataFrame): occupancy in Whole Interest Area
        occupancy_core (pd.DataFrame): occupancy in Core Interest Area
        occupancy_margin (pd.DataFrame): occupancy in Margin Interest Area
        fig_filename (string): filename of the saved figure
    """
    ymin = 0
    ymax = 20
    ystep = 4
    lw=1.5
    plt.figure()
    # Whole Interest Area
    plt.subplot(3, 1, 1)
    plt.plot(occupancy_whole['Time']/1000, occupancy_whole['Occupancy'], 'b-', lw, alpha=0.6)
    plt.xlabel('time/second')
    plt.ylabel('# of visitors')
    plt.ylim(ymin, ymax)
    plt.yticks(np.arange(ymin,ymax,ystep))
    plt.title('Estimated # of visitors in Whole Interest Area')
    plt.grid(True, linestyle=':')

    # Core Interest Area
    plt.subplot(3, 1, 2)
    plt.plot(occupancy_core['Time']/1000, occupancy_core['Occupancy'], 'r-', lw, alpha=0.6)
    plt.xlabel('time/second')
    plt.ylabel('# of visitors')
    plt.ylim(ymin, ymax)
    plt.yticks(np.arange(ymin,ymax,ystep))
    plt.title('Estimated # of visitors in Core Interest Area')
    plt.grid(True, linestyle=':')

    # Margin Interest Area
    plt.subplot(3, 1, 3)
    plt.plot(occupancy_margin['Time']/1000, occupancy_margin['Occupancy'], 'g-', lw, alpha=0.6)
    plt.xlabel('time/second')
    plt.ylabel('# of visitors')
    plt.ylim(ymin, ymax)
    plt.yticks(np.arange(ymin,ymax,ystep))
    plt.title('Estimated # of visitors in Margin Interest Area')
    plt.grid(True, linestyle=':')

    plt.tight_layout()
    #plt.show()
    plt.savefig(fig_filename, dpi = 300)

def plot_estimated_occupancy(occupancy_whole, occupancy_core, occupancy_margin, fig_filename):
    ymin=0
    ymax=20
    ystep=4
    
    plt.figure()
    # Whole Interest Area
    plt.plot(occupancy_whole['Time']/1000, occupancy_whole['Occupancy'], 'r-', lw=1.5, alpha=0.6)
    # Core Interest Area
    plt.plot(occupancy_core['Time']/1000, occupancy_core['Occupancy'], 'g-', lw=1.5, alpha=0.6)
    # Margin Interest Area
    plt.plot(occupancy_margin['Time']/1000, occupancy_margin['Occupancy'], 'b-', lw=1.5, alpha=0.6)
    plt.legend(('Whole Interest Area','Core Interest Area','Margin Interest Area'))

    plt.xlabel('time/second')
    plt.ylabel('# of visitors')
    plt.ylim(ymin, ymax, ystep)
    plt.title('Estimated # of visitors in Three Interest Areas')
    plt.grid(True, linestyle=':')

    plt.tight_layout()
    plt.show()
    plt.savefig(fig_filename, dpi = 300)

def moving_smoothing(values, window_size, smooth_type='mode', stride = 1):
    """
    Smoothen estimated occupancy.
    Args:
        values (pandas.DataFrame): 
            values['Time']: time in millisecond
            values['Occupancy']: estimated # of visitors
        window_size(int): the size of sliding window
        smooth_type (string): 
            1. 'mode'
            2. 'mean'
            3. 'min'
            4. 'median'
        stride (int): the stride between two consecutive windows
    Returns:
        smooth_time (np.array): smooth time i.e. the max time in each window
        smooth_occupancy (np.array): smooth occupancy i.e. the mode occupancy in each window
    """
    group_time = []
    group_occupancy = []
    for i in range(0, math.ceil((len(values['Time'])-window_size+1)/stride)):
        group_time.append(values['Time'][i:i+window_size])
        group_occupancy.append(values['Occupancy'][i:i+window_size])
    
    smooth_time = []
    smooth_occupancy = []
    for i in range(len(group_time)):
        smooth_time.append(min(group_time[i])) # max time in the group
        if smooth_type == 'mode':
            smooth_occupancy.append(mode(group_occupancy[i])[0][0]) # mode occupancy in the group
        elif smooth_type == 'mean':
            smooth_occupancy.append(np.round(np.mean(group_occupancy[i])))
        elif smooth_type == 'min':
            smooth_occupancy.append(np.round(np.min(group_occupancy[i])))
        elif smooth_type == 'median':
            smooth_occupancy.append(np.round(np.median(group_occupancy[i])))
        else:
            print('Please choose a proper smooth_type.')
    smooth_values = pd.DataFrame(data={'Time': np.array(smooth_time),
                                       'Occupancy': np.array(smooth_occupancy,dtype=int)})
    return smooth_values#np.array(smooth_time), np.array(smooth_occupancy)

def interpret_senario(occupancy_whole, occupancy_core, occupancy_margin, senarios_truth_table):
    """
    Args:
        occupancy_whole (pd.DataFrame): estimation of coccupancy in whole intrest area
        occupancy_core (pd.DataFrame): estimation of coccupancy in core intrest area
        occupancy_margin (pd.DataFrame): estimation of coccupancy in margin intrest area
        senarios_truth_table (pandas.DataFrame): senarios truth table which has information on
            how to interpret senario.
    Returns:
        senario_sequence (np.array): sequnce of interpreted senario discription according to "Senario Truth Value Table"
        event_sequence (np.array): sequence of interpreted senario code according to "Senario Truth Value Table"
            Note: Different from "Senario Truth Value Table", in this sequence we convert all impossible cases into 0 rather than their original senario code.
        event_time (np.array): the time of each event in millisecond.
    """
    senario_sequence = []
    event_sequence = []
    event_time = []
    for i in range(len(occupancy_whole['Occupancy'])-1):
        change_x = occupancy_core['Occupancy'][i+1] - occupancy_core['Occupancy'][i]
        change_y = occupancy_margin['Occupancy'][i+1] - occupancy_margin['Occupancy'][i]
        change_z = occupancy_whole['Occupancy'][i+1] - occupancy_whole['Occupancy'][i]
        # code: 
        #    0: hold
        #    1: increase
        #    2: decrease
        if change_x == 0:
            x = 0
        elif change_x > 0:
            x = 1
        elif change_x < 0:
            x = 2

        if change_y == 0:
            y = 0
        elif change_y > 0:
            y = 1
        elif change_y < 0:
            y = 2

        if change_z == 0:
            z = 0
        elif change_z > 0:
            z = 1
        elif change_z < 0:
            z = 2
        # convert ternary to decimal
        senario_index = z + y*3 + x*3^2
        senario_sequence.append(senarios_truth_table['Explanation'][senario_index])
        if senarios_truth_table['Truth value'][senario_index] == 0:
            # convert all impossible cases into 0
            event_sequence.append(0)
            #event_sequence.append(senario_index)
        else:
            event_sequence.append(senario_index)
        event_time.append(occupancy_whole['Time'][i])
    return np.array(senario_sequence), np.array(event_sequence), np.array(event_time)

def plot_detected_interesting_event(senario_sequence, event_sequence, event_time, fig_filename):
    ymin = 0
    ymax = 26.0005
    ystep = 1
    plt.figure(figsize=(10, 6))
    plt.scatter(event_time/1000, event_sequence)
    plt.xlabel('time/second')
    plt.ylabel('Event Description')
    plt.ylim(ymin, ymax)
    plt.yticks(np.arange(ymin,ymax,ystep), senarios_truth_table['Explanation'],
               rotation=45, fontsize = 6)
    ax2 = plt.twinx()
    plt.ylabel('Event Code')
    plt.yticks(np.arange(ymin,ymax,ystep), np.arange(ymin,ymax,ystep))
    plt.title('Detected Interesting Events')
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.savefig(fig_filename, dpi = 300)

def tag_interesting_event_description_on_video(video_filename,
                                               smooth_type, window_size, stride,
                                               senario_sequence, event_sequence, event_time):
    """
    Args:
        video_filename (string): filename of video
        smooth_type (string): smooth type (hyper-parameter of smooth method)
        window_size (int): size of smooth window (hyper-parameter of smooth method)
        stride (int): stride size (hyper-parameter of smooth method)
        senario_sequence (np.array): sequnce of interpreted senario discription according to "Senario Truth Value Table"
        event_sequence (np.array): sequence of interpreted senario code according to "Senario Truth Value Table"
            Note: Different from "Senario Truth Value Table", in this sequence we convert all impossible cases into 0 rather than their original senario code.
        event_time (np.array): the time of each event in millisecond.
    """
    camera = cv2.VideoCapture(video_filename)
    (grabbed, frame) = camera.read()
    fheight, fwidth, channels= frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_tagged_camera_frame = cv2.VideoWriter(video_filename.split('.avi')[0]+'_tagged_smooth_type_{}_window_size_{}_stride_{}.avi'.format(smooth_type,window_size,stride),fourcc, camera.get(cv2.CAP_PROP_FPS), (fwidth,fheight))
    # loop over the frames of the video
    total_frame_number = camera.get(cv2.CAP_PROP_FRAME_COUNT)
    max_line_character_num = 60 # 60 characters each line
    detected_event_time = 0
    detected_event_senario = ''
    line_num = 1
    for frame_count in range(len(event_time)):
        if frame_count % 200 == 0:
            print('Processing frame: {}'.format(frame_count))
        (grabbed, frame) = camera.read()
        if grabbed == True:
            cv2.putText(frame, "smooth_type: {}, window_size: {}, stride: {}.".format(smooth_type,window_size,stride), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            time = camera.get(cv2.CAP_PROP_POS_MSEC) #Current position of the video file in milliseconds.
            event_index = frame_count
            if event_sequence[event_index] != 0: # 0 means 'impossible event'
                detected_event_time = time
                detected_event_senario = senario_sequence[event_index]
                cv2.putText(frame, "Detect Interesting Event at: {}s.".format(int(detected_event_time/1000)), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                line_num = np.ceil(len(detected_event_senario)/max_line_character_num)
                for i in range(int(line_num)):
                    if i < line_num:
                        cv2.putText(frame, "{}".format(detected_event_senario[i*max_line_character_num:(i+1)*max_line_character_num]), (10, 180+30*(i)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    else:
                        cv2.putText(frame, "{}".format(detected_event_senario[i*max_line_character_num:end]), (10, 180+30*(i)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            else: # repeat text from last detected event
                cv2.putText(frame, "Interesting Event Time: {}s".format(int(detected_event_time/1000)), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                for i in range(int(line_num)):
                    if i < line_num:
                        cv2.putText(frame, "{}".format(detected_event_senario[i*max_line_character_num:(i+1)*max_line_character_num]), (10, 180+30*(i)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    else:
                        cv2.putText(frame, "{}".format(detected_event_senario[i*max_line_character_num:end]), (10, 180+30*(i)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            # save processed videos
            out_tagged_camera_frame.write(frame)
        else:
            # Pass this frame if cannot grab an image.
            print('Frame: {}, grabbed={} and frame={}'.format(frame_count, grabbed, frame))


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", default='/home/lingheng/project/lingheng/ROM_raw_videos/Camera1_test.mp4', help="path to the video file")
    ap.add_argument("-o", "--output_directory", default='/home/lingheng/project/lingheng/ROM_processed_videos', help="directory to save processed video")
    
    args = vars(ap.parse_args())
    
    if args.get("video", None) is None:
        raise Error("No input video!!")
    # otherwise, we are reading from a video file
    else:
        camera = cv2.VideoCapture(args["video"])
    ########################################################################
    #                         Estimate Occupancy                           #
    ########################################################################
    # frames per second (fps) in the raw video
    fps = camera.get(cv2.CAP_PROP_FPS)
    frame_count = 1
    print("Raw frames per second: {0}".format(fps))
    # prepare to save video
    (grabbed, frame) = camera.read()
    ## downsample frame
    #downsample_rate = 0.5
    #frame = cv2.resize(frame,None,fx=downsample_rate, fy=downsample_rate, interpolation = cv2.INTER_LINEAR)
    # crop frame
    original_h, original_w, channels= frame.shape
    top_edge = int(original_h*(1/10))
    down_edge = int(original_h*1)
    left_edge = int(original_w*(1/5))
    right_edge = int(original_w*(4/5))
    frame_cropped = frame[top_edge:down_edge,left_edge:right_edge,:].copy() # must use copy(), otherwise slice only return address i.e. not hard copy

    cropped_h, cropped_w, channels = frame_cropped.shape
    fwidth = cropped_w 
    fheight = cropped_h
    print("Frame width:{}, Frame height:{}.".format(cropped_w , cropped_h))
    # Define the polygon of Core Interest Area
    point_1 = [int(0.17 * cropped_w), int(0.20 * cropped_h)]
    point_2 = [int(0.17 * cropped_w), int(0.62 * cropped_h)]
    point_3 = [int(0.44 * cropped_w), int(0.82 * cropped_h)]
    point_4 = [int(0.61 * cropped_w), int(0.72 * cropped_h)]
    point_5 = [int(0.61 * cropped_w), int(0.20 * cropped_h)]
    core_interest_area_polygon = np.array([point_1,point_2,point_3,point_4,point_5])

    # get output video file name
    file_path = args["video"].split('/')
    file_name, _= file_path[-1].split('.')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    output_video_filename = os.path.join(args['output_directory'],'{}_processed.avi'.format(file_name))
    out_camera_frame_whole = cv2.VideoWriter(output_video_filename,fourcc, fps, (fwidth,fheight))

    # get output estimated occupancy file name
    out_occupancy_whole = os.path.join(args['output_directory'],'{}_processed_occupancy_whole.csv'.format(file_name))
    out_occupancy_core = os.path.join(args['output_directory'],'{}_processed_occupancy_core.csv'.format(file_name))
    out_occupancy_margin = os.path.join(args['output_directory'],'{}_processed_occupancy_margin.csv'.format(file_name))
    with open(out_occupancy_whole, 'a') as csv_datafile:
        fieldnames = ['Time', 'Occupancy']
        writer = csv.DictWriter(csv_datafile, fieldnames = fieldnames)
        writer.writeheader()
    with open(out_occupancy_core, 'a') as csv_datafile:
        fieldnames = ['Time', 'Occupancy']
        writer = csv.DictWriter(csv_datafile, fieldnames = fieldnames)
        writer.writeheader()    
    with open(out_occupancy_margin, 'a') as csv_datafile:
        fieldnames = ['Time', 'Occupancy']
        writer = csv.DictWriter(csv_datafile, fieldnames = fieldnames)
        writer.writeheader()      
    
    # loop over the frames of the video
    total_frame_number = camera.get(cv2.CAP_PROP_FRAME_COUNT)
    for frame_count in range(int(total_frame_number)):
        if frame_count % 200 == 0:
            print('Processing frame: {}'.format(frame_count))
        (grabbed, frame) = camera.read()
        if grabbed == True:
            time = camera.get(cv2.CAP_PROP_POS_MSEC) #Current position of the video file in milliseconds.
            ## downsample frame
            #frame = cv2.resize(frame,None,fx=downsample_rate, fy=downsample_rate, interpolation = cv2.INTER_LINEAR)
            # crop frame
            frame_cropped = frame[top_edge:down_edge,left_edge:right_edge,:].copy() # must use copy()

            # 1. Whole Interest Area
            # Output keypoints and the image with the human skeleton blended on it
            #    (num_people, 25_keypoints, x_y_confidence) = keypoints_whole_interest_area.shape
            keypoints_whole_interest_area, output_image_whole_interest_area = openpose.forward(frame_cropped, True)

            # 2. Core Interest Area
            core_interest_area_mask = np.zeros(frame_cropped.shape[:2], np.uint8)
            cv2.drawContours(core_interest_area_mask, [core_interest_area_polygon], -1, (255, 255, 255), -1, cv2.LINE_AA)
            core_interest_area = cv2.bitwise_and(output_image_whole_interest_area, frame_cropped, mask=core_interest_area_mask)

            # 3. Margin Interest Area
            margin_interest_area = cv2.bitwise_xor(output_image_whole_interest_area, core_interest_area)
            # TODO: infer occupancy from "keypoints_whole_interest_area"

            # draw the text and timestamp on the frame
            occupancy_whole = keypoints_whole_interest_area.shape[0]
            occupancy_core = 0
            occupancy_margin = 0
            for people in keypoints_whole_interest_area:
                # Sort all keypoints and pick up the one with the highest confidence
                # Meaning of keypoints (https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md)
                ordered_keypoints = people[people[:,2].argsort(),:] # increasing order
                x, y = ordered_keypoints[-1][:2]
                #pdb.set_trace()
                # Choose the one with higher confidence to calculatate occupancy and location
                if cv2.pointPolygonTest(core_interest_area_polygon, (x, y), False) == 1:
                    occupancy_core += 1
                else:
                    occupancy_margin += 1

            cv2.drawContours(output_image_whole_interest_area, [core_interest_area_polygon], -1, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(output_image_whole_interest_area, "Whole Occupancy: {}, Core Occupancy: {}, Margin Occupancy: {}".format(occupancy_whole, occupancy_core, occupancy_margin), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(core_interest_area, "Core Occupancy: {}".format(occupancy_core), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(margin_interest_area, "Margin Occupancy: {}".format(occupancy_margin), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # save estimated occupancy data
            fieldnames = ['Time', 'Occupancy']
            with open(out_occupancy_whole, 'a') as csv_datafile:
                writer = csv.DictWriter(csv_datafile, fieldnames = fieldnames)
                writer.writerow({'Time':time, 'Occupancy': occupancy_whole})
            with open(out_occupancy_core, 'a') as csv_datafile:
                writer = csv.DictWriter(csv_datafile, fieldnames = fieldnames)
                writer.writerow({'Time':time, 'Occupancy': occupancy_core})
            with open(out_occupancy_margin, 'a') as csv_datafile:
                writer = csv.DictWriter(csv_datafile, fieldnames = fieldnames)
                writer.writerow({'Time':time, 'Occupancy': occupancy_margin})
            # save processed videos
            out_camera_frame_whole.write(output_image_whole_interest_area)
        else:
            # Pass this frame if cannot grab an image.
            print('Frame: {}, grabbed={} and frame={}'.format(frame_count, grabbed, frame))

    ########################################################################
    #    Smoothen Estimated Occupancy, then detect interesting event       #
    ########################################################################
    
    # read estimated occupancy in Three Interest Areas
    occupancy_whole = pd.read_csv(out_occupancy_whole)
    occupancy_core = pd.read_csv(out_occupancy_core)
    occupancy_margin = pd.read_csv(out_occupancy_margin)

    # save plot of estimated occupancy in Three Interest Areas
    fig_filename = 'Subplot_Estimated_Occupancy.png'
    subplot_estimated_occupancy(occupancy_whole, occupancy_core, occupancy_margin, fig_filename)
    fig_filename = 'Plot_Estimated_Occupancy.png'
    plot_estimated_occupancy(occupancy_whole, occupancy_core, occupancy_margin, fig_filename)
    
    # smoothen
    window_size = 25
    smooth_type='mean'
    stride = 1
    smooth_occupancy_whole = moving_smoothing(occupancy_whole, window_size, smooth_type)
    smooth_occupancy_core = moving_smoothing(occupancy_core, window_size, smooth_type)
    smooth_occupancy_margin = moving_smoothing(occupancy_margin, window_size, smooth_type)

    fig_filename = 'Subplot_Smooth_Estimated_Occupancy.png'
    subplot_estimated_occupancy(smooth_occupancy_whole,smooth_occupancy_core,smooth_occupancy_margin, fig_filename)
    fig_filename = 'Plot_Smooth_Estimated_Occupancy.png'
    plot_estimated_occupancy(smooth_occupancy_whole,smooth_occupancy_core,smooth_occupancy_margin, fig_filename)
    
    # load Senario Truth Table
    senarios_truth_table = pd.read_csv('analize_visitor_in_and_out_senario_truth_table.csv')
    
    # Interpret
    senario_sequence, event_sequence, event_time = interpret_senario(smooth_occupancy_core, 
                                                                     smooth_occupancy_margin, 
                                                                     smooth_occupancy_whole, 
                                                                     senarios_truth_table)
    # Plot interesting events
    fig_filename = 'Plot_Interesting_Event_smooth_type_{}_window_size_{}_stride{}'.format(smooth_type, window_size, stride)
    plot_detected_interesting_event(senario_sequence, event_sequence, event_time, fig_filename)
    
    # Tag
    tag_interesting_event_description_on_video(output_video_filename, 
                                               smooth_type, window_size, stride,
                                               senario_sequence, event_sequence, event_time)

