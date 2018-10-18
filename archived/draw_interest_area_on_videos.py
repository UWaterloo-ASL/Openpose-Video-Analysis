#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 02:05:28 2018

@author: jack.lingheng.meng
"""

import cv2
import os
import glob
import numpy as np
import time
import argparse

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", default='', help="path to the video file")
    ap.add_argument("-o", "--output_directory", default='', help="directory to save processed video")
    args = vars(ap.parse_args())

    if args.get("video", None) is None:
        raise Error("No input video!!")
    # otherwise, we are reading from a video file
    else:
        f_path = args["video"]

    f_name = f_path.split('/')[-1]
    print(''.format(f_name))
    output_video_dir = args['output_directory']#'../ROM_raw_videos_with_interst_area_test'
    sub_dir = f_path.split('/')[-2]
    
    # Get Camera index
    camera_index = int(f_name[6])
    print('Camera index: {}'.format(camera_index))

    # Get frame size
    camera = cv2.VideoCapture(f_path)
    (grabbed, frame) = camera.read()
    original_h, original_w, channels= frame.shape

    # Define the polygon of Whole Interest Area for videos from Camera1 or Camera2
    if camera_index == 1:
        # crop frame: Camera1
        top_edge = int(original_h*(1/10))
        down_edge = int(original_h*1)
        left_edge = int(original_w*(1/5))
        right_edge = int(original_w*(4/5))
        point_1 = [left_edge, top_edge]
        point_2 = [left_edge, down_edge]
        point_3 = [right_edge, down_edge]
        point_4 = [right_edge, top_edge]
        whole_interest_area_polygon = np.array([point_1,point_2,point_3,point_4])
    elif camera_index == 2:
        # crop frame: Camera2
        top_edge = int(original_h*(1/10))
        down_edge = int(original_h*(4/5))
        left_edge = int(original_w*(2.5/5))
        right_edge = int(original_w*(1))
        point_1 = [left_edge, top_edge]
        point_2 = [left_edge, down_edge]
        point_3 = [right_edge, down_edge]
        point_4 = [right_edge, top_edge]
        whole_interest_area_polygon = np.array([point_1,point_2,point_3,point_4])
    else:
        # crop frame: test video
        top_edge = int(original_h*(1/10))
        down_edge = int(original_h*1)
        left_edge = int(original_w*(1/5))
        right_edge = int(original_w*(4/5))
        print('Polygon: Video not from Camera1 or Camera2!')
        point_1 = [left_edge, top_edge]
        point_2 = [left_edge, down_edge]
        point_3 = [right_edge, down_edge]
        point_4 = [right_edge, top_edge]
        whole_interest_area_polygon = np.array([point_1,point_2,point_3,point_4])

    # Define the polygon of Core Interest Area for videos from Camera1 or Camera2
    cropped_w = right_edge - left_edge
    cropped_h = down_edge - top_edge
    if camera_index == 1:
        # polygon for Camera1
        point_1 = [left_edge + int(0.17 * cropped_w), top_edge + int(0.20 * cropped_h)]
        point_2 = [left_edge + int(0.17 * cropped_w), top_edge + int(0.62 * cropped_h)]
        point_3 = [left_edge + int(0.44 * cropped_w), top_edge + int(0.82 * cropped_h)]
        point_4 = [left_edge + int(0.61 * cropped_w), top_edge + int(0.72 * cropped_h)]
        point_5 = [left_edge + int(0.61 * cropped_w), top_edge + int(0.20 * cropped_h)]
        core_interest_area_polygon = np.array([point_1,point_2,point_3,point_4,point_5])
    elif camera_index == 2:
        # polygon for Camera2
        point_1 = [left_edge + int(0.15 * cropped_w), top_edge + int(0.05 * cropped_h)]
        point_2 = [left_edge + int(0.15 * cropped_w), top_edge + int(0.65 * cropped_h)]
        point_3 = [left_edge + int(0.95 * cropped_w), top_edge + int(0.75 * cropped_h)]
        point_4 = [left_edge + int(0.95 * cropped_w), top_edge + int(0.05 * cropped_h)]
        core_interest_area_polygon = np.array([point_1,point_2,point_3,point_4])
    else:
        # polygon for test video
        point_1 = [left_edge + int(0.17 * cropped_w), top_edge + int(0.20 * cropped_h)]
        point_2 = [left_edge + int(0.17 * cropped_w), top_edge + int(0.62 * cropped_h)]
        point_3 = [left_edge + int(0.44 * cropped_w), top_edge + int(0.82 * cropped_h)]
        point_4 = [left_edge + int(0.61 * cropped_w), top_edge + int(0.72 * cropped_h)]
        point_5 = [left_edge + int(0.61 * cropped_w), top_edge + int(0.20 * cropped_h)]
        print('Polygon: Video not from Camera1 or Camera2!')
        core_interest_area_polygon = np.array([point_1,point_2,point_3,point_4,point_5])

    # Define VideoWriter
    output_video_sub_dir = args['output_directory']#os.path.join(output_video_dir, sub_dir)
    if not os.path.exists(output_video_sub_dir):
        os.makedirs(output_video_sub_dir)

    output_video_filename = os.path.join(output_video_sub_dir,'{}_draw_interest_area.avi'.format(f_name.split('.mp4')[0]))
    out_camera = cv2.VideoWriter(output_video_filename, int(camera.get(cv2.CAP_PROP_FOURCC)), camera.get(cv2.CAP_PROP_FPS), (original_w, original_h))

    # loop over the frames of the video
    camera = cv2.VideoCapture(f_path)
    total_frame_number = camera.get(cv2.CAP_PROP_FRAME_COUNT)
    print('Total frame number: {}'.format(total_frame_number))
    start_time = time.time()

    for frame_count in range(int(total_frame_number)):
        # get time
        frame_time = camera.get(cv2.CAP_PROP_POS_MSEC)
        
        if frame_count % 2000 == 0:
            print('Processing frame: {}'.format(frame_count))
            print('Elapsed time: {}s'.format(time.time() - start_time))
        (grabbed, frame) = camera.read()
        if grabbed == True:
            
            cv2.drawContours(frame, [whole_interest_area_polygon], -1, (255, 255, 0), 6, cv2.LINE_AA)
            cv2.drawContours(frame, [core_interest_area_polygon], -1, (255, 0, 0), 6, cv2.LINE_AA)
            
            s_temp = int(frame_time/1000)
            m = s_temp // 60
            s = s_temp % 60
            cv2.putText(frame, "Original video time: {}m {}s".format(m, s), (10, 1060), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
            
            if camera_index == 1:
                cv2.putText(frame, "Camera View", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
                cv2.putText(frame, "Whole Interest Area", (left_edge+10, top_edge+80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 4)
                cv2.putText(frame, "Core Interest", (point_1[0]+10, point_1[1]+80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4)
                cv2.putText(frame, "Area", (point_1[0]+10, point_1[1]+130), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4)
            elif camera_index == 2:
                cv2.putText(frame, "Camera View", (800, 1000), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
                cv2.putText(frame, "Whole Interest Area", (left_edge+10, down_edge-50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 4)
                cv2.putText(frame, "Core Interest Area", (point_2[0]+10, point_2[1]-50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4)
            else:
                pass
            # save processed videos
            out_camera.write(frame)
        else:
            # Pass this frame if cannot grab an image.
            print('Frame: {}, grabbed={}'.format(frame_count, grabbed))