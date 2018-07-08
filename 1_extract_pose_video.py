# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse

import pdb

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
    # frames per second (fps) in the raw video
    fps = camera.get(cv2.CAP_PROP_FPS)
    print("Raw frames per second: {0}".format(fps))
    # prepare to save video
    (grabbed, frame) = camera.read()
    # downsample frame
    downsample_rate = 0.5
    frame = cv2.resize(frame,None,fx=downsample_rate, fy=downsample_rate, interpolation = cv2.INTER_LINEAR)
    # crop frame
    original_h, original_w, channels= frame.shape
    top_edge = int(original_h*(1/10))
    down_edge = int(original_h*1)
    left_edge = int(original_w*(1/5))
    right_edge = int(original_w*(4/5))
    frame_cropped = frame[top_edge:down_edge,left_edge:right_edge,:].copy() # must use copy(), otherwise slice only return address i.e. not hard copy

    fheight, fwidth, channels = frame_cropped.shape
    #pdb.set_trace()
    print("Frame width:{}, Frame height:{}.".format(fwidth , fheight))
    
    # get output file name
    file_path = args["video"].split('/')
    file_name, _= file_path[-1].split('.')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_camera_frame = cv2.VideoWriter(os.path.join(args['output_directory'],'{}_processed.avi'.format(file_name)),fourcc, 20.0, (fwidth,fheight))

    # loop over the frames of the video
    while True:
        (grabbed, frame) = camera.read()
        # if the frame could not be grabbed, then we have reached the end
        # of the video
        if not grabbed:
            break
        # downsample frame
        frame = cv2.resize(frame,None,fx=downsample_rate, fy=downsample_rate, interpolation = cv2.INTER_LINEAR)
        # crop frame
        frame_cropped = frame[top_edge:down_edge,left_edge:right_edge,:].copy() # must use copy()
        #pdb.set_trace()
        # Output keypoints and the image with the human skeleton blended on it
        keypoints, output_image = openpose.forward(frame_cropped, True)
        # Print the human pose keypoints, i.e., a [#people x #keypoints x 3]-dimensional numpy object with the keypoints of all the people on that image
        #print(keypoints.shape)
        
        # draw the text and timestamp on the frame
        occupancy = keypoints.shape[0]
        cv2.putText(output_image, "Current Occupancy: {}".format(occupancy), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # save video
        #cv2.imwrite('messigray_1gpu_cropped2.png',output_image)
        #cv2.imwrite('messigray_1gpu_cropped2_frame.png',frame_cropped)
        out_camera_frame.write(output_image)
        cv2.waitKey(15)
        
