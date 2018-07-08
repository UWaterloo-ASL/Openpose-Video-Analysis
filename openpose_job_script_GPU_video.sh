#!/bin/bash
#SBATCH --account=def-dkulic
#SBATCH --gres=gpu:1              # request GPU "generic resource"
#SBATCH --cpus-per-task=4    #Maximum of CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=8000M               # memory per node
#SBATCH --time=0-00:30            # time (DD-HH:MM)
#SBATCH --output=%N-%j.out        # %N for node name, %j for jobID

## Main processing command
## --video: path to the raw video file
## --write_video: path to save processed video

cd ~/openpose
./build/examples/openpose/openpose.bin --video /home/lingheng/openpose_input/LAS_ROM_clip_camera1_cropped_video.avi --write_video /home/lingheng/openpose_output/LAS_ROM_clip_camera1_cropped_video_results.avi --display 0 --num_gpu 1
