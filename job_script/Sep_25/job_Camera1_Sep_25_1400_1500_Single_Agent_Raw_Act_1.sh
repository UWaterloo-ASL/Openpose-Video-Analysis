#!/bin/bash
#SBATCH --account=def-dkulic
#SBATCH --gres=gpu:1              # request GPU generic resource
#SBATCH --cpus-per-task=2    #Maximum of CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=8000M               # memory per node
#SBATCH --time=0-01:30            # time (DD-HH:MM)
#SBATCH --output=./job_script_output/Camera1_Sep_25_1400_1500_Single_Agent_Raw_Act_1_%N-%j.out        # %N for node name, %j for jobID
## Main processing command
## -v: path to the raw video file
## -o: directory to save processed video
python ./process_video_low_frequent_frame.py -v ../ROM_raw_videos_clips/Sep_25/Camera1_Sep_25_1400_1500_Single_Agent_Raw_Act_1.mp4 -o ../ROM_raw_videos_clips_processed_camera2/Sep_25
