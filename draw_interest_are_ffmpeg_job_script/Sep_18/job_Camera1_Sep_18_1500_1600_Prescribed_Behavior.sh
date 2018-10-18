#!/bin/bash
#SBATCH --account=def-dkulic
#SBATCH --cpus-per-task=2    #Maximum of CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=8000M               # memory per node
#SBATCH --time=0-01:00            # time (DD-HH:MM)
#SBATCH --output=./draw_interest_area_ffmpge_job_script_output/Camera1_Sep_18_1500_1600_Prescribed_Behavior_%N-%j.out        # %N for node name, %j for jobID
## Main processing command
ffmpeg -i /project/6001934/lingheng/ROM_Video_Process/ROM_raw_videos/Sep_18/Camera1_Sep_18_1500_1600_Prescribed_Behavior.mp4 -i /project/6001934/lingheng/ROM_Video_Process/Openpose_Video_Analysis_Code/camera1_transparent_img.png -filter_complex "[0:v][1:v] overlay=0:0" -c:a copy /project/6001934/lingheng/ROM_Video_Process/ROM_raw_videos_with_interst_area_ffmpeg_new3/Sep_18/Camera1_Sep_18_1500_1600_Prescribed_Behavior_ffmpeg_with_interest_area.mp4