#!/bin/bash
#SBATCH --account=def-dkulic
#SBATCH --gres=gpu:1              # request GPU "generic resource"
#SBATCH --cpus-per-task=2    #Maximum of CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=8000M               # memory per node
#SBATCH --time=0-04:00            # time (DD-HH:MM)
#SBATCH --output=../%N-%j.out        # %N for node name, %j for jobID

## Main processing command
## -v: path to the raw video file
## -o: directory to save processed video
python ./process_video.py -v '../ROM_raw_videos/Camera1_1pm_2pm_July_4_2018.mp4' -o '../ROM_processed_and_tagged_videos'
