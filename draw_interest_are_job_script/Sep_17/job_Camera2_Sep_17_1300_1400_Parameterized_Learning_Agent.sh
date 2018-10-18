#!/bin/bash
#SBATCH --account=def-dkulic
#SBATCH --cpus-per-task=2    #Maximum of CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=8000M               # memory per node
#SBATCH --time=0-02:30            # time (DD-HH:MM)
#SBATCH --output=./draw_interest_are_job_script_output/Camera2_Sep_17_1300_1400_Parameterized_Learning_Agent_%N-%j.out        # %N for node name, %j for jobID
## Main processing command
## -v: path to the raw video file
## -o: directory to save processed video
python ./draw_interest_area_on_videos.py -v ../ROM_raw_videos/Sep_17/Camera2_Sep_17_1300_1400_Parameterized_Learning_Agent.mp4 -o ../ROM_raw_videos_with_interst_area_new4/Sep_17
