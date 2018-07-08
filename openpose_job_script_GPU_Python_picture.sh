#!/bin/bash
#SBATCH --account=def-dkulic
#SBATCH --gres=gpu:1              # request GPU "generic resource"
#SBATCH --cpus-per-task=4    #Maximum of CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=8000M               # memory per node
#SBATCH --time=0-00:30            # time (DD-HH:MM)
#SBATCH --output=%N-%j.out        # %N for node name, %j for jobID


python 1_extract_pose_picture.py