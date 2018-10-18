#!/bin/sh
#SBATCH --account=def-dkulic
#SBATCH --array=0-18
#SBATCH --gres=gpu:1              # request GPU "generic resource"
#SBATCH --cpus-per-task=2    #Maximum of CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=8000M               # memory per node
#SBATCH --time=0-01:30            # time (DD-HH:MM)
#SBATCH --output=../%N-%j.out        # %N for node name, %j for jobID

for dir in `ls ../ROM_raw_videos_clips`
do
    echo "direstory: $dir"
    if [ "$dir" == "Sep_12" ];then
        for file in `ls ../ROM_raw_videos_clips/$dir`
            do 
                echo "video: $file"
                python $SLURM_ARRAY_TASK_ID ./process_video_low_frequent_frame.py -v "../ROM_raw_videos_clips/$dir/$file" -o "../ROM_raw_videos_clips_processed/$dir"
            done
    fi
done