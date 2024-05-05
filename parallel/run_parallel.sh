#!/bin/bash

### USE
# Need to specify job name, array parameters, and number of tasks per node (split so we can fit a good amount of jobs"
# sbatch -J <job_name> --array=0-10 run_config_array.sh 
### INPUT
# $1 = configuration file


### USE
# Need to specify job name, array parameters, and number of tasks per node (split so we can fit a good amount of jobs"
# sbatch -J <job_name> --array=0-10 run_config_array.sh 
### INPUT
# $1 = configuration file
# $2 = base_save_dir

CONFIG_FILE=$1
mkdir "parallel-logs/${CONFIG_FILE}"
JOB_LOG="parallel-logs/${CONFIG_FILE}/${CONFIG_FILE}.log"
PARALLEL_CONFIGS="--delay .2 -j 4 --joblog ${JOB_LOG} --termseq USR1,200,TERM,100,KILL,25"
# SRUN_CONFIGS="-N 1 -n 1 --exclusive" # 1 CPU per job?

output_str=`python run_from_config.py --id 0 --config $1 --base_save_dir ~/tmp --get_num_jobs`
NUM_JOBS=`echo "$output_str" | tail -n1`
SEQUENCE=`seq 0 $(($NUM_JOBS-1))`

echo "NUM TOTAL JOBS: $NUM_JOBS"
echo "ARRAY TASK: $SLURM_ARRAY_TASK_ID"
echo "SEQUENCE RAN: $SEQUENCE"

# parallel $PARALLEL_CONFIGS srun $SRUN_CONFIGS python run_from_config_runner.py --id {1} --config $CONFIG_FILE --base_save_dir $2  ::: $SEQUENCE
# parallel $PARALLEL_CONFIGS python run_from_config_runner.py --id {1} --config $CONFIG_FILE --base_save_dir $2 --num_threads $SLURM_CPUS_PER_TASK  ::: $SEQUENCE
