#!/bin/bash
#SBATCH --account=rrg-whitem
#SBATCH --mail-user=mkschleg@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --signal=USR1@90
#SBATCH -o job_out/%x_retry.out # Standard output
#SBATCH -e job_out/%x_retry.err # Standard error

### USE
# Need to specify job name, array parameters, and number of tasks per node (split so we can fit a good amount of jobs"
# sbatch -J <job_name> --array=0-10 run_config_array.sh 
### INPUT
# $1 = configuration file


source /project/6010404/mkschleg/tsallis_inac/TSALLIS_INAC/bin/activate
### USE
# Need to specify job name, array parameters, and number of tasks per node (split so we can fit a good amount of jobs"
# sbatch -J <job_name> --array=0-10 run_config_array.sh 
### INPUT
# $1 = configuration file
# $2 = base_save_dir

# mkdir "parallel-logs/${SLURM_JOB_NAME}"

JOB_LOG=$1
PARALLEL_CONFIGS="--delay .2 -j ${SLURM_NTASKS} --joblog ${JOB_LOG}"
SRUN_CONFIGS="-N 1 -n 1 --exclusive" # 1 CPU per job?

parallel $PARALLEL_CONFIGS --retry-failed

