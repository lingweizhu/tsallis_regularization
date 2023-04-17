#!/bin/bash
#SBATCH --account=rrg-whitem
#SBATCH --mail-user=mkschleg@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --signal=USR1@90
#SBATCH -o job_out/%x_%a.out # Standard output
#SBATCH -e job_out/%x_%a.err # Standard error

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

source /project/6010404/mkschleg/tsallis_inac/TSALLIS_INAC/bin/activate

mkdir "parallel-logs/${SLURM_JOB_NAME}"

JOB_LOG="parallel-logs/${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}.log"
PARALLEL_CONFIGS="--delay .2 -j ${SLURM_NTASKS} --joblog ${JOB_LOG}"
SRUN_CONFIGS="-N 1 -n 1 --exclusive" # 1 CPU per job?

output_str=`python run_from_config.py --id 0 --config $1 --base_save_dir ~/tmp --get_num_jobs`
NUM_JOBS=`echo "$output_str" | tail -n1`
SEQ_START=$SLURM_ARRAY_TASK_ID # make sure this is starting at 0.
if [[ -v CUSTOM_ARRAY_TASK_COUNT ]]
then
    # This is for re-running an entire set of arrays. Need to set CUSTOM_ARRAY_TASK_COUNT.
    SEQUENCE=`seq $SEQ_START $CUSTOM_ARRAY_TASK_COUNT $(($NUM_JOBS-1))`
else
    # this is for the initial run.
    SEQUENCE=`seq $SEQ_START $SLURM_ARRAY_TASK_COUNT $(($NUM_JOBS-1))`
fi

CONFIG_FILE=$1

echo "NUM TOTAL JOBS: $NUM_JOBS"
echo "ARRAY TASK: $SLURM_ARRAY_TASK_ID"
echo "SEQUENCE RAN: $SEQUENCE"

parallel $PARALLEL_CONFIGS srun $SRUN_CONFIGS python run_from_config.py --id {1} --config $CONFIG_FILE --base_save_dir $2  ::: $SEQUENCE

