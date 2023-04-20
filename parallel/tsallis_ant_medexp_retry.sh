#!/bin/bash
#SBATCH --account=rrg-whitem
#SBATCH --mail-user=mkschleg@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH -o tsallis_ant_medexp_1-retry.out # Standard output
#SBATCH -e tsallis_ant_medexp_1-retry.err # Standard error
#SBATCH --cpus-per-task=21
#SBATCH --time=00-16:00
#SBATCH --mem-per-cpu=4000M     # Total memory for all tasks

source /project/6010404/mkschleg/tsallis_inac/TSALLIS_INAC/bin/activate

parallel -j21 --results "/home/mkschleg/scratch/tsallis_inac/out-tsallis_ant_medexp" --retry-failed --joblog tsallis_ant_medexp.log
