#!/bin/bash
#SBATCH --account=rrg-whitem
#SBATCH --mail-user=mkschleg@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH -o tsallis_ant_medexp_2.out # Standard output
#SBATCH -e tsallis_ant_medexp_2.err # Standard error
#SBATCH --cpus-per-task=48
#SBATCH --time=00-16:00
#SBATCH --mem-per-cpu=4000M     # Total memory for all tasks

source /project/6010404/mkschleg/tsallis_inac/TSALLIS_INAC/bin/activate

parallel -j48 --results "/home/mkschleg/scratch/tsallis_inac/out-tsallis_ant_medexp-2" --joblog tsallis_ant_medexp-2.log python run_from_config.py --id {1} --config configs/tsallis_ant_medexp.toml --base_save_dir /home/mkschleg/scratch/tsallis_inac/tsallis_ant_medexp ::: `seq 48 95`
