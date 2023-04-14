#!/bin/bash
#SBATCH --account=rrg-whitem
#SBATCH --mail-user=mkschleg@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH -o tsallis_ant_medexp_3.out # Standard output
#SBATCH -e tsallis_ant_medexp_3.err # Standard error
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=5
#SBATCH --time=00-36:00
#SBATCH --mem-per-cpu=4000M     # Total memory for all tasks

source /project/6010404/mkschleg/tsallis_inac/TSALLIS_INAC/bin/activate

parallel --delay .2 -j 4 --results "/home/mkschleg/scratch/tsallis_inac/out-tsallis_ant_medexp-3" --joblog tsallis_ant_medexp-3.log srun -N 1 -n 1 --exclusive python run_from_config.py --id {1} --config configs/tsallis_ant_medexp.toml --base_save_dir /home/mkschleg/scratch/tsallis_inac/tsallis_ant_medexp ::: `seq 96 99`
