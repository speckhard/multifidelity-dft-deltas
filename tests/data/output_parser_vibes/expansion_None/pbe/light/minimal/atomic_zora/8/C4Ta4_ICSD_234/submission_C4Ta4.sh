#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./djob.out.%j
#SBATCH -e ./djob.err.%j
# Initial working directory:
#SBATCH -D ./
# Job Name:
#SBATCH -J submission_C4Ta4
# Number of nodes and MPI tasks per node:
#BATCH --nodes=1
#SBATCH --ntasks-per-node=32
#
#SBATCH --mail-type=none
#SBATCH --mail-user=<userid>@rzg.mpg.de
#
# Wall clock limit:
#SBATCH --time=00:10:00


export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1


# Queue information.
#SBATCH --partition=general
# Command to run aims code.
srun /u/dansp/aims_2020/FHIaims2020/build/aims.200112_2.scalapack.mpi.x > aims.out 2> aims.err