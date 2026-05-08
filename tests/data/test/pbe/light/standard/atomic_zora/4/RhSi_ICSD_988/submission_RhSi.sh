#!/bin/bash -l
# Standard output and error:
#SBATCH -o tests/data/test/pbe/light/standard/atomic_zora/4/RhSi_ICSD_988/djob.out.%j
#SBATCH -e tests/data/test/pbe/light/standard/atomic_zora/4/RhSi_ICSD_988/djob.err.%j
#SBATCH -D tests/data/test/pbe/light/standard/atomic_zora/4/RhSi_ICSD_988
#SBATCH -J submission_RhSi
#SBATCH -t 12:00:00
#SBATCH -p standard96
#SBATCH -N 1
#SBATCH --ntasks-per-node=40
#SBATCH -A bep00108
#SBATCH --mail-type=none
#SBATCH --export ALL

module load intel
module load impi
module load anaconda3

ulimit -s unlimited

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=false

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${MKLROOT}/lib/intel64/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${INTELROOT}/lib/intel64/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${I_MPI_ROOT}/intel64/lib

# Command to run aims code.
vibes run relaxation | tee tests/data/test/pbe/light/standard/atomic_zora/4/RhSi_ICSD_988/relaxation.log