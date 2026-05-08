#!/bin/bash -l
# Standard output and error:
#SBATCH -o /home/dts/Documents/theory/errorbar_modelling/tests/data/test/pbe/light/tier1/atomic_zora/4/RhSi_ICSD_988/bandstructure/djob.out.%j
#SBATCH -e /home/dts/Documents/theory/errorbar_modelling/tests/data/test/pbe/light/tier1/atomic_zora/4/RhSi_ICSD_988/bandstructure/djob.err.%j
#SBATCH -D /home/dts/Documents/theory/errorbar_modelling/tests/data/test/pbe/light/tier1/atomic_zora/4/RhSi_ICSD_988/bandstructure
#SBATCH -J submission_RhSi
#SBATCH -t 48:00:00
#SBATCH -p medium40
#SBATCH -N 1
#SBATCH -n 40
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
srun None > aims.out 2> aims.err