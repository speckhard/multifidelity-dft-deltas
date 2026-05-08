#!/bin/bash -l
# Standard output and error:
#SBATCH -o /scratch/projects/bep00108/aflow_binaries_exciting_3000_4220//GGA_PBE/precision_0_5/rmt_scaling_0_95/8/O36Si18_ICSD_170544/djob.out.%j
#SBATCH -e /scratch/projects/bep00108/aflow_binaries_exciting_3000_4220//GGA_PBE/precision_0_5/rmt_scaling_0_95/8/O36Si18_ICSD_170544/djob.err.%j
#SBATCH -D /scratch/projects/bep00108/aflow_binaries_exciting_3000_4220//GGA_PBE/precision_0_5/rmt_scaling_0_95/8/O36Si18_ICSD_170544
#SBATCH -J submission_O36Si18
#SBATCH -t 12:00:00
#SBATCH -p standard96
#SBATCH -N 1
#SBATCH --ntasks-per-node=12
#SBATCH -A bep00098
#SBATCH --mail-type=none
#SBATCH --export ALL

module load intel/19.0.5
module load impi/2018.5
  
# Set the number of OpenMP threads as given by the SLURM parameter "cpus-per-task"
export OMP_NUM_THREADS=8
mpirun /home/bepdansp/exciting/exciting/bin/exciting_mpismp input.xml