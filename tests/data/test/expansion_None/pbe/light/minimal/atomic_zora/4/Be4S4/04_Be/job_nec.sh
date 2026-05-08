#!/bin/bash -l
#$ -j y
#$ -cwd
#$ -m n
#$ -N x04_Be
#$ -pe impi 24
#$ -l h_rt=86400.0
#$ -l h_vmem=22G
module purge
module load intel mkl impi
module list
export TMPDIR=/tmp

mpiexec -ppn 12 -envlist LD_LIBRARY_PATH -n $NSLOTS /mnt/lxfs2/scratch/bieniek/binaries/fhi-aims.071914_7/bin/aims.071914_7.scalapack.mpi.x &> output.out


