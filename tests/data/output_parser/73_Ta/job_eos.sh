#######################################################
#example SGE batch script for a plain MPI program using
#512 cores (16 nodes x 32 cores per node) with 32 MPI 
#tasks per node
#executed under the intel runtime 
#######################################################

## run in /bin/bash
#$ -S /bin/bash
## do not join stdout and stderr
#$ -j n
## name of the job
#$ -N x73_Ta
## execute job from the current working directory
#$ -cwd
## do not send mail
#$ -m n
## request 16 nodes (x 32 cores), must be a multiple of 32  
#$ -pe impi_hydra 64
## run for xx minutes
#$ -l h_rt=4:00:00

module load intel impi mkl
ulimit -s unlimited
OMP_NUM_THREADS=1
export OMP_NUM_THREADS
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MKL_HOME/lib/intel64/ 
## to save memory at high core counts, the "connectionless user datagram protocol" 
## can be enabled (might come at the expense of speed)
## see https://software.intel.com/en-us/articles/dapl-ud-support-in-intel-mpi-library
# export I_MPI_DAPL_UD=1

##gather MPI statistics to be analyzed with itac's mps tool
# export I_MPI_STATS=all

##gather MPI debug information (high verbosity)
# export I_MPI_DEBUG=5

mpiexec -n $NSLOTS /scratch/biebj/binaries/fhi-aims.071914_7/bin/aims.071914_7.scalapack.mpi.x &> output.out

