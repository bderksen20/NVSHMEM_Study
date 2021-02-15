#!/bin/bash
#SBATCH --job-name=mpi_test
#SBATCH --nodes=2  #number of nodes requested
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --cluster=mpi # mpi, gpu and smp on H2P
#SBATCH --output=test_hello.txt
#SBATCH --partition=ib
#SBATCH --time=0-00:10:00 # 6 hours walltime in dd-hh:mm format
#SBATCH --qos=short # required if walltime is greater than 3 days

module load gcc/8.2.0
module load openmpi/4.0.3

# Navigate to resource dir
cd ~/NVSHMEM_Study/mpi_cuda

echo "Testing OpenMPI Hello World!!!"

# Launch program
mpirun -np 4 ./base_mmult_2sd

