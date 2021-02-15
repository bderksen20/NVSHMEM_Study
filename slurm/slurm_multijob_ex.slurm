#!/bin/bash
#SBATCH --job-name=mpi_test
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --cluster=mpi # mpi, gpu and smp on H2P
#SBATCH --partition=ib
#SBATCH --time=0-00:10:00 # 6 hours walltime in dd-hh:mm format
#SBATCH --qos=short # required if walltime is greater than 3 days

module load gcc/8.2.0
module load openmpi/4.0.3

# Navigate to resource dir
cd ~/NVSHMEM_Study/mpi_cuda

echo "Testing OpenMPI Hello World!!!"

# Launch program
for i in 1 2 4 8;
do
	sbatch --nodes $i --output test_$inodes.txt mpirun -np $i ./base_mmult_2sd
done

