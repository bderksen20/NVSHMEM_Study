# CUDA Aware MPI Notes/Documentation
 
Some notes on everything involved in setting up and running CUDA-Aware MPI programs on clusters.  

#### Resources used:  
   - [CUDA-Aware MPI NVIDIA Dev Notes](https://developer.nvidia.com/blog/introduction-cuda-aware-mpi/)  
   - [CUDA-Aware MPI NVIDIA Example](https://github.com/NVIDIA-developer-blog/code-samples/blob/master/posts/cuda-aware-mpi-example/src/CUDA_Aware_MPI.c)
   - [OpenMPI CUDA Documentation](https://www.open-mpi.org/faq/?category=runcuda)
   - [NVIDIA Jacobi Solver](https://github.com/NVIDIA-developer-blog/code-samples)

#### Setup Notes:  
   - [CUDA-Aware OpenMPI Setup](https://kose-y.github.io/blog/2017/12/installing-cuda-aware-mpi/)  
   - Using OpenMPI, build must be configured in order for the "awareness" to work. Build with --> ./configure --with-cuda  
              > *CRC **DOES NOT** have CUDA-Awareness enabled for openmpi currently*  
              > *BRIDGES-2 **DOES** have CUDA-Awareness enabled*
   - Check with: `ompi_info --parsable -l 9 --all | grep mpi_built_with_cuda_support:value`
             
#### Program Flow Notes:  
   - [Multi GPU Programming with MPI](https://developer.nvidia.com/gtc/2020/video/s21067)
   
   1. Need to setup a global rank/topology for process mapping on cluster (we are spanning multiple nodes which may contain multiple GPUs)

#### Cluster/Compilation Notes:
   - [Bridges-2 User Guide](https://www.psc.edu/resources/bridges-2/user-guide-2/#intro)
   - Load modules: `modle load openmpi gcc cuda`
   - Compiling cluster_tests.c (includes mpi & cuda code): `mpicc cluster_tests.c -o cluster_tests -lcudart`  
              > Good idea to have seperate MPI & CUDA files (main and kernel, etc.), seperately compile (mpicc & nvcc), and link for larger apps
   - Launch interactive instance with 2 nodes, 8 GPUs on each for 30 min: `interact -p GPU --gres=gpu:8 -N 2 -t 30:00`
   - Run cluster_tests: `mpiexec -np 4 ./cluster_tests`
 



