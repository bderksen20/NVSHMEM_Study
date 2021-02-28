# CUDA Aware MPI Notes/Documentation
 
#### Resources used:  
   - [CUDA-Aware MPI NVIDIA Dev Notes](https://developer.nvidia.com/blog/introduction-cuda-aware-mpi/)  
   - [CUDA-Aware MPI NVIDIA Example](https://github.com/NVIDIA-developer-blog/code-samples/blob/master/posts/cuda-aware-mpi-example/src/CUDA_Aware_MPI.c)
   - [OpenMPI CUDA Documentation](https://www.open-mpi.org/faq/?category=runcuda)

#### Setup Notes:  
   - [CUDA-Aware OpenMPI Setup](https://kose-y.github.io/blog/2017/12/installing-cuda-aware-mpi/)  
   - Using OpenMPI, build must be configured in order for the "awareness" to work. Build with --> ./configure --with-cuda  
              > *is CRC openmpi already CUDA configured???? can check at compile time, goto openmpi doc*  
   - Check with:
     ```
     ompi_info --parsable -l 9 --all | grep mpi_built_with_cuda_support:value
     ```
             
#### Program Flow Notes:  
   - [Helpful Video](https://www.youtube.com/watch?v=kIgbQQXbnto)
   
   1. Need to setup a global rank/topology for process mapping on cluster (we are spanning multiple nodes which may contain multiple GPUs)


#### Compilation Notes:

