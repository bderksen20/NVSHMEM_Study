/*	Bill Derksen - 2/2021
* 
*	Various functions to test/probe cluster for....
*   - Global rank check
* 
*/

// cuda+mpi
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "mpi.h"

//std
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char* argv[]) {

	// Set device before MPI init
	//int local_rank = -1;
	//MPI_Comm_rank(MPI_COMM_WORLD, &local_rank);
	int local_rank = atoi(getenv("OMPI_COMM_WORLD_LOCAL_RANK"));
	int n_devices = 0;
	cudaGetDeviceCount(&n_devices);
	cudaSetDevice(local_rank % n_devices);

	printf("Cuda device count : %d\n", n_devices);
	//cout << "Cuda device count: " << n_devices << "\n";

	// MPI Init...
	int size, rank;
	MPI_Init(&argc, &argv);	                /* starts MPI */
	MPI_Comm_size(MPI_COMM_WORLD, &size);	/* get number of processes */
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);	/* get current process (not thread) id */

	// Print rank / device pair
	//cout << "Hello cluster, from rank " << rank << " on device " << (local_rank % n_devices) << "\n";
	printf("Hello cluster, from rank %d on device %d!!!\n", rank, (local_rank % n_devices);

	// MPI FINALIZE
	MPI_Finalize();

	return 0;
}