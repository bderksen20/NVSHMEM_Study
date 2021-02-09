/*  MPI MPI MPI MPI MPI MPI MPI MPI MPI MPI MPI MPI MPI MPI MPI MPI MPI MPI MPI

    Bill Derksen
    Parallelized MMULT Matrix Multiplication w/ MPI 2-sided
    
    TODO: why is this in C?
*/

/* Compilation:    /usr/lib64/mpich/bin/mpicc mpi_program.c -o mpi_program
   Execution:      /usr/lib64/mpich/bin/mpiexec -f hosts -n 100 ./mpi_program

      run with 1 process example:    mpiexec -n 1 ./mpi_program  
      run with 4 processes example:  mpiexec -n 4 ./mpi_program  
*/

/*  MPI Fxn Structure:
    
    MPI datatypes:  MPI_<c data type>, examples: MPI_CHAR, MPI_DOUBLE, MPI_INT

    MPI_Send:   int MPI_Send(
                    void*           msg_buf_p       (in, the message)
                    int             msg_size        (in, message size)
                    MPI_Datatype    msg_type        (in, message type)
                    int             dest            (in, where to send message)
                    int             tag             (in, )
                    MPI_Comm        communicator    (in));

    MPI_Recv:   int MPI_Recv(
                    void*           msg_buf_p       (out, the message)
                    int             msg_size        (in, message size)
                    MPI_Datatype    msg_type        (in, message type)
                    int             source          (in, msg source)
                    int             tag             (in)
                    MPI_Comm        communicator    (in)
                    MPI_Status      status_p        (out));

    MPI_Get_count(
        MPI_Status*     status_p    (in)
        MPI_Datatype    type        (in)
        int*            count_p     (in)
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <time.h>
#include <sys/time.h>

#define        NROW    1024
#define        NCOL    NROW

#define TEST_RESULTS

//Input Array A
int inputArrayA[NROW][NCOL];
//Input Array B
int inputArrayB[NROW][NCOL];
//Input Array C
int inputArrayC[NROW][NCOL];
//Output Array AB
int tempArrayAB[NROW][NCOL];
//Output Array D
int outputArrayD[NROW][NCOL];

struct timeval startTime;
struct timeval finishTime;
double timeIntervalLength;

/* MAIN */
int main(int argc, char *argv[]) {

    int rank, size;
    int source;
    int i, j, k;
    double totalSum;
    

    //INITIALIZE ARRAYS
    for (i = 0; i < NROW; i++) {
        for (j = 0; j < NCOL; j++) {
            inputArrayA[i][j]= i+j;
            inputArrayB[i][j]= j+j;
            inputArrayC[i][j]= i*j;
            tempArrayAB[i][j]= 0;
            outputArrayD[i][j]= 0;
        }
    }

    /* MPI INITIALIZATIONS */
    MPI_Init(&argc, &argv);	/* starts MPI */
    MPI_Comm_size(MPI_COMM_WORLD, &size);	/* get number of processes */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);	/* get current process (not thread) id */

    /* DATA INITIALIZATIONS */
    int thread_row_load = NROW / size;
    int firstRow = rank * thread_row_load;
    int lastRow = (rank + 1) * thread_row_load - 1;
    int scatter_mat[thread_row_load][NCOL];  // matrix to be scattered 
    int r_count = 0;

    //Get the start time
    gettimeofday(&startTime, NULL); /* START TIME */

    /* PERFORM COMPUTATION */
    // Process computes its respective chunk of result rows
    // Don't need to worry about critical sections here since each destination in resulting matrix is only being written to once 
    for (i = firstRow; i <= lastRow; i++) {
        for (j = 0; j < NCOL; j++) {
            for (k = 0; k < NROW; k++) {
                tempArrayAB[i][j] += inputArrayA[i][k] * inputArrayB[k][j];
            }
        }
    }

    // Need to wait for respective chunk of AB computeation to be complete for D = AB * C to be computed 
    // Since the above code is doing just that, have the process go ahead and work on this computation
    for (i = firstRow; i <= lastRow; i++) {
        for (j = 0; j < NCOL; j++) {
            for (k = 0; k < NROW; k++) {
                outputArrayD[i][j] += tempArrayAB[i][k] * inputArrayC[k][j];
                
            }

            scatter_mat[r_count][j] = outputArrayD[i][j];   //put partial-result matrix into matrix to send

        }

        r_count++;
    }

    // NOTE: at this point, each process has calculated a chunk of rows of the solution D
    // To obtain the full result, master proces needs to gather the results and put them in a single matrix
    
    /* MPI COMM STRUCTURE */
    // send respective chunk of solution matrix back to master and compile
    if (rank != 0) {     // for worker processes, send chunk matrix to host

         // Send chunk-ed partial result matrix
         MPI_Send(&(scatter_mat[0][0]), thread_row_load*NCOL, MPI_INT, 0, 0, MPI_COMM_WORLD);

    }
    else {              // for master process,  recieve scatter matrices and map to outputArrayD

        
        for (source = 1; source < size; source++) {

            int row_grab = 0, x, y;

            // Recieve Partial Matrix Result
            MPI_Recv(&(scatter_mat[0][0]), thread_row_load *NCOL, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            /* Print scatter mat, error checking
            printf("\nArray Scatter Mat of %d: \n", rank);
            for (i = 0; i < thread_row_load; i++) {
                for (j = 0; j < NCOL; j++) {
                    printf("%d \t ", scatter_mat[i][j]);
                }
                printf("\n");
             }
             printf("\n"); */

            // Merge partial result matrices in to final result matrix D
            for (x = source * thread_row_load; x < (source + 1) * thread_row_load; x ++) {
                for (y = 0; y < NCOL; y++) {
                    outputArrayD[x][y] = scatter_mat[row_grab][y];
                }

                row_grab++;
            }

        }

    }

    /* MASTER PROCESS COMPLETION */
    if (rank == 0) {
    
        //Get the end time
        gettimeofday(&finishTime, NULL);  /* END TIME */

        #ifdef TEST_RESULTS
        //CALCULATE TOTAL SUM
        //[Just for verification]
         totalSum = 0;

         /*
         printf("\nOutput array: \n");
         for (i = 0; i < NROW; i++) {
             for (j = 0; j < NCOL; j++) {
                 printf("%d \t", outputArrayD[i][j]);
             }
             printf("\n");
         }
         printf("\n");
         */

        for (i = 0; i < NROW; i++) {
            for (j = 0; j < NCOL; j++) {
               totalSum += (double)outputArrayD[i][j];
            }
        }

        printf("\nTotal Sum = %g\n", totalSum);
        #endif

        //Calculate the interval length
         timeIntervalLength = (double)(finishTime.tv_sec - startTime.tv_sec) * 1000000 + (double)(finishTime.tv_usec - startTime.tv_usec);
         timeIntervalLength = timeIntervalLength / 1000;

        //Print the interval length
        printf("Interval length: %g msec.\n", timeIntervalLength);

    }

    // MPI FINALIZE
    MPI_Finalize();

    return 0;
}
