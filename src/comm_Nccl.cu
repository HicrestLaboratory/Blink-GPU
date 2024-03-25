#include <stdio.h>
#include "mpi.h"

#include <nccl.h>
#include "cuda.h"
#include <cuda_runtime.h>
#include <string.h>
#include <unistd.h>
#include <inttypes.h>
#include <chrono>

#define MPI

#include "../include/error.h"
#include "../include/type.h"
#include "../include/gpu_ops.h"
#include "../include/device_assignment.h"
#include "../include/cmd_util.h"
#include "../include/prints.h"

#ifdef MPIX_CUDA_AWARE_SUPPORT
/* Needed for MPIX_Query_cuda_support(), below */
#include "mpi-ext.h"
#endif

#define BUFF_CYCLE 28
#define LOOP_COUNT 50

#define WARM_UP 5

int main(int argc, char *argv[])
{    
    /* -------------------------------------------------------------------------------------------
        MPI Initialization 
    --------------------------------------------------------------------------------------------*/
    MPI_Init(&argc, &argv);
    int size, rank, nodeComm, nnodes, mynode;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int dev = assignDeviceToProcess(&nodeComm, &nnodes, &mynode);


    /* -------------------------------------------------------------------------------------------
        NCCL Initialization
    --------------------------------------------------------------------------------------------*/
    ncclUniqueId Id;
    ncclComm_t NCCL_COMM_WORLD;
    double groupStartEnd = 0.0, initRank = 0.0;
    double start = 0;
    start = MPI_Wtime();
    ncclGroupStart();
    groupStartEnd += MPI_Wtime() - start;
    start = MPI_Wtime();
    if (rank == 0) { NCCLCHECK( ncclGetUniqueId(&Id) ); }
    MPI_Bcast(&Id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
    NCCLCHECK( ncclCommInitRank(&NCCL_COMM_WORLD, size, Id, rank) );
    initRank += MPI_Wtime() - start;
    start = MPI_Wtime();
    ncclGroupEnd();
    groupStartEnd += MPI_Wtime() - start;

    printf("# %f %f\n", groupStartEnd, initRank);

    MPI_Finalize();
    return(0);
}
