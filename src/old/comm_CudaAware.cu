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
#include <sys/time.h>

double timeInSeconds(void) {
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return ((((long long)tv.tv_sec)*1000) + (tv.tv_usec/1000)) / 1000.0;
}

#define BUFF_CYCLE 28
#define LOOP_COUNT 50

#define WARM_UP 5

int main(int argc, char *argv[])
{    
    
    double init = 0, finalize = 0;
    double start = timeInSeconds();
    MPI_Init(&argc, &argv);
    init += timeInSeconds() - start;

    int size, rank, nodeComm, nnodes, mynode;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int dev = assignDeviceToProcess(&nodeComm, &nnodes, &mynode);

    start = timeInSeconds();
    MPI_Finalize();
    finalize += timeInSeconds() - start;
    printf("# %f %f\n", init, finalize);

    return(0);
}
