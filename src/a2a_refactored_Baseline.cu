#include <stdio.h>
#include "mpi.h"
#include "cuda.h"
#include <cuda_runtime.h>
#include <string.h>
#include <unistd.h>
#include <inttypes.h>

#define MPI

#include "../include/error.h"
#include "../include/type.h"
#include "../include/gpu_ops.h"
#include "../include/device_assignment.h"
#include "../include/cmd_util.h"
#include "../include/prints.h"
#include "../include/records.h"
#include "../include/communication_buffers.h"

#ifdef MPIX_CUDA_AWARE_SUPPORT
/* Needed for MPIX_Query_cuda_support(), below */
#include "mpi-ext.h"
#endif

int main(int argc, char *argv[])
{

    /* -------------------------------------------------------------------------------------------
        MPI Initialization 
    --------------------------------------------------------------------------------------------*/
    MPI_Init(&argc, &argv);

    int size, nnodes;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rank, mynode;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int namelen;
    char host_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(host_name, &namelen);
    MPI_Barrier(MPI_COMM_WORLD);

    printf("Size = %d, myrank = %d, host_name = %s\n", size, rank, host_name);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    // Map MPI ranks to GPUs
    int num_devices = 0;
    cudaErrorCheck( cudaGetDeviceCount(&num_devices) );

    MPI_Comm nodeComm;

    int dev = assignDeviceToProcess(&nodeComm, &nnodes, &mynode);
    // print device affiniy
#ifndef SKIPCPUAFFINITY
    if (0==rank) printf("List device affinity:\n");
    check_cpu_and_gpu_affinity(dev);
    if (0==rank) printf("List device affinity done.\n\n");
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    int mynodeid = -1, mynodesize = -1;
    MPI_Comm_rank(nodeComm, &mynodeid);
    MPI_Comm_size(nodeComm, &mynodesize);

    // Compile-time and run-time checks
    if(rank == 0) compiletime_runtime_checks(stdout); fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Type_set_name(MPI_dtype,     "MPI_dtype");
    MPI_Type_set_name(MPI_dtype_big, "MPI_dtype_big");

    /* -------------------------------------------------------------------------------------------
        Reading command line inputs
    --------------------------------------------------------------------------------------------*/

    // Parse command-line options
    Config * config = (Config *)(malloc(sizeof(Config)));
    parse_args(argc, argv, config);

     /* -------------------------------------------------------------------------------------------
        Loop from 8 B to 1 GB
    -------------------------------------------------------------------------------------------- */

    CommunicationBuffers<dtype> buffs;

    RecordsStruct rec;
    rec.init(config, ALL2ALL);

    for(int j=0; j<rec.niter; j++){

        if (j!=0) rec.increase_N();
    
        buffs.init(ALL2ALL, rec.N, MPI_COMM_WORLD, RANDOM_INT8);
        if (j<5) buffs.print('s', rank, stdout);

        buffs.sendBuff_reduction(&(rec.sendSideChecks[j]));

        /*

        Implemetantion goes here

        */

        rec.print_iter_info(rank);
        if (rank == 0) {printf("%i#", j); fflush(stdout);}
        for(int i=1-(WARM_UP); i<=rec.nrepetitions; i++){
            MPI_Barrier(MPI_COMM_WORLD);
            rec.record_time_start(i);


            cudaErrorCheck( cudaMemcpy(buffs.sBuff.host, buffs.sBuff.device, buffs.sBuff.bytes, cudaMemcpyDeviceToHost) );
            MPI_Alltoall(buffs.sBuff.host, buffs.sMpicount, buffs.sMpiDtype, buffs.rBuff.host, buffs.rMpicount, buffs.rMpiDtype, MPI_COMM_WORLD);
            cudaErrorCheck( cudaMemcpy(buffs.rBuff.device, buffs.rBuff.host, buffs.rBuff.bytes, cudaMemcpyHostToDevice) );


            rec.record_time_stop(j, i);
            if (rank == 0) {printf("%%"); fflush(stdout);}
        }
        if (rank == 0) printf("#\n"); fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);

        buffs.recvBuff_reduction(&(rec.recvSideChecks[j]));

        buffs.clear(rank);
    }

    rec.time_maxreduce(MPI_COMM_WORLD);
    rec.correctness_check(MPI_COMM_WORLD);
    rec.print_statistics(config, rank, size);

    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    char *s = (char*)malloc(sizeof(char)*(20*(rec.niter) + 100));
    sprintf(s, "[%d] %15s = ", rank, "sendSideChecks");
    for (int i=0; i<rec.niter; i++) {
        sprintf(s+strlen(s), " %5d", rec.sendSideChecks[i]);
    }
    sprintf(s+strlen(s), " (for Error)\n");
    printf("%s", s);
    fflush(stdout);

    sprintf(s, "[%d] %15s = ", rank, "recvSideChecks");
    for (int i=0; i<rec.niter; i++) {
        sprintf(s+strlen(s), " %5d", rec.recvSideChecks[i]);
    }
    sprintf(s+strlen(s), " (for Error)\n");
    printf("%s", s);
    fflush(stdout);

    sprintf(s, "[%d] %15s = ", rank, "check_results");
    for (int i=0; i<rec.niter; i++) {
        sprintf(s+strlen(s), " %5d", rec.check_results[i]);
    }
    sprintf(s+strlen(s), " (for Error)\n");
    printf("%s", s);
    fflush(stdout);

    rec.free_timers();
    rec.free_correctness();
    MPI_Finalize();
    return(0);
}
