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
#include "../include/communicators.h"
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

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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

    // ----- TMP test communicator use -----
    MpiComms *communicators = (MpiComms*)malloc(sizeof(MpiComms));
    init_comms(config, communicators);

    comm_graph graph;
    graph.init(communicators);
    graph.geninclist(CROSS_NODE);
    if (rank == 0) graph.print(stdout);

    bool test = check_node(communicators);
    if (rank == 0) fprintf(stdout, "Node check %s\n", (test) ? "true" : "false");

    // test = check_addr(communicators);
    // if (rank == 0) fprintf(stdout, "Addr check %s\n", (test) ? "true" : "false");

    int num_devices_2 = 0;
    cudaErrorCheck( cudaGetDeviceCount(&num_devices_2) );
    MPI_Allreduce(MPI_IN_PLACE, &num_devices_2, 1, MPI_INT, MPI_MIN, communicators->cross_comm.comm);

    if (num_devices_2 != communicators->node_comm.size) {
        fprintf(stderr, "Error: ngpus per node must be the same on all the nodes and must be the same of the nodeComm size.\n");
        MPI_Abort(MPI_COMM_WORLD, __LINE__);
    }
    cudaSetDevice(communicators->node_comm.rank);

#ifndef SKIPCPUAFFINITY
    if (0==rank) printf("List device affinity:\n");
    check_cpu_and_gpu_affinity(communicators->node_comm.rank);
    if (0==rank) printf("List device affinity done.\n\n");
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

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
