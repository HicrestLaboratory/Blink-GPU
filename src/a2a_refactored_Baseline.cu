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
    communicators->init(config);

    bool test = communicators->check_node();
    if (rank == 0) fprintf(stdout, "Node check %s\n", (test) ? "true" : "false");

    communicators->assign_cuda_gpu();

#ifndef SKIPCPUAFFINITY
    if (0==rank) printf("List device affinity:\n");
    check_cpu_and_gpu_affinity(communicators->node_comm.rank);
    if (0==rank) printf("List device affinity done.\n\n");
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    comm_graph graph;
    graph.init(communicators);
    graph.geninclist(WORLD);
    if (rank == 0) graph.print(stdout);

    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

     /* -------------------------------------------------------------------------------------------
        Loop from 8 B to 1 GB
    -------------------------------------------------------------------------------------------- */

    RecordsStruct rec;
    rec.init(config, ALL2ALL);
    CommunicationBuffers<dtype> buffs;

    for(int j=0; j<rec.niter; j++){

        if (j!=0) rec.increase_N();
    
        buffs.init(ALL2ALL, rec.N, graph.inccomm.comm, RANDOM_INT8);
        // if (j<5) buffs.print('s', rank, stdout);

        buffs.sendBuff_reduction(&(rec.sendSideChecks[j]));

        /*

        Implemetantion goes here

        */

        rec.print_iter_info(rank);
        if (rank == 0) {printf("%i#", j); fflush(stdout);}
        for(int i=1-(WARM_UP); i<=rec.nrepetitions; i++){
            MPI_Barrier(graph.inccomm.comm);
            rec.record_time_start(i);


            cudaErrorCheck( cudaMemcpy(buffs.sBuff.host, buffs.sBuff.device, buffs.sBuff.bytes, cudaMemcpyDeviceToHost) );
            MPI_Alltoall(buffs.sBuff.host, buffs.sMpicount, buffs.sMpiDtype, buffs.rBuff.host, buffs.rMpicount, buffs.rMpiDtype, graph.inccomm.comm);
            cudaErrorCheck( cudaMemcpy(buffs.rBuff.device, buffs.rBuff.host, buffs.rBuff.bytes, cudaMemcpyHostToDevice) );


            rec.record_time_stop(j, i);
            if (rank == 0) {printf("%%"); fflush(stdout);}
        }
        if (rank == 0) printf("#\n"); fflush(stdout);
        MPI_Barrier(graph.inccomm.comm);

        buffs.recvBuff_reduction(&(rec.recvSideChecks[j]));

        buffs.clear(rank);
    }

    rec.time_maxreduce(graph.inccomm.comm);
    rec.correctness_check(graph.inccomm.comm);
    rec.print_statistics(config, rank, size);

    fflush(stdout);
    MPI_Barrier(graph.inccomm.comm);

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
