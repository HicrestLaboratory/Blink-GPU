#include <stdio.h>
#include "mpi.h"
#include "cuda.h"
#include <cuda_runtime.h>
#include <string.h>
#include <unistd.h>
#include <inttypes.h>

#define MPI

#include "error.h"
#include "type.h"
#include "gpu_ops.h"
#include "device_assignment.h"
#include "cmd_util.h"
#include "prints.h"
#include "records.h"
#include "netcommunicators.h"
#include "communication_buffers.h"

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

    MPI_Type_set_name(MPI_dtype,     "MPI_dtype");
    MPI_Type_set_name(MPI_dtype_big, "MPI_dtype_big");

    // Compile-time and run-time checks
    if(rank == 0) compiletime_runtime_checks(stdout); fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    /* -------------------------------------------------------------------------------------------
        Reading command line inputs and communicators init
    --------------------------------------------------------------------------------------------*/

    Config * config = (Config *)(malloc(sizeof(Config)));
    parse_args(argc, argv, config);

    // ----- Set ProcessEnc & MPI comms -----

    ProcessEnv penv;
    penv.init_processenv();

    MpiNetworkComms netcomms;
    netcomms.init(penv);
    if(netcomms.world.rank==0) netcomms.graph.netPrint();

    fflush(stdout);
    MPI_Barrier(netcomms.world.comm);

#ifndef SKIPCPUAFFINITY
    if (0==rank) printf("List device affinity:\n");
    check_cpu_and_gpu_affinity(netcomms.subcomms[netcomms.nfields-1].rank);
    if (0==rank) printf("List device affinity done.\n\n");
    MPI_Barrier(netcomms.world.comm);
#endif

    fflush(stdout);
    MPI_Barrier(netcomms.world.comm);

     /* -------------------------------------------------------------------------------------------
        Loop from 8 B to 1 GB
    -------------------------------------------------------------------------------------------- */

    RecordsStruct rec;
    CommunicationBuffers<dtype> buffs;
    rec.init(config, netcomms.world, ALL2ALL);

    for(int j=0; j<rec.niter; j++){
        if (j!=0) rec.increase_msgsize();
    
        buffs.init(rec.type, rec.msgsize, rec.comm, RANDOM_INT8);
        if ((config->verbose > 2) && (j<3)) buffs.print('s', rank, stdout);

        buffs.sendBuff_reduction(&(rec.sendSideChecks[j]));

        /*

        Implemetantion goes here

        */

        rec.print_iter_info(rank);
        if (rank == 0) {printf("%i#", j); fflush(stdout);}
        for(int i=1-(WARM_UP); i<=rec.nrepetitions; i++){
            MPI_Barrier(rec.comm);
            rec.record_time_start(i);


            MPI_Alltoall(buffs.sBuff.device, buffs.sMpicount, buffs.sMpiDtype, buffs.rBuff.device, buffs.rMpicount, buffs.rMpiDtype, rec.comm);


            rec.record_time_stop(j, i);
            if (rank == 0) {printf("%%"); fflush(stdout);}
        }
        if (rank == 0) printf("#\n"); fflush(stdout);
        MPI_Barrier(rec.comm);

        buffs.recvBuff_reduction(&(rec.recvSideChecks[j]));

        buffs.clear(rank);
    }

    rec.get_statistics(config);

    fflush(stdout);
    MPI_Barrier(rec.comm);

    char *s = (char*)malloc(sizeof(char)*(20*(rec.niter) + 100));
    if (config->verbose > 1) {
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
    }

    if (config->verbose > 0) {
        sprintf(s, "[%d] %15s = ", rank, "check_results");
        for (int i=0; i<rec.niter; i++) {
            sprintf(s+strlen(s), " %5d", rec.check_results[i]);
        }
        sprintf(s+strlen(s), " (for Error)\n");
        printf("%s", s);
        fflush(stdout);
    }

    rec.clear();
    MPI_Finalize();
    return(0);
}
