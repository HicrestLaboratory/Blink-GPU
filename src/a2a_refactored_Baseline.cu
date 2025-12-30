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

    int max_j;
    int loop_count;
    int buff_cycle;
    int fix_buff_size;

    // Parse command-line options
    Config * config = (Config *)(malloc(sizeof(Config)));
    parse_args(argc, argv, config);

    loop_count    = config->loop_count;
    buff_cycle    = config->buff_cycle;
    max_j         = config->max_buff_size;
    fix_buff_size = config->fix_buff_size;

     /* -------------------------------------------------------------------------------------------
        Loop from 8 B to 1 GB
    -------------------------------------------------------------------------------------------- */

    CommunicationBuffers<dtype> buffs;

    RecordsStruct rec;
    rec.init(config);

    int *error = (int*)malloc(sizeof(int)*buff_cycle);
    int *my_error = (int*)malloc(sizeof(int)*buff_cycle);
    cktype *cpu_checks = (cktype*)malloc(sizeof(cktype)*buff_cycle);
    cktype *gpu_checks = (cktype*)malloc(sizeof(cktype)*buff_cycle);
    for(int j=0; j<rec.niter; j++){

        if (j!=0) rec.increase_N();
    
        buffs.init(ALL2ALL, rec.N, MPI_COMM_WORLD);

        cktype *my_cpu_check = (cktype*)malloc(sizeof(cktype)*size);
        cktype *recv_cpu_check = (cktype*)malloc(sizeof(cktype)*size), gpu_check = 0;
        for (int i=0; i<size; i++)
            my_cpu_check[i] = 0U;

        // Initialize all elements of A to 0.0
        for(SZTYPE i=0; i<(rec.N)*size; i++) {
            ((dtype*)buffs.sBuff.host)[i] = 1U * (rank+1);
            ((dtype*)buffs.rBuff.host)[i] = 0U;
        }

        for (int i=0; i<size; i++)
            gpu_device_reduce(((dtype*)buffs.sBuff.device) + (i*(rec.N))*sizeof(dtype), (rec.N), &my_cpu_check[i]);


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

        // TODO reintegrate
        gpu_device_reduce((dtype*)buffs.rBuff.device, size*(rec.N), &gpu_check);
        MPI_Alltoall(my_cpu_check, 1, MPI_cktype, recv_cpu_check, 1, MPI_cktype, MPI_COMM_WORLD);

        cpu_checks[j] = 0;
        gpu_checks[j] = gpu_check;
        for (int i=0; i<size; i++)
            cpu_checks[j] += recv_cpu_check[i];
        my_error[j] = abs(gpu_checks[j] - cpu_checks[j]);

        buffs.clear(rank);
        free(recv_cpu_check);
        free(my_cpu_check);
    }

    rec.init_iter_var(config);

    MPI_Allreduce(my_error, error, buff_cycle, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(rec.inner_elapsed_time, rec.elapsed_time, buff_cycle*loop_count, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    for(int j=0; j<rec.niter; j++){
        if (j!=0) rec.increase_N();

        SZTYPE num_B, int_num_GB;
        double num_GB;

        num_B = sizeof(dtype)*(rec.N)*(size-1);
        // TODO: maybe we can avoid if and just divide always by B_in_GB
        if (j < 31) {
            SZTYPE B_in_GB = 1 << 30;
            num_GB = (double)num_B / (double)B_in_GB;
        } else {
            SZTYPE M = 1 << (j - 30);            
            num_GB = sizeof(dtype)*M*(size-1);
        }

        double avg_time_per_transfer = 0.0;
        for (int i=0; i<rec.nrepetitions; i++) {
            avg_time_per_transfer += rec.inner_elapsed_time[(j*rec.nrepetitions)+i];
            if(rank == 0) printf("\tTransfer size (B): %10" PRIu64 ", Transfer Time (s): %15.9f, Bandwidth (GiB/s): %15.9f, Iteration %d\n", num_B, rec.inner_elapsed_time[(j*rec.nrepetitions)+i], num_GB/rec.inner_elapsed_time[(j*rec.nrepetitions)+i], i);
        }
        avg_time_per_transfer /= ((double)loop_count);

        if(rank == 0) printf("[Average] Transfer size (B): %10" PRIu64 ", Transfer Time (s): %15.9f, Bandwidth (GiB/s): %15.9f, Error: %d\n", num_B, avg_time_per_transfer, num_GB/avg_time_per_transfer, error[j] );
        fflush(stdout);
    }

    char *s = (char*)malloc(sizeof(char)*(20*buff_cycle + 100));
    sprintf(s, "[%d] recv_cpu_check = %u", rank, cpu_checks[0]);
    for (int i=fix_buff_size; i<max_j; i++) {
        sprintf(s+strlen(s), " %10d", cpu_checks[i]);
    }
    sprintf(s+strlen(s), " (for Error)\n");
    printf("%s", s);
    fflush(stdout);

    sprintf(s, "[%d] gpu_checks = %u", rank, gpu_checks[0]);
    for (int i=fix_buff_size; i<max_j; i++) {
        sprintf(s+strlen(s), " %10d", gpu_checks[i]);
    }
    sprintf(s+strlen(s), " (for Error)\n");
    printf("%s", s);
    fflush(stdout);

    free(error);
    free(my_error);
    free(cpu_checks);
    free(gpu_checks);
    rec.free_timers();
    MPI_Finalize();
    return(0);
}
