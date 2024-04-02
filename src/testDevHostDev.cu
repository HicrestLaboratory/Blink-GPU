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

#ifdef MPIX_CUDA_AWARE_SUPPORT
/* Needed for MPIX_Query_cuda_support(), below */
#include "mpi-ext.h"
#endif

#define BUFF_CYCLE 31
#define LOOP_COUNT 50

#define WARM_UP 5

int main(int argc, char *argv[])
{


    /* -------------------------------------------------------------------------------------------
        MPI Initialization 
    --------------------------------------------------------------------------------------------*/
//     MPI_Init(&argc, &argv);
//
//     int size, nnodes;
//     MPI_Comm_size(MPI_COMM_WORLD, &size);
//
//     int rank, mynode;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//
//     int namelen;
//     char host_name[MPI_MAX_PROCESSOR_NAME];
//     MPI_Get_processor_name(host_name, &namelen);
//     MPI_Barrier(MPI_COMM_WORLD);
//
//     printf("Size = %d, myrank = %d, host_name = %s\n", size, rank, host_name);
//     fflush(stdout);
//     MPI_Barrier(MPI_COMM_WORLD);
//
//     MPI_Status stat;
//
    // Map MPI ranks to GPUs
    int num_devices = 0;
    cudaErrorCheck( cudaGetDeviceCount(&num_devices) );
//
//     MPI_Comm nodeComm;
//     int dev = assignDeviceToProcess(&nodeComm, &nnodes, &mynode);
//     // print device affiniy
// #ifndef SKIPCPUAFFINITY
//     if (0==rank) printf("List device affinity:\n");
//     check_cpu_and_gpu_affinity(dev);
//     if (0==rank) printf("List device affinity done.\n\n");
//     MPI_Barrier(MPI_COMM_WORLD);
// #endif
//
//     int mynodeid = -1, mynodesize = -1;
//     MPI_Comm_rank(nodeComm, &mynodeid);
//     MPI_Comm_size(nodeComm, &mynodesize);
//
//     int rank2 = size-1;
//
//     // Get the group or processes of the default communicator
//     MPI_Group world_group;
//     MPI_Comm_group(MPI_COMM_WORLD, &world_group);
//
//     // Keep only the processes 0 and 1 in the new group.
//     int ranks[2];
//     ranks[0] = 0;
//     ranks[1] = rank2;
//     MPI_Group pp_group;
//     MPI_Group_incl(world_group, 2, ranks, &pp_group);
//
//     // Create the new communicator from that group of processes.
//     MPI_Comm ppComm;
//     MPI_Comm_create(MPI_COMM_WORLD, pp_group, &ppComm);
//
//     // Do a broadcast only between the processes of the new communicator.
//
//     if(ppComm == MPI_COMM_NULL) {
//         // I am not part of the ppComm.
//         printf("Process %d did not take part to the ppComm.\n", rank);
//     } else {
//         // I am part of the new ppComm.
//         printf("Process %d took part to the ppComm.\n", rank);
//     }
//
//     // Keep only the first sender processe (i.e. 0) in the new group.
//     int ranks_0[1];
//     ranks_0[0] = 0;
//     MPI_Group firstsender_group;
//     if (rank == 0)
//         MPI_Group_incl(world_group, 1, ranks_0, &firstsender_group);
//     else
//         MPI_Group_incl(world_group, 0, ranks_0, &firstsender_group);
//
//     // Create the new communicator from that group of processes.
//     MPI_Comm firstsenderComm;
//     MPI_Comm_create(MPI_COMM_WORLD, firstsender_group, &firstsenderComm);
//
//     // Do a broadcast only between the processes of the new communicator.
//
//     if(firstsenderComm == MPI_COMM_NULL) {
//         printf("Process %d did not take part to the firstsenderComm.\n", rank);
//     } else {
//         printf("Process %d took part to the firstsenderComm.\n", rank);
//     }
//     MPI_Barrier(MPI_COMM_WORLD);

    /* -------------------------------------------------------------------------------------------
        Reading command line inputs
    --------------------------------------------------------------------------------------------*/

    int opt;
    int max_j;
    int flag_b = 0;
    int flag_l = 0;
    int flag_x = 0;
    int loop_count = LOOP_COUNT;
    int buff_cycle = BUFF_CYCLE;
    int fix_buff_size = 0;

    // Parse command-line options
    read_line_parameters(argc, argv, 0,
                         &flag_b, &flag_l, &flag_x,
                         &loop_count, &buff_cycle, &fix_buff_size);
    if(flag_x && fix_buff_size >= buff_cycle){buff_cycle = fix_buff_size + 1;}    
    // Print message based on the flags
    if (flag_b) printf("Flag b was set with argument: %d\n", buff_cycle);
    if (flag_l) printf("Flag l was set with argument: %d\n", loop_count);
    if (flag_x) printf("Flag x was set with argument: %d\n", fix_buff_size);

    max_j = (flag_x == 0) ? buff_cycle : (fix_buff_size + 1) ;
    /*if (rank == 0)*/ printf("buff_cycle: %d loop_count: %d max_j: %d\n", buff_cycle, loop_count, max_j);
    if (flag_x > 0) printf("fix_buff_size is set as %d\n", fix_buff_size);

     /* -------------------------------------------------------------------------------------------
        Loop from 8 B to 1 GB
    --------------------------------------------------------------------------------------------*/

    SZTYPE N;
    if (fix_buff_size<=30) {
        N = 1 << (fix_buff_size - 1);
    } else {
        N = 1 << 30;
        N <<= (fix_buff_size - 31);
    }

    double start_time, stop_time;
    int *error = (int*)malloc(sizeof(int)*buff_cycle);
    int *my_error0 = (int*)malloc(sizeof(int)*buff_cycle);
    int *my_error1 = (int*)malloc(sizeof(int)*buff_cycle);
    cktype *cpu_checks0 = (cktype*)malloc(sizeof(cktype)*buff_cycle);
    cktype *gpu_checks0 = (cktype*)malloc(sizeof(cktype)*buff_cycle);
    cktype *cpu_checks1 = (cktype*)malloc(sizeof(cktype)*buff_cycle);
    cktype *gpu_checks1 = (cktype*)malloc(sizeof(cktype)*buff_cycle);
    double *elapsed_time = (double*)malloc(sizeof(double)*buff_cycle*loop_count);
    double *inner_elapsed_time = (double*)malloc(sizeof(double)*buff_cycle*loop_count);
//     if (rank == 0 || rank == rank2) {
        for(int j=fix_buff_size; j<max_j; j++) {

            (j!=0) ? (N <<= 1) : (N = 1);
            /*if (rank == 0)*/ {printf("%i#", j); fflush(stdout);}

            // Allocate memory for A on CPU
            dtype *A0, *B0, *A1, *B1;
            cktype my_cpu_check0 = 0, recv_cpu_check0, gpu_check0 = 0;
            cktype my_cpu_check1 = 0, recv_cpu_check1, gpu_check1 = 0;
#ifdef PINNED
            cudaHostAlloc(&A0, N*sizeof(dtype), cudaHostAllocDefault);
            cudaHostAlloc(&B0, N*sizeof(dtype), cudaHostAllocDefault);
            cudaHostAlloc(&A1, N*sizeof(dtype), cudaHostAllocDefault);
            cudaHostAlloc(&B1, N*sizeof(dtype), cudaHostAllocDefault);
#else
            A0 = (dtype*)malloc(N*sizeof(dtype));
            B0 = (dtype*)malloc(N*sizeof(dtype));
            A1 = (dtype*)malloc(N*sizeof(dtype));
            B1 = (dtype*)malloc(N*sizeof(dtype));
#endif

            // Initialize all elements of A to 0.0
            for(SZTYPE i=0; i<N; i++){
                A0[i] = 1U /** (rank+1)*/;
                B0[i] = 0U;

                A1[i] = 2U /** (rank+1)*/;
                B1[i] = 0U;
            }

            dtype *d_B0 = NULL, *d_B1 = NULL;
            cudaSetDevice(0);
            cudaErrorCheck( cudaMalloc(&d_B0, N*sizeof(dtype)) );
            cudaErrorCheck( cudaMemcpy(d_B0, B0, N*sizeof(dtype), cudaMemcpyHostToDevice) );

            cudaSetDevice(1);
            cudaErrorCheck( cudaMalloc(&d_B1, N*sizeof(dtype)) );
            cudaErrorCheck( cudaMemcpy(d_B1, B1, N*sizeof(dtype), cudaMemcpyHostToDevice) );

            dtype *d_A0 = NULL, *d_A1 = NULL;
            cudaSetDevice(0);
            cudaErrorCheck( cudaMalloc(&d_A0, N*sizeof(dtype)) );
            cudaErrorCheck( cudaMemcpy(d_A0, A0, N*sizeof(dtype), cudaMemcpyHostToDevice) );
            gpu_device_reduce(d_A0, N, &my_cpu_check0);

            cudaSetDevice(1);
            cudaErrorCheck( cudaMalloc(&d_A1, N*sizeof(dtype)) );
            cudaErrorCheck( cudaMemcpy(d_A1, A1, N*sizeof(dtype), cudaMemcpyHostToDevice) );
            gpu_device_reduce(d_A1, N, &my_cpu_check1);

            int tag1 = 10;
            int tag2 = 20;

            /*

            Implemetantion goes here

            */
            for(int i=1-(WARM_UP); i<=loop_count; i++){
                start_time = MPI_Wtime();

                cudaErrorCheck( cudaMemcpy(B1, d_A0, N*sizeof(dtype), cudaMemcpyDeviceToHost) );
                cudaSetDevice(1);
                cudaErrorCheck( cudaMemcpy(d_B1, B1, N*sizeof(dtype), cudaMemcpyHostToDevice) );

                cudaErrorCheck( cudaMemcpy(B0, d_A1, N*sizeof(dtype), cudaMemcpyDeviceToHost) );
                cudaSetDevice(0);
                cudaErrorCheck( cudaMemcpy(d_B0, B0, N*sizeof(dtype), cudaMemcpyHostToDevice) );

                stop_time = MPI_Wtime();
                if (i>0) inner_elapsed_time[(j-fix_buff_size)*loop_count+i-1] = stop_time - start_time;

                /*if (rank == 0)*/ {printf("%%"); fflush(stdout);}
            }
            /*if (rank == 0)*/ {printf("#\n"); fflush(stdout);}


            cudaSetDevice(0);
            gpu_device_reduce(d_B0, N, &gpu_check0);

            cudaSetDevice(1);
            gpu_device_reduce(d_B1, N, &gpu_check1);
//             if(rank == 0){
//                 MPI_Send(&my_cpu_check,   1, MPI_cktype, rank2, tag1, MPI_COMM_WORLD);
//                 MPI_Recv(&recv_cpu_check, 1, MPI_cktype, rank2, tag2, MPI_COMM_WORLD, &stat);
//             } else if(rank == rank2){
//                 MPI_Recv(&recv_cpu_check, 1, MPI_cktype, 0, tag1, MPI_COMM_WORLD, &stat);
//                 MPI_Send(&my_cpu_check,   1, MPI_cktype, 0, tag2, MPI_COMM_WORLD);
//             }
            memcpy(&recv_cpu_check0, &my_cpu_check1, sizeof(cktype));
            memcpy(&recv_cpu_check1, &my_cpu_check0, sizeof(cktype));

            gpu_checks0[j] = gpu_check0;
            gpu_checks1[j] = gpu_check1;
            cpu_checks0[j] = recv_cpu_check0;
            cpu_checks1[j] = recv_cpu_check1;
            my_error0[j] = abs(gpu_checks0[j] - cpu_checks0[j]);
            my_error1[j] = abs(gpu_checks1[j] - cpu_checks1[j]);

            cudaSetDevice(0);
            cudaErrorCheck( cudaFree(d_A0) );
            cudaErrorCheck( cudaFree(d_B0) );

            cudaSetDevice(1);
            cudaErrorCheck( cudaFree(d_A1) );
            cudaErrorCheck( cudaFree(d_B1) );
#ifdef PINNED
            cudaFreeHost(A0);
            cudaFreeHost(B0);
            cudaFreeHost(A1);
            cudaFreeHost(B1);
#else
            free(A0);
            free(B0);
            free(A1);
            free(B1);
#endif
        }

        if (fix_buff_size<=30) {
            N = 1 << (fix_buff_size - 1);
        } else {
            N = 1 << 30;
            N <<= (fix_buff_size - 31);
        }

//         MPI_Allreduce(my_error, error, buff_cycle, MPI_INT, MPI_MAX, ppComm);
        for (int i=0; i<buff_cycle; i++)
            error[i] = (my_error1[i] > my_error0[i]) ? my_error1[i] : my_error0[i];
        //MPI_Allreduce(inner_elapsed_time, elapsed_time, buff_cycle*loop_count, MPI_DOUBLE, MPI_MAX, firstsenderComm);
        memcpy(elapsed_time, inner_elapsed_time, buff_cycle*loop_count*sizeof(double)); // No need to do allreduce, there is only one rank in firstsenderComm
        for(int j=fix_buff_size; j<max_j; j++) {
            (j!=0) ? (N <<= 1) : (N = 1);

            SZTYPE num_B, int_num_GB;
            double num_GB;

            num_B = sizeof(dtype)*N;
            // TODO: maybe we can avoid if and just divide always by B_in_GB
            if (j < 31) {
                SZTYPE B_in_GB = 1 << 30;
                num_GB = (double)num_B / (double)B_in_GB;
            } else {
                SZTYPE M = 1 << (j - 30);            
                num_GB = sizeof(dtype)*M;
            }

            double avg_time_per_transfer = 0.0;
            for (int i=0; i<loop_count; i++) {
                elapsed_time[(j-fix_buff_size)*loop_count+i] /= 2.0;
                avg_time_per_transfer += elapsed_time[(j-fix_buff_size)*loop_count+i];
                /*if(rank == 0)*/ printf("\tTransfer size (B): %10" PRIu64 ", Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f, Iteration %d\n", num_B, elapsed_time[(j-fix_buff_size)*loop_count+i], num_GB/elapsed_time[(j-fix_buff_size)*loop_count+i], i);
            }
            avg_time_per_transfer /= (double)loop_count;

            /*if(rank == 0)*/ printf("[Average] Transfer size (B): %10" PRIu64 ", Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f, Error: %d\n", num_B, avg_time_per_transfer, num_GB/avg_time_per_transfer, error[j] );
            fflush(stdout);
        }


        char *s = (char*)malloc(sizeof(char)*(20*buff_cycle + 100));
        sprintf(s, "[%d] recv_cpu_check = %u", 0, cpu_checks0[0]);
        for (int i=fix_buff_size; i<max_j; i++) {
            sprintf(s+strlen(s), " %10d", cpu_checks0[i]);
        }
        sprintf(s+strlen(s), " (for Error)\n");
        printf("%s", s);
        sprintf(s, "[%d] recv_cpu_check = %u", 1, cpu_checks1[0]);
        for (int i=fix_buff_size; i<max_j; i++) {
            sprintf(s+strlen(s), " %10d", cpu_checks1[i]);
        }
        sprintf(s+strlen(s), " (for Error)\n");
        printf("%s", s);
        fflush(stdout);

        sprintf(s, "[%d] gpu_checks = %u", 0, gpu_checks0[0]);
        for (int i=fix_buff_size; i<max_j; i++) {
            sprintf(s+strlen(s), " %10d", gpu_checks0[i]);
        }
        sprintf(s+strlen(s), " (for Error)\n");
        printf("%s", s);

        sprintf(s, "[%d] gpu_checks = %u", 1, gpu_checks1[0]);
        for (int i=fix_buff_size; i<max_j; i++) {
            sprintf(s+strlen(s), " %10d", gpu_checks1[i]);
        }
        sprintf(s+strlen(s), " (for Error)\n");
        printf("%s", s);
        fflush(stdout);
//     }
    free(error);
    free(my_error0);
    free(my_error1);
    free(cpu_checks0);
    free(cpu_checks1);
    free(gpu_checks0);
    free(gpu_checks1);
    free(elapsed_time);
    free(inner_elapsed_time);
//     MPI_Finalize();
    return(0);
}
