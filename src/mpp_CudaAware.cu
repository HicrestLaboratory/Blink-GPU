#include <stdio.h>
#include "mpi.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <inttypes.h>

#define MPI

#include "../include/error.h"
#include "../include/type.h"
#include "../include/gpu_ops.h"
#include "../include/device_assignment.h"
#include "../include/prints.h"

#ifdef MPIX_CUDA_AWARE_SUPPORT
/* Needed for MPIX_Query_cuda_support(), below */
#include "mpi-ext.h"
#endif

#define BUFF_CYCLE 31
#define LOOP_COUNT 50

#define WARM_UP 5

void read_line_parameters (int argc, char *argv[], int myrank,
                           int *flag_b, int *flag_l, int *flag_x, int *flag_p,
                           int *loop_count, int *buff_cycle, int *fix_buff_size, int *ncouples ) {

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-l") == 0) {
            if (i == argc) {
                if (myrank == 0) {
                    fprintf(stderr, "Error: specified -l without a value.\n");
                }

                exit(__LINE__);
            }

            *flag_l = 1;
            *loop_count = atoi(argv[i + 1]);
            if (*loop_count <= 0) {
                fprintf(stderr, "Error: loop_count must be a positive integer.\n");
                exit(__LINE__);
            }
            i++;
        } else if (strcmp(argv[i], "-b") == 0) {
            if (i == argc) {
                if (myrank == 0) {
                    fprintf(stderr, "Error: specified -b without a value.\n");
                }

                exit(__LINE__);
            }

            *flag_b = 1;
            *buff_cycle = atoi(argv[i + 1]);
            if (*buff_cycle <= 0) {
                fprintf(stderr, "Error: buff_cycle must be a positive integer.\n");
                exit(__LINE__);
            }
            i++;
        } else if (strcmp(argv[i], "-x") == 0) {
            if (i == argc) {
                if (myrank == 0) {
                fprintf(stderr, "Error: specified -x without a value.\n");
                }

                exit(__LINE__);
            }

            *flag_x = 1;
            *fix_buff_size = atoi(argv[i + 1]);
            if (*fix_buff_size < 0) {
                fprintf(stderr, "Error: fixed buff_size must be >= 0.\n");
                exit(__LINE__);
            }

            i++;
        } else if (strcmp(argv[i], "-p") == 0) {
            if (i == argc) {
                if (myrank == 0) {
                fprintf(stderr, "Error: specified -p without a value.\n");
                }

                exit(__LINE__);
            }

            *flag_p = 1;
            *ncouples = atoi(argv[i + 1]);
            if (*ncouples < 0) {
                fprintf(stderr, "Error: number of ping-pong couples must be >= 1.\n");
                exit(__LINE__);
            }

            i++;
        } else {
            if (0 == myrank) {
                fprintf(stderr, "Unknown option: %s\n", argv[i]);
            }

            exit(__LINE__);
        }
    }
}

// ----------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    printf("Compile time check:\n");
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    printf("This MPI library has CUDA-aware support.\n", MPIX_CUDA_AWARE_SUPPORT);
#elif defined(MPIX_CUDA_AWARE_SUPPORT) && !MPIX_CUDA_AWARE_SUPPORT
    printf("This MPI library does not have CUDA-aware support.\n");
#else
    printf("This MPI library cannot determine if there is CUDA-aware support.\n");
#endif /* MPIX_CUDA_AWARE_SUPPORT */

    printf("Run time check:n");
#if defined(MPIX_CUDA_AWARE_SUPPORT)
    if (1 == MPIX_Query_cuda_support()) {
        printf("This MPI library has CUDA-aware support.\n");
    } else {
        printf("This MPI library does not have CUDA-aware support.\n");
    }
#else /* !defined(MPIX_CUDA_AWARE_SUPPORT) */
    printf("This MPI library cannot determine if there is CUDA-aware support.\n");
#endif /* MPIX_CUDA_AWARE_SUPPORT */




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

    MPI_Status stat;

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

    // Check that all the nodes has the same size
    int nodesize;
    MPI_Allreduce(&mynodesize, &nodesize, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (nodesize != mynodesize) {
        fprintf(stderr, "Error at node %d: mynodesize (%d) does not metch with nodesize (%d)\n", rank, mynodesize, nodesize);
        fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, __LINE__);
    } else {
        if (rank == 0) printf("All the nodes (%d) have the same size (%d)\n", nnodes, nodesize);
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* -------------------------------------------------------------------------------------------
        Reading command line inputs
    --------------------------------------------------------------------------------------------*/

    int opt;
    int max_j;
    int flag_b = 0;
    int flag_l = 0;
    int flag_x = 0;
    int flag_p = 0;
    int loop_count = LOOP_COUNT;
    int buff_cycle = BUFF_CYCLE;
    int fix_buff_size = 0;
    int ncouples = 4;

    // Parse command-line options
    read_line_parameters(argc, argv, rank,
                         &flag_b, &flag_l, &flag_x, &flag_p,
                         &loop_count, &buff_cycle, &fix_buff_size, &ncouples);
    if(flag_x && fix_buff_size >= buff_cycle){buff_cycle = fix_buff_size + 1;} 
       
    // Print message based on the flags
    if (flag_p && rank == 0) printf("Flag p was set with argument: %d\n", ncouples);
    if (flag_b && rank == 0) printf("Flag b was set with argument: %d\n", buff_cycle);
    if (flag_l && rank == 0) printf("Flag l was set with argument: %d\n", loop_count);
    if (flag_x && rank == 0) printf("Flag x was set with argument: %d\n", fix_buff_size);


    if (flag_p) {
        if (nnodes > 1) {
            if (nodesize < ncouples) {
                fprintf(stderr, "Error: mynode (%s) has less gpus (%d) then the required by -p flag (%d)\n", host_name, nodesize, ncouples);
                MPI_Abort(MPI_COMM_WORLD, __LINE__);
            }
        } else {
            if (ncouples > 1) {
                fprintf(stderr, "Error: Multi-Ping-Pong does not support the single node set-up\n");
                MPI_Abort(MPI_COMM_WORLD, __LINE__);
            }
        }
    }

    if(!flag_p){ncouples = nodesize;}

    max_j = (flag_x == 0) ? buff_cycle : (fix_buff_size + 1) ;
    if (rank == 0) printf("buff_cycle: %d loop_count: %d max_j: %d\n", buff_cycle, loop_count, max_j);
    if (flag_x > 0 && rank == 0) printf("fix_buff_size is set as %d\n", fix_buff_size);

    /* -------------------------------------------------------------------------------------------
        MPI Initialize Peer-to-peer communicators
    --------------------------------------------------------------------------------------------*/

    MPI_Comm ppCouples, ppAllCouples, ppFirstSenders;
    int ppCouples_rank, ppAllCouples_rank, ppFirstSenders_rank;
    int ppCouples_size, ppAllCouples_size, ppFirstSenders_size;
    int colour4ppCouples, colour4ppAllCouples , colour4ppFirstSenders;

    int my_peer = -1;
    if ((mynode == 0 || mynode == nnodes-1) && mynodeid < ncouples)
        my_peer = (mynode == 0) ? (rank + (nnodes-1)*nodesize) : (rank - (nnodes-1)*nodesize);

    colour4ppCouples = ((mynode == 0 || mynode == nnodes-1) && mynodeid < ncouples) ? (mynodeid) : (MPI_UNDEFINED);
    MPI_Comm_split(MPI_COMM_WORLD, colour4ppCouples, rank, &ppCouples);
    if(ppCouples != MPI_COMM_NULL) {
        MPI_Comm_rank(ppCouples, &ppCouples_rank);
        MPI_Comm_size(ppCouples, &ppCouples_size);
    } else {
        ppCouples_rank = -1;
        ppCouples_size = -1;
    }

    colour4ppAllCouples = ((mynode == 0 || mynode == nnodes-1) && mynodeid < ncouples) ? (1) : (MPI_UNDEFINED);
    MPI_Comm_split(MPI_COMM_WORLD, colour4ppAllCouples, rank, &ppAllCouples);
    if(ppAllCouples != MPI_COMM_NULL) {
        MPI_Comm_rank(ppAllCouples, &ppAllCouples_rank);
        MPI_Comm_size(ppAllCouples, &ppAllCouples_size);
    } else {
        ppAllCouples_rank = -1;
        ppAllCouples_size = -1;
    }

    colour4ppFirstSenders = (rank < ncouples) ? (mynode) : (MPI_UNDEFINED);
    MPI_Comm_split(MPI_COMM_WORLD, colour4ppFirstSenders, rank, &ppFirstSenders);
    if(ppFirstSenders != MPI_COMM_NULL) {
        MPI_Comm_rank(ppFirstSenders, &ppFirstSenders_rank);
        MPI_Comm_size(ppFirstSenders, &ppFirstSenders_size);
    } else {
        ppFirstSenders_rank = -1;
        ppFirstSenders_size = -1;
    }

    if(ppCouples == MPI_COMM_NULL)
        printf("Process %d did not take part to the ppCouples.\n", rank);
    else
        printf("Process %d took part to the ppCouples (with rank %d and size %d).\n", rank, ppCouples_rank, ppCouples_size);
    fflush(stdout);
    if(ppAllCouples == MPI_COMM_NULL)
        printf("Process %d did not take part to the ppAllCouples.\n", rank);
    else
        printf("Process %d took part to the ppAllCouples (with rank %d and size %d).\n", rank, ppAllCouples_rank, ppAllCouples_size);
    fflush(stdout);
    if(ppFirstSenders == MPI_COMM_NULL)
        printf("Process %d did not take part to the ppFirstSenders.\n", rank);
    else
        printf("Process %d took part to the ppFirstSenders (with rank %d and size %d).\n", rank, ppFirstSenders_rank, ppFirstSenders_size);
    fflush(stdout);

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
    int *my_error = (int*)malloc(sizeof(int)*buff_cycle);
    cktype *cpu_checks = (cktype*)malloc(sizeof(cktype)*buff_cycle);
    cktype *gpu_checks = (cktype*)malloc(sizeof(cktype)*buff_cycle);
    double *elapsed_time = (double*)malloc(sizeof(double)*buff_cycle*loop_count);
    double *inner_elapsed_time = (double*)malloc(sizeof(double)*buff_cycle*loop_count);
    if (ppCouples != MPI_COMM_NULL) {
        for(int j=fix_buff_size; j<max_j; j++){

            (j!=0) ? (N <<= 1) : (N = 1);
            if (rank == 0) {printf("%i#", j); fflush(stdout);}

            // Allocate memory for A on CPU
            dtype *A, *B;
#ifdef PINNED
            cudaHostAlloc(&A, N*sizeof(dtype), cudaHostAllocDefault);
            cudaHostAlloc(&B, N*sizeof(dtype), cudaHostAllocDefault);
#else
            A = (dtype*)malloc(N*sizeof(dtype));
            B = (dtype*)malloc(N*sizeof(dtype));
#endif
            cktype my_cpu_check = 0, recv_cpu_check, gpu_check = 0;

            // Initialize all elements of A to 0.0
	        for(SZTYPE i=0; i<N; i++){
                A[i] = 1U * (rank+1);
                B[i] = 0U;
            }

            dtype *d_B;
            cudaErrorCheck( cudaMalloc(&d_B, N*sizeof(dtype)) );
            cudaErrorCheck( cudaMemcpy(d_B, B, N*sizeof(dtype), cudaMemcpyHostToDevice) );

            dtype *d_A;
            cudaErrorCheck( cudaMalloc(&d_A, N*sizeof(dtype)) );
            cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(dtype), cudaMemcpyHostToDevice) );
            gpu_device_reduce(d_A, N, &my_cpu_check);

            int tag1 = 10;
            int tag2 = 20;
            MPI_Request* request = (MPI_Request*) malloc(sizeof(MPI_Request)*2*ncouples);

            /*

            Implemetantion goes here

            */

            for(int i=1-(WARM_UP); i<=loop_count; i++){
                MPI_Barrier(ppAllCouples);
                start_time = MPI_Wtime();
		
                if(rank < my_peer){
                    MPI_Isend(d_A, N, MPI_dtype, my_peer, tag1, MPI_COMM_WORLD, &(request[ppAllCouples_rank]));
                    MPI_Recv(d_B, N, MPI_dtype, my_peer, tag2, MPI_COMM_WORLD, &stat);
                }
                else {
                    MPI_Recv(d_B, N, MPI_dtype, my_peer, tag1, MPI_COMM_WORLD, &stat);
		            MPI_Isend(d_A, N, MPI_dtype, my_peer, tag2, MPI_COMM_WORLD, &(request[ppAllCouples_rank]));
                }
                MPI_Wait(&(request[ppAllCouples_rank]), MPI_STATUS_IGNORE);

		        stop_time = MPI_Wtime();
		        if (i>0) inner_elapsed_time[(j-fix_buff_size)*loop_count+i-1] = stop_time - start_time;

                if (rank == 0) {printf("%%"); fflush(stdout);}
            }
            if (rank == 0) {printf("#\n"); fflush(stdout);}




            gpu_device_reduce(d_B, N, &gpu_check);
            if(rank < my_peer){
                MPI_Send(&my_cpu_check,   1, MPI_cktype, my_peer, tag1, MPI_COMM_WORLD);
                MPI_Recv(&recv_cpu_check, 1, MPI_cktype, my_peer, tag2, MPI_COMM_WORLD, &stat);
            } else {
                MPI_Recv(&recv_cpu_check, 1, MPI_cktype, my_peer, tag1, MPI_COMM_WORLD, &stat);
                MPI_Send(&my_cpu_check,   1, MPI_cktype, my_peer, tag2, MPI_COMM_WORLD);
            }

            gpu_checks[j] = gpu_check;
            cpu_checks[j] = recv_cpu_check;
            my_error[j] = abs(gpu_checks[j] - cpu_checks[j]);

            fflush(stdout);
            cudaErrorCheck( cudaFree(d_A) );
            cudaErrorCheck( cudaFree(d_B) );
#ifdef PINNED
            cudaFreeHost(A);
            cudaFreeHost(B);
#else
            free(A);
            free(B);
#endif
            free(request);
        }

        if (fix_buff_size<=30) {
            N = 1 << (fix_buff_size - 1);
        } else {
            N = 1 << 30;
            N <<= (fix_buff_size - 31);
        }

        MPI_Allreduce(my_error, error, buff_cycle, MPI_INT, MPI_MAX, ppAllCouples);
        if(ppFirstSenders != MPI_COMM_NULL) {
            MPI_Allreduce(inner_elapsed_time, elapsed_time, buff_cycle*loop_count, MPI_DOUBLE, MPI_MAX, ppFirstSenders);
        }
        for(int j=fix_buff_size; j<max_j; j++) {
            (j!=0) ? (N <<= 1) : (N = 1);

            SZTYPE num_B, int_num_GB;
            double num_GB;

            num_B = sizeof(dtype)*N*ncouples;
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
                if(rank == 0) printf("\tTransfer size (B): %10" PRIu64 ", Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f, Iteration %d\n", num_B, elapsed_time[(j-fix_buff_size)*loop_count+i], num_GB/elapsed_time[(j-fix_buff_size)*loop_count+i], i);
            }
            avg_time_per_transfer /= (double)loop_count;

            if(rank == 0) printf("[Average] Transfer size (B): %10" PRIu64 ", Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f, Error: %d\n", num_B, avg_time_per_transfer, num_GB/avg_time_per_transfer, error[j] );
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
    }

    free(error);
    free(my_error);
    free(cpu_checks);
    free(gpu_checks);
    free(elapsed_time);
    free(inner_elapsed_time);
    MPI_Finalize();
    return(0);
}
