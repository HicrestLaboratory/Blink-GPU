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

#include "../include/common.h"

#define BUFF_CYCLE 31
#define LOOP_COUNT 50

#define WARM_UP 5

int main(int argc, char *argv[])
{
    compile_time_check();

    /* -------------------------------------------------------------------------------------------
        MPI Initialization 
    --------------------------------------------------------------------------------------------*/
    MPI_Init(&argc, &argv);

    MPI_Status stat;
    MPI_Comm nodeComm;
    int num_devices, my_dev;
    int size, nnodes, rank, mynode, mynodeid, mynodesize;

    my_mpi_init(&size, &nnodes, &rank, &mynode, &num_devices, &my_dev, &nodeComm, &mynodeid, &mynodesize);

    int rank2 = size-1;
    MPI_Comm ppComm, firstsenderComm;
    define_pp_comm (rank, 0, rank2, &ppComm, &firstsenderComm);

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
    new_read_line_parameters(argc, argv, rank,
                         &flag_b, &flag_l, &flag_x,
                         &loop_count, &buff_cycle, &fix_buff_size, &max_j);

    // Print message based on the flags
    print_line_parameters (rank, flag_b, flag_l, flag_x, loop_count, buff_cycle, fix_buff_size, max_j );

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
    if (rank == 0 || rank == rank2) {
        for(int j=fix_buff_size; j<max_j; j++) {

            (j!=0) ? (N <<= 1) : (N = 1);
            if (rank == 0) {printf("%i#", j); fflush(stdout);}

            // Allocate memory for A on CPU
            dtype *A, *B;
            cktype my_cpu_check = 0, recv_cpu_check, gpu_check = 0;
#ifdef PINNED
            cudaHostAlloc(&A, N*sizeof(dtype), cudaHostAllocDefault);
            cudaHostAlloc(&B, N*sizeof(dtype), cudaHostAllocDefault);
#else
            A = (dtype*)malloc(N*sizeof(dtype));
            B = (dtype*)malloc(N*sizeof(dtype));
#endif

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

            /*

            Implemetantion goes here

            */

            for(int i=1-(WARM_UP); i<=loop_count; i++){
                MPI_Barrier(ppComm);
                start_time = MPI_Wtime();

                if(rank == 0){
                    cudaErrorCheck( cudaMemcpy(A, d_A, N*sizeof(dtype), cudaMemcpyDeviceToHost) );
                    MPI_Send(A, N, MPI_dtype, rank2, tag1, MPI_COMM_WORLD);
                    MPI_Recv(B, N, MPI_dtype, rank2, tag2, MPI_COMM_WORLD, &stat);
                    cudaErrorCheck( cudaMemcpy(d_B, B, N*sizeof(dtype), cudaMemcpyHostToDevice) );
                }
                else if(rank == rank2){
                    MPI_Recv(B, N, MPI_dtype, 0, tag1, MPI_COMM_WORLD, &stat);
                    cudaErrorCheck( cudaMemcpy(d_B, B, N*sizeof(dtype), cudaMemcpyHostToDevice) );
                    cudaErrorCheck( cudaMemcpy(A, d_A, N*sizeof(dtype), cudaMemcpyDeviceToHost) );
                    MPI_Send(A, N, MPI_dtype, 0, tag2, MPI_COMM_WORLD);
                }

                stop_time = MPI_Wtime();
                if (i>0) inner_elapsed_time[(j-fix_buff_size)*loop_count+i-1] = stop_time - start_time;

                if (rank == 0) {printf("%%"); fflush(stdout);}
            }
            if (rank == 0) {printf("#\n"); fflush(stdout);}


            gpu_device_reduce(d_B, N, &gpu_check);
            if(rank == 0){
                MPI_Send(&my_cpu_check,   1, MPI_cktype, rank2, tag1, MPI_COMM_WORLD);
                MPI_Recv(&recv_cpu_check, 1, MPI_cktype, rank2, tag2, MPI_COMM_WORLD, &stat);
            } else if(rank == rank2){
                MPI_Recv(&recv_cpu_check, 1, MPI_cktype, 0, tag1, MPI_COMM_WORLD, &stat);
                MPI_Send(&my_cpu_check,   1, MPI_cktype, 0, tag2, MPI_COMM_WORLD);
            }

            gpu_checks[j] = gpu_check;
            cpu_checks[j] = recv_cpu_check;
            my_error[j] = abs(gpu_checks[j] - cpu_checks[j]);

            cudaErrorCheck( cudaFree(d_A) );
            cudaErrorCheck( cudaFree(d_B) );
#ifdef PINNED
            cudaFreeHost(A);
            cudaFreeHost(B);
#else
            free(A);
            free(B);
#endif
        }

        if (fix_buff_size<=30) {
            N = 1 << (fix_buff_size - 1);
        } else {
            N = 1 << 30;
            N <<= (fix_buff_size - 31);
        }

        MPI_Allreduce(my_error, error, buff_cycle, MPI_INT, MPI_MAX, ppComm);
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
                if(rank == 0) printf("\tTransfer size (B): %10" PRIu64 ", Transfer Time (s): %15.9f, Bandwidth (GiB/s): %15.9f, Iteration %d\n", num_B, elapsed_time[(j-fix_buff_size)*loop_count+i], num_GB/elapsed_time[(j-fix_buff_size)*loop_count+i], i);
            }
            avg_time_per_transfer /= (double)loop_count;

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
