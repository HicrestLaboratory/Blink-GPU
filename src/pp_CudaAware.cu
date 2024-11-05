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
#include "../include/cmd_util.h"
#include "../include/prints.h"

#ifdef MPIX_CUDA_AWARE_SUPPORT
/* Needed for MPIX_Query_cuda_support(), below */
#include "mpi-ext.h"
#endif

#include "../include/common.h"

#define MYBENCH_CODE "pp"
#define MYIMPL_CODE "CudaAware"

#define BUFF_CYCLE 31
#define LOOP_COUNT 50

#define WARM_UP 5

#ifndef MYNCCL
#define NCCL_FLAG 0
#define TTYPE double
#define MPI_TTYPE MPI_DOUBLE
#else
#define NCCL_FLAG 1
#define TTYPE float
#define MPI_TTYPE MPI_FLOAT
#endif

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

#ifdef PICODCGMI
    PICODCGMI_START( fix_buff_size , loop_count , rank )
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    SZTYPE N = define_buffer_len(fix_buff_size);

    int *error, *my_error;
    TTYPE start_time, stop_time;
    cktype *cpu_checks, *gpu_checks;
    TTYPE *elapsed_time, *inner_elapsed_time;
    timers_and_checks_alloc<TTYPE>(buff_cycle, loop_count, &error, &my_error, &cpu_checks, &gpu_checks, &elapsed_time, &inner_elapsed_time);
    if (rank == 0 || rank == rank2) {
        for(int j=fix_buff_size; j<max_j; j++){

            (j!=0) ? (N <<= 1) : (N = 1);
            if (rank == 0) {printf("%i#", j); fflush(stdout);}

            // Allocate memory for A on CPU
            dtype *A, *B;
            cktype my_cpu_check = 0, recv_cpu_check, gpu_check = 0;
            alloc_host_buffers(rank, &A, N*sizeof(dtype), &B, N*sizeof(dtype), ppComm);

            // Initialize all elements of A to 1*(rank+1) and B to 0.0
            INIT_HOST_BUFFER(A, N, 1U * (rank+1))
            INIT_HOST_BUFFER(B, N, 0U )

            dtype *d_A, *d_B;
            alloc_device_buffers(A, &d_A, N*sizeof(dtype), B, &d_B, N*sizeof(dtype));
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
                    MPI_Send(d_A, N, MPI_dtype, rank2, tag1, MPI_COMM_WORLD);
                    MPI_Recv(d_B, N, MPI_dtype, rank2, tag2, MPI_COMM_WORLD, &stat);
                }
                else if(rank == rank2){
                    MPI_Recv(d_B, N, MPI_dtype, 0, tag1, MPI_COMM_WORLD, &stat);
                    MPI_Send(d_A, N, MPI_dtype, 0, tag2, MPI_COMM_WORLD);
                }

                stop_time = MPI_Wtime();
                if (i>0) inner_elapsed_time[(j-fix_buff_size)*loop_count+i-1] = stop_time - start_time;

                if (rank == 0) {printf("%%"); fflush(stdout);}
            }
            if (rank == 0) {printf("#\n"); fflush(stdout);}




            share_check_vectors(rank, 0, rank2, d_B, N, &my_cpu_check, &recv_cpu_check, &gpu_check, &(gpu_checks[j]), &(cpu_checks[j]), &(my_error[j]));

            free_device_buffers(d_A, d_B);
            free_host_buffers(A, B);
        }

        N = define_buffer_len(fix_buff_size);

        MPI_Allreduce(my_error, error, buff_cycle, MPI_INT, MPI_MAX, ppComm);
        //MPI_Allreduce(inner_elapsed_time, elapsed_time, buff_cycle*loop_count, MPI_TTYPE, MPI_MAX, firstsenderComm);
        memcpy(elapsed_time, inner_elapsed_time, buff_cycle*loop_count*sizeof(TTYPE)); // No need to do allreduce, there is only one rank in firstsenderComm

        print_times<TTYPE>(rank, N, fix_buff_size, max_j, loop_count, elapsed_time, error, NCCL_FLAG);

        print_errors(rank, buff_cycle, fix_buff_size, max_j, cpu_checks, gpu_checks);
    }

#ifdef PICODCGMI
    PICODCGMI_STOP( rank )
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    free(error);
    free(my_error);
    free(cpu_checks);
    free(gpu_checks);
    free(elapsed_time);
    free(inner_elapsed_time);
    MPI_Finalize();
    return(0);
}
