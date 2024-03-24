#include <stdio.h>
#include "mpi.h"

#include <hip/hip_runtime.h>
#include <string.h>
#include <unistd.h>
#include <inttypes.h>

#include "../include/error.h"
#include "../include/type.h"
#include "../include/gpu_ops.h"
#include <chrono>

#define BUFF_CYCLE 31
#define LOOP_COUNT 50

#define WARM_UP 5

void read_line_parameters (int argc, char *argv[], int myrank,
                           int *flag_b, int *flag_l, int *flag_x,
                           int *loop_count, int *buff_cycle, int *fix_buff_size, int *g0, int *g1, int *g2 ) {

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
            if (*loop_count < 0) { // Can be 0 for endless mode (only for a2a and incast)
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
        } else if (strcmp(argv[i], "-g0") == 0) {
            if (i == argc) {
                if (myrank == 0) {
                fprintf(stderr, "Error: specified -x without a value.\n");
                }

                exit(__LINE__);
            }
            *g0 = atoi(argv[i + 1]);

            i++;
        } else if (strcmp(argv[i], "-g1") == 0) {
            if (i == argc) {
                if (myrank == 0) {
                fprintf(stderr, "Error: specified -x without a value.\n");
                }

                exit(__LINE__);
            }
            *g1 = atoi(argv[i + 1]);
            
            i++;
        } else if (strcmp(argv[i], "-g2") == 0) {
            if (i == argc) {
                if (myrank == 0) {
                fprintf(stderr, "Error: specified -x without a value.\n");
                }

                exit(__LINE__);
            }
            *g2 = atoi(argv[i + 1]);
            
            i++;
        } else {
            if (0 == myrank) {
                fprintf(stderr, "Unknown option: %s\n", argv[i]);
            }

            exit(__LINE__);
        }
    }
}


int main(int argc, char *argv[])
{

    int opt;
    int max_j;
    int flag_b = 0;
    int flag_l = 0;
    int flag_x = 0;
    int loop_count = LOOP_COUNT;
    int buff_cycle = BUFF_CYCLE;
    int fix_buff_size = 0;
    int g0=-1, g1=-1, g2=-1;

    // Parse command-line options
    read_line_parameters(argc, argv, 0,
                         &flag_b, &flag_l, &flag_x,
                         &loop_count, &buff_cycle, &fix_buff_size, &g0, &g1, &g2);
    if(flag_x && fix_buff_size >= buff_cycle){buff_cycle = fix_buff_size + 1;}    
    // Print message based on the flags
    printf("Flag b was set with argument: %d\n", buff_cycle);
    printf("Flag l was set with argument: %d\n", loop_count);
    printf("Flag x was set with argument: %d\n", fix_buff_size);

    max_j = (flag_x == 0) ? buff_cycle : (fix_buff_size + 1) ;
    printf("buff_cycle: %d loop_count: %d max_j: %d\n", buff_cycle, loop_count, max_j);
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
    int *my_error_b = (int*)malloc(sizeof(int)*buff_cycle);
    int *my_error_c = (int*)malloc(sizeof(int)*buff_cycle);
    cktype *cpu_checks = (cktype*)malloc(sizeof(cktype)*buff_cycle);
    cktype *gpu_checks_b = (cktype*)malloc(sizeof(cktype)*buff_cycle);
    cktype *gpu_checks_c = (cktype*)malloc(sizeof(cktype)*buff_cycle);
    double *elapsed_time_b = (double*)malloc(sizeof(double)*buff_cycle*loop_count);
    double *elapsed_time_c = (double*)malloc(sizeof(double)*buff_cycle*loop_count);
    double *elapsed_time_total = (double*)malloc(sizeof(double)*buff_cycle*loop_count);
    double *inner_elapsed_time_b = (double*)malloc(sizeof(double)*buff_cycle*loop_count);
    double *inner_elapsed_time_c = (double*)malloc(sizeof(double)*buff_cycle*loop_count);
    double *inner_elapsed_time_total = (double*)malloc(sizeof(double)*buff_cycle*loop_count);

    int gpu_a = g0, gpu_b = g1, gpu_c = g2;
    printf("Use gpu g0 %d, g1 %d, g2 %d \n", gpu_a, gpu_b, gpu_c);
    for(int j=fix_buff_size; j<max_j; j++){
        (j!=0) ? (N <<= 1) : (N = 1);
        printf("%i#", j); fflush(stdout);
       
        dtype *d_A, *d_B, *d_C;
        cktype send_check, recv_check_b, recv_check_c;

        hipSetDevice(gpu_a);
        hipErrorCheck( hipMalloc(&d_A, N*sizeof(dtype)) );
        hipErrorCheck( hipMemset(d_A, 1, N*sizeof(dtype)) );
        gpu_device_reduce(d_A, N, &send_check);
        hipSetDevice(gpu_b);
        hipErrorCheck( hipMalloc(&d_B, N*sizeof(dtype)) );
        hipErrorCheck( hipMemset(d_B, 0, N*sizeof(dtype)) );
        hipSetDevice(gpu_c);
        hipErrorCheck( hipMalloc(&d_C, N*sizeof(dtype)) );
        hipErrorCheck( hipMemset(d_C, 0, N*sizeof(dtype)) );
        
        /*

        Implemetantion goes here

        */

        for(int i=1-(WARM_UP); i<=loop_count; i++){
            hipEvent_t start_b, start_c, stop_b, stop_c;
            hipErrorCheck( hipEventCreate(&start_b) );
            hipErrorCheck( hipEventCreate(&start_c) );
            hipErrorCheck( hipEventCreate(&stop_b) );
            hipErrorCheck( hipEventCreate(&stop_c) );

            hipStream_t stream_b, stream_c;
            hipErrorCheck( hipStreamCreate(&stream_b) );
            hipErrorCheck( hipStreamCreate(&stream_c) );

            float ms_b, ms_c;

            const unsigned long int start_time_us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
            hipEventRecord(start_b, stream_b);
            hipErrorCheck( hipMemcpyAsync(d_B, d_A, N*sizeof(dtype), hipMemcpyDeviceToDevice, stream_b));
            hipEventRecord(stop_b, stream_b);

            hipEventRecord(start_c, stream_c);
            hipErrorCheck( hipMemcpyAsync(d_C, d_A, N*sizeof(dtype), hipMemcpyDeviceToDevice, stream_c));
            hipEventRecord(stop_c, stream_c);
            hipEventSynchronize(stop_b);
            hipEventSynchronize(stop_c);
            hipStreamSynchronize(stream_b);
            hipStreamSynchronize(stream_c);
            hipErrorCheck( hipDeviceSynchronize() );
            const unsigned long int end_time_us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

            hipErrorCheck( hipEventElapsedTime(&ms_b, start_b, stop_b) );
            hipErrorCheck( hipEventElapsedTime(&ms_c, start_c, stop_c) );


            if (i>0) inner_elapsed_time_b[(j-fix_buff_size)*loop_count+i-1] = ms_b/1000.0;
            if (i>0) inner_elapsed_time_c[(j-fix_buff_size)*loop_count+i-1] = ms_c/1000.0;
            if (i>0) inner_elapsed_time_total[(j-fix_buff_size)*loop_count+i-1] = (end_time_us-start_time_us)/1e6;
            printf("%f %f %f\n", ms_b/1000.0, ms_c/1000.0, (end_time_us-start_time_us)/1e6 );

           printf("%%"); fflush(stdout);
        }
        printf("#\n"); fflush(stdout);

        gpu_device_reduce(d_B, N, &recv_check_b);
        gpu_device_reduce(d_C, N, &recv_check_c);

        cpu_checks[j] = send_check;
        gpu_checks_b[j] = recv_check_b;
        gpu_checks_c[j] = recv_check_c;

        my_error_b[j] = abs(gpu_checks_b[j] - cpu_checks[j]);
        my_error_c[j] = abs(gpu_checks_c[j] - cpu_checks[j]);

        hipErrorCheck( hipFree(d_A) );
        hipErrorCheck( hipFree(d_B) );
    }

    if (fix_buff_size<=30) {
        N = 1 << (fix_buff_size - 1);
    } else {
        N = 1 << 30;
        N <<= (fix_buff_size - 31);
    }

    memcpy(elapsed_time_b, inner_elapsed_time_b, buff_cycle*loop_count*sizeof(double));
    memcpy(elapsed_time_c, inner_elapsed_time_c, buff_cycle*loop_count*sizeof(double));
    memcpy(elapsed_time_total, inner_elapsed_time_total, buff_cycle*loop_count*sizeof(double));
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

        double avg_time_per_transfer_b = 0.0, avg_time_per_transfer_c = 0.0, avg_time_per_transfer_total = 0.0;
        for (int i=0; i<loop_count; i++) {
            // We only copy in one direction
            // elapsed_time_b[(j-fix_buff_size)*loop_count+i] /= 2.0;
            // elapsed_time_c[(j-fix_buff_size)*loop_count+i] /= 2.0;
            avg_time_per_transfer_b += elapsed_time_b[(j-fix_buff_size)*loop_count+i];
            avg_time_per_transfer_c += elapsed_time_c[(j-fix_buff_size)*loop_count+i];
            avg_time_per_transfer_total += elapsed_time_total[(j-fix_buff_size)*loop_count+i];
            printf("\tTransfer size (B): %10" PRIu64 ", Transfer Time (s): %15.9f/%15.9f (total: %15.9f), Bandwidth (GB/s): %15.9f/%15.9f (Overall: %15.9f), Iteration %d\n", num_B, elapsed_time_b[(j-fix_buff_size)*loop_count+i], elapsed_time_c[(j-fix_buff_size)*loop_count+i], elapsed_time_total[(j-fix_buff_size)*loop_count+i], num_GB/elapsed_time_b[(j-fix_buff_size)*loop_count+i], num_GB/elapsed_time_c[(j-fix_buff_size)*loop_count+i], num_GB*2/elapsed_time_total[(j-fix_buff_size)*loop_count+i], i);
        }
        avg_time_per_transfer_b /= (double)loop_count;
        avg_time_per_transfer_c /= (double)loop_count;
        avg_time_per_transfer_total /= (double)loop_count;

         printf("[Average] Transfer size (B, MB, GB): %10" PRIu64 ", %4" PRIu64 ", %2" PRIu64 ", Transfer Time (s): %15.9f/%15.9f (Overall: %15.9f), Bandwidth (GB/s): %15.9f/%15.9f (total: %15.9f), Error: %d\ / %d\n", num_B, num_B >> 20, num_B >>30, avg_time_per_transfer_b,  avg_time_per_transfer_c, avg_time_per_transfer_total, num_GB/avg_time_per_transfer_b, num_GB/avg_time_per_transfer_c, num_GB*2/avg_time_per_transfer_total,my_error_b[j], my_error_c[j] );
        fflush(stdout);
    }

    char *s = (char*)malloc(sizeof(char)*(20*buff_cycle + 100));
    sprintf(s, "recv_cpu_check = %u", cpu_checks[0]);
    for (int i=fix_buff_size; i<max_j; i++) {
        sprintf(s+strlen(s), " %10d", cpu_checks[i]);
    }
    sprintf(s+strlen(s), " (for Error)\n");
    printf("%s", s);
    fflush(stdout);

    sprintf(s, "gpu_checks_b = %u", gpu_checks_b[0]);
    for (int i=fix_buff_size; i<max_j; i++) {
        sprintf(s+strlen(s), " %10d", gpu_checks_b[i]);
    }
    sprintf(s, "gpu_checks_c = %u", gpu_checks_b[0]);
    for (int i=fix_buff_size; i<max_j; i++) {
        sprintf(s+strlen(s), " %10d", gpu_checks_c[i]);
    }
    sprintf(s+strlen(s), " (for Error)\n");
    printf("%s", s);
    fflush(stdout);


    free(error);
    free(my_error_b);
    free(my_error_c);
    free(cpu_checks);
    free(gpu_checks_b);
    free(gpu_checks_c);
    free(elapsed_time_b);
    free(elapsed_time_c);
    free(inner_elapsed_time_b);
    free(inner_elapsed_time_c);

    return(0);
}
