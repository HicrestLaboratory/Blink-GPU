#pragma once

#include "common.h"
#include "type.h"
#include "cmd_util.h"

struct RecordsStruct {

    // ---------- For buffers ----------
    SZTYPE N; // This rapresent a loop variable for the current buffer size
    int niter;
    int nrepetitions;
    size_t large_count;

    // ---------- For timers ----------
    double  stop_time;
    double  start_time;
    double *elapsed_time;
    double *inner_elapsed_time;

    // ---------- For buffers ----------
    void init_iter_var(Config *config) {
        if (config->fix_buff_size != 0) {
            niter = 1;

            if (config->fix_buff_size<=30) {
                N = 1 << (config->fix_buff_size - 1);
            } else {
                N = 1 << 30;
                N <<= (config->fix_buff_size - 31);
            }
        } else {
            N = 1;
            niter = config->buff_cycle;
        }

        nrepetitions = config->loop_count;
    }

    void increase_N(void) {
        N <<= 1;
    }

    void update_large_count(int rank) {
        large_count = 0;
        if(N >= 8 && N % 8 == 0){ // Check if I can use 64-bit data types
            large_count = N / 8;
            if (large_count >= ((u_int64_t) (1UL << 32)) - 1) { // If large_count can't be represented on 32 bits
                if(rank == 0){
                    printf("\tTransfer size (B): -1, Transfer Time (s): -1, Bandwidth (GiB/s): -1, Iteration -1\n");
                }
                MPI_Abort(MPI_COMM_WORLD, -1);
            }
        }else{
            if (N >= ((u_int64_t) (1UL << 32)) - 1) { // If N can't be represented on 32 bits
                if(rank == 0){
                    printf("\tTransfer size (B): -1, Transfer Time (s): -1, Bandwidth (GiB/s): -1, Iteration -1\n");
                }
                MPI_Abort(MPI_COMM_WORLD, -1);
            }
        }
    }

    void print_iter_info(int rank, FILE *fp = stdout) {
        if(rank == 0){
            fprintf(fp, "Record structure info: niter = %d, nrepetitions = %d, current_N = %lu\n", niter, nrepetitions, N);
        }
        fflush(fp);
        MPI_Barrier(MPI_COMM_WORLD);
    }


    // ---------- For timers ----------
    void init_timers(Config *config) {
        elapsed_time       = (double*)malloc(sizeof(double)*niter*nrepetitions);
        inner_elapsed_time = (double*)malloc(sizeof(double)*niter*nrepetitions);
    }

    void record_time_start(int current_repetition) {
        start_time = MPI_Wtime();
    }

    void record_time_stop(int current_iter, int current_repetition) {
        stop_time = MPI_Wtime();
        if (current_repetition>0) inner_elapsed_time[(current_iter*nrepetitions)+(current_repetition-1)] = stop_time - start_time;
    }

    void free_timers(void) {
        free(elapsed_time);
        free(inner_elapsed_time);
    }


    // ---------- Overall ----------
    void init(Config *config) {
        init_iter_var(config);
        init_timers(config);
    }

};
