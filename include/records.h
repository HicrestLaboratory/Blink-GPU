#pragma once

#include "common.h"
#include "type.h"
#include "cmd_util.h"

struct RecordsStruct {


    SZTYPE N; // This rapresent a loop variable for the current buffer size
    int iter_last;
    int iter_start;
    size_t large_count;

    void init_with_config(Config *config) {
        if (config->fix_buff_size != 0) {
            iter_start = config->fix_buff_size;
            iter_last  = config->max_buff_size;

            if (config->fix_buff_size<=30) {
                N = 1 << (config->fix_buff_size - 1);
            } else {
                N = 1 << 30;
                N <<= (config->fix_buff_size - 31);
            }
        } else {
            N = 1;
            iter_start = 0;
            iter_last  = config->max_buff_size;
        }
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
};
