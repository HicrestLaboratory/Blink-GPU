#pragma once

#include "common.h"
#include "type.h"
#include "cmd_util.h"

struct RecordsStruct {

    // ---------- For buffers ----------
    SZTYPE N; // This rapresent a loop variable for the current buffer size
    int niter;
    int nrepetitions;

    // ---------- For timers ----------
    double  stop_time;
    double  start_time;
    double *elapsed_time;
    double *inner_elapsed_time;

    // ---------- For correctness ----------
    bool    *check_results;
    cktype *sendSideChecks;
    cktype *recvSideChecks;
    CommunicatioType  type;

    // ---------- For statistics ----------
    SZTYPE num_B;
    double num_GB;

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

    void print_iter_info(int rank, FILE *fp = stdout) {
        if(rank == 0){
            fprintf(fp, "Record structure info: niter = %d, nrepetitions = %d, current_N = %lu\n", niter, nrepetitions, N);
        }
        fflush(fp);
        MPI_Barrier(MPI_COMM_WORLD);
    }


    // ---------- For timers ----------
    void init_timers(void) {
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

    void time_maxreduce (MPI_Comm comm) {
        MPI_Allreduce(inner_elapsed_time, elapsed_time, niter*nrepetitions, MPI_DOUBLE, MPI_MAX, comm);
    }

    void free_timers(void) {
        free(elapsed_time);
        free(inner_elapsed_time);
    }

    // ---------- For correctness ----------
    void init_correctness(CommunicatioType t) {
        type = t;
        sendSideChecks = (cktype*)malloc(sizeof(cktype)*niter);
        recvSideChecks = (cktype*)malloc(sizeof(cktype)*niter);
        check_results  = (bool*)  malloc(sizeof(cktype)*niter);
    }

    bool correctness_check(MPI_Comm comm) {
        MPI_Allreduce(MPI_IN_PLACE, sendSideChecks, niter, MPI_cktype, MPI_SUM, comm);
        if ((type != ALLGATHER) && (type != ALLREDUCE) && (type != BCAST))
            MPI_Allreduce(MPI_IN_PLACE, recvSideChecks, niter, MPI_cktype, MPI_SUM, comm);

        bool overall_check = true;
        for (int i=0; i<niter; i++) {
            check_results[i] = (sendSideChecks[i] == recvSideChecks[i]);
            overall_check   &= check_results[i];
        }
        return(overall_check);
    }

    void free_correctness(void) {
        free(sendSideChecks);
        free(recvSideChecks);
    }

    // ---------- For statistics ----------
    void compute_numB (CommunicatioType type, int size) {
        switch (type) {
            case ALL2ALL:
                num_B = sizeof(dtype)*(N)*(size-1);
                break;

            case ALLREDUCE:
                num_B = sizeof(dtype)*N*((size-1)/(float)size)*2;
                break;

            case ALLGATHER:
                num_B = UINT64_MAX; // NOTE: Place-holder
                break;

            case SCATTER:
                num_B = sizeof(dtype)*N*(size-1); // NOTE: To Check
                break;

            case GATHER:
                num_B = UINT64_MAX; // NOTE: Place-holder
                break;

            case SENDRECV:
                num_B = sizeof(dtype)*N;
                break;

            case BCAST:
                num_B = sizeof(dtype)*N*(size-1); // NOTE: To Check
                break;

            default:
                num_B = UINT64_MAX;
                break;
        }

        SZTYPE B_in_GB = 1 << 30;
        num_GB = (double)num_B / (double)B_in_GB;
        // fprintf(stdout, "num_B: %lu, num_GB: %lu\n", num_B, num_GB);
    }

    // ---------- Overall ----------
    void init(Config *config, CommunicatioType t) {
        init_iter_var(config);
        init_correctness(t);
        init_timers();
    }

};
