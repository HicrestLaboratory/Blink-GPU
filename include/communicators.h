#pragma once

/* Example:
 * int size, rank, namelen;
 * char host_name[MPI_MAX_PROCESSOR_NAME];
 * MY_MPI_INIT(size, rank, namelen, host_name)
 */

#define MY_MPI_INIT(SZ, RK, NL, HN)         \
    MPI_Init(&argc, &argv);                 \
    MPI_Comm_size(MPI_COMM_WORLD, &(SZ));   \
    MPI_Comm_rank(MPI_COMM_WORLD, &(RK));   \
                                            \
    MPI_Get_processor_name(HN, &(NL));      \
    MPI_Barrier(MPI_COMM_WORLD);            \
                                            \
    {                                                                                           \
        MPI_Status stat;\
        char *all_hostnames[SZ];                                                                \
        if ((RK) == 0) for(int i=0; i<SZ; i++) all_hostnames[i] = (char*)malloc(sizeof(char)*(NL)); \
        for(int i=0; i<SZ; i++) { \
            if (RK == i) MPI_Send(HN              , NL, MPI_CHAR, 0, i, MPI_COMM_WORLD); \
            if (RK == 0) MPI_Recv(all_hostnames[i], NL, MPI_CHAR, i, i, MPI_COMM_WORLD, &stat); \
        } \
        if (RK == 0) {                                                                          \
            printf("MPI_COMM_WORLD size is %d\n", SZ);                                          \
            for (int i=0; i<SZ; i++) {                                                          \
                printf("\trank %d was assigned to host %s\n", i, all_hostnames[i]);     \
            }                                                                                   \
        }                                                                                       \
        if ((RK) == 0) for(int i=0; i<SZ; i++) free(all_hostnames[i]);                          \
    }                                                                                           \
    fflush(stdout);                                                                             \
    MPI_Barrier(MPI_COMM_WORLD);                                                                \


