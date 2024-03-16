#pragma once

#include <sched.h>
#include "mpi.h"

#ifndef SKIPCPUAFFINITY
/*
 * Check CPU affniity and print, this is needed to be called after MPI init
 */
void check_cpu_and_gpu_affinity (int dev_id) {
    unsigned int cpu_id, node_id, ret;
    
    ret = getcpu(&cpu_id, &node_id);
    if (0 != ret)
    {
        printf("Get cpu affinity failed\n");
        return;
    }

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    unsigned int* all_cpu_id = (unsigned int *) malloc(sizeof(unsigned int)*size);
    unsigned int* all_node_id = (unsigned int *) malloc(sizeof(unsigned int)*size);
    all_cpu_id[rank] = cpu_id;
    all_node_id[rank] = node_id;

    for (int root_rank = 0; root_rank < size; root_rank++)
    {
        MPI_Bcast(all_cpu_id+root_rank, 1, MPI_INT, root_rank, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(all_node_id+root_rank, 1, MPI_INT, root_rank, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Collect device id
    int* all_dev_id = (int *) malloc(sizeof(int)*size);
    all_dev_id[rank] = dev_id;

    for (int root_rank = 0; root_rank < size; root_rank++)
    {
        MPI_Bcast(all_dev_id+root_rank, 1, MPI_INT, root_rank, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Collect node name'
    int namelen;
    size_t bytes;
    char     host_name[MPI_MAX_PROCESSOR_NAME];
    char (*host_names)[MPI_MAX_PROCESSOR_NAME];
    bytes = size * sizeof(char[MPI_MAX_PROCESSOR_NAME]);
    host_names = (char (*)[MPI_MAX_PROCESSOR_NAME]) malloc(bytes);

    MPI_Get_processor_name(host_name,&namelen);
    host_names = (char (*)[MPI_MAX_PROCESSOR_NAME]) malloc(bytes);
    strcpy(host_names[rank], host_name);
    for (int n=0; n<size; n++)
    {
        MPI_Bcast(&(host_names[n]), MPI_MAX_PROCESSOR_NAME, MPI_CHAR, n, MPI_COMM_WORLD);
    }

    // Print at node 0
    if (0 == rank) {
        for (int root_rank = 0; root_rank < size; root_rank++){
            printf("Machine name: %s, Rank: %d, Node_id: %u, CPU_id: %u, Device_id: %u.\n",host_names[root_rank], root_rank, all_node_id[root_rank], all_cpu_id[root_rank], all_dev_id[root_rank]);
        }
    }
 
    free(all_cpu_id);
    free(all_node_id);
    free(all_dev_id);
    free(host_names);

    return;
}
#endif
