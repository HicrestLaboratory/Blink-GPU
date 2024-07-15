#include <stdio.h>
#include "mpi.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <inttypes.h>

#define MPI

#include "nvToolsExt.h"

#include "../include/error.h"
#include "../include/type.h"
#include "../include/gpu_ops.h"
#include "../include/device_assignment.h"
#include "../include/prints.h"
#include "../include/communicators.h"
#include "../include/common.h"

#ifdef MPIX_CUDA_AWARE_SUPPORT
/* Needed for MPIX_Query_cuda_support(), below */
#include "mpi-ext.h"
#endif

#define BUFF_CYCLE 31
#define LOOP_COUNT 50

#define WARM_UP 5

// ---------------------------------------
void PICO_enable_peer_access(int myrank, int deviceCount, int mydev) {
    // Pick all the devices that can access each other's memory for this test
    // Keep in mind that CUDA has minimal support for fork() without a
    // corresponding exec() in the child process, but in this case our
    // spawnProcess will always exec, so no need to worry.
    cudaDeviceProp prop;
    int allPeers = 1, myIPC = 1, allIPC;
    cudaErrorCheck(cudaGetDeviceProperties(&prop, mydev));

    int* canAccesPeer = (int*) malloc(sizeof(int)*deviceCount*deviceCount);
    for (int i = 0; i < deviceCount*deviceCount; i++) canAccesPeer[i] = 0;

    // CUDA IPC is only supported on devices with unified addressing
    if (!prop.unifiedAddressing) {
      myIPC = 0;
    } else {
    }
    // This sample requires two processes accessing each device, so we need
    // to ensure exclusive or prohibited mode is not set
    if (prop.computeMode != cudaComputeModeDefault) {
      myIPC = 0;
    }

    MPI_Allreduce(&myIPC, &allIPC, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!allIPC) {
      exit(__LINE__);
    }

    if (myrank == 0) {
      for (int i = 0; i < deviceCount; i++) {
        for (int j = 0; j < deviceCount; j++) {
          if (j != i) {
            int canAccessPeerIJ, canAccessPeerJI;
            cudaErrorCheck( cudaDeviceCanAccessPeer(&canAccessPeerJI, j, i) );
            cudaErrorCheck( cudaDeviceCanAccessPeer(&canAccessPeerIJ, i, j) );

            canAccesPeer[i * deviceCount + j] = (canAccessPeerIJ) ? 1 : 0;
            canAccesPeer[j * deviceCount + i] = (canAccessPeerJI) ? 1 : 0;
            if (!canAccessPeerIJ || !canAccessPeerJI) allPeers = 0;
          } else {
            canAccesPeer[i * deviceCount + j] = -1;
          }
        }
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&allPeers, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(canAccesPeer, deviceCount*deviceCount, MPI_INT, 0, MPI_COMM_WORLD);

    if (allPeers) {
      // Enable peers here.  This isn't necessary for IPC, but it will
      // setup the peers for the device.  For systems that only allow 8
      // peers per GPU at a time, this acts to remove devices from CanAccessPeer
      for (int j = 0; j < deviceCount; j++) {
        if (j != mydev) {
          cudaErrorCheck(cudaDeviceEnablePeerAccess(j, 0));
        }
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

void PICO_disable_peer_access(int deviceCount, int mydev){
    MPI_Barrier(MPI_COMM_WORLD);
    for (int j = 0; j < deviceCount; j++) {
      if (j != mydev) {
        cudaErrorCheck(cudaDeviceDisablePeerAccess(j));
      }
    }
}

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

    /* -------------------------------------------------------------------------------------------
        MPI Initialization 
    --------------------------------------------------------------------------------------------*/
        int nnodes, mynode; // tmp
    int size, rank, namelen;
    char host_name[MPI_MAX_PROCESSOR_NAME];
    MY_MPI_INIT(size, rank, namelen, host_name)

    MPI_Status stat;

    // Map MPI ranks to GPUs
    int num_devices = 0;
    cudaErrorCheck( cudaGetDeviceCount(&num_devices) );
    //     cudaErrorCheck( cudaSetDevice(rank % num_devices) );

    MPI_Comm nodeComm;
    int dev = assignDeviceToProcess(&nodeComm, &nnodes, &mynode);
    cudaSetDevice(dev);

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
    if (nnodes > 1) {
        MPI_Allreduce(&mynodesize, &nodesize, sizeof(int), MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (nodesize != mynodesize) {
            fprintf(stderr, "Error at node %d: mynodesize (%d) does not metch with nodesize (%d)\n", rank, mynodesize, nodesize);
            fflush(stderr);
            MPI_Abort(MPI_COMM_WORLD, __LINE__);
        } else {
            if (rank == 0) printf("All the nodes (%d) have the same size (%d)\n", nnodes, nodesize);
            fflush(stdout);
        }
    } else {
        nodesize = mynodesize;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* -------------------------------------------------------------------------------------------
        CUDA AWARE CHECK
    --------------------------------------------------------------------------------------------*/

    cudaAwareCheck();

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


    printf("[%d] DBG check at line %d\n", rank, __LINE__); fflush(stdout);

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


    printf("[%d] DBG check at line %d\n", rank, __LINE__); fflush(stdout);

    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    /* -------------------------------------------------------------------------------------------
        MPI Initialize Peer-to-peer communicators
    --------------------------------------------------------------------------------------------*/

    int my_peer = -1;
    if ((mynode == 0 || mynode == nnodes-1) && mynodeid < ncouples)
        my_peer = (mynode == 0) ? (rank + (nnodes-1)*nodesize) : (rank - (nnodes-1)*nodesize);

    printf("[%d] DBG check at line %d\n", rank, __LINE__); fflush(stdout);

    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

     /* -------------------------------------------------------------------------------------------
        Loop from 8 B to 1 GB
    --------------------------------------------------------------------------------------------*/

    PICO_enable_peer_access(mynodeid, nodesize, dev);


    printf("[%d] DBG check at line %d\n", rank, __LINE__); fflush(stdout);

    SZTYPE N;
    if (fix_buff_size<=30) {
        N = 1 << (fix_buff_size - 1);
    } else {
        N = 1 << 30;
        N <<= (fix_buff_size - 31);
    }


    printf("[%d] DBG check at line %d\n", rank, __LINE__); fflush(stdout);

    MPI_Status IPCstat;
    dtype *peerBBuffers[ncouples], *peerAggBuffer;
    cudaEvent_t event;
    cudaIpcMemHandle_t sendBHandle, recvBHandle[ncouples], sendAggHandle, recvAggHandle;

    cudaStream_t Streams[4];
    double start_time, stop_time;
    double *elapsed_time = (double*)malloc(sizeof(double)*buff_cycle*loop_count);
    double *inner_elapsed_time = (double*)malloc(sizeof(double)*buff_cycle*loop_count);
//     if (ppCouples != MPI_COMM_NULL) {
        for(int j=fix_buff_size; j<max_j; j++){

            (j!=0) ? (N <<= 1) : (N = 1);

            for (int k=0; k<4; k++) {cudaErrorCheck(cudaStreamCreate(&Streams[k]));}

            // Allocate memory for A on CPU
            dtype *d_Agg;
            dtype *A, *B;
            alloc_host_buffers(rank, &A, N, &B, N);

            // Initialize all elements of A to 0.0
            INIT_HOST_BUFFER(A, N, 1U * (rank+1))
            INIT_HOST_BUFFER(B, N, 0U)

            dtype *d_A, *d_B;
            alloc_device_buffers(A, &d_A, N, B, &d_B, N);

            if (mynodeid == 0) {
                cudaErrorCheck( cudaMalloc(&d_Agg, N*ncouples*sizeof(dtype)) );
                cudaErrorCheck( cudaMemset(d_Agg, 0U, N*ncouples*sizeof(dtype)) );
            }

            int tag1 = 10;
            int tag2 = 20;
            MPI_Request request[2*ncouples];

            /*

            Implemetantion goes here

            */

            if (rank == 0) {printf("%i#", j); fflush(stdout);}

            PUSH_RANGE("initializeIPC", 0)

            // Generate IPC MemHandle
            cudaErrorCheck( cudaIpcGetMemHandle((cudaIpcMemHandle_t*)&sendBHandle, d_B) );
            if (mynodeid==0) { cudaErrorCheck( cudaIpcGetMemHandle((cudaIpcMemHandle_t*)&sendAggHandle, d_Agg) ); }

            // Share IPC MemHandle
            MPI_Gather(&sendBHandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, &recvBHandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 0, MPI_COMM_WORLD);
            if (mynodeid == 0)
                MPI_Bcast(&sendAggHandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 0, MPI_COMM_WORLD);
            else
                MPI_Bcast(&recvAggHandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 0, MPI_COMM_WORLD);

            // Open MemHandles
            if (mynodeid == 0) {
                for (int i=0; i<ncouples; i++) {
                    if (i != 0) {
                        cudaErrorCheck( cudaIpcOpenMemHandle((void**)&peerBBuffers[i], *(cudaIpcMemHandle_t*)&recvBHandle[i], cudaIpcMemLazyEnablePeerAccess) );
                    } else {
                        peerBBuffers[i] = d_B;
                    }
                }
                peerAggBuffer = d_Agg;
            } else {
                cudaErrorCheck( cudaIpcOpenMemHandle((void**)&peerAggBuffer, *(cudaIpcMemHandle_t*)&recvAggHandle, cudaIpcMemLazyEnablePeerAccess) );
            }
            MPI_Barrier(MPI_COMM_WORLD);
            POP_RANGE

            for(int i=1-(WARM_UP); i<=loop_count; i++){
                MPI_Barrier(MPI_COMM_WORLD);
                start_time = MPI_Wtime();


                for (int k=0; k<2; k++) {

                    PUSH_RANGE("IPCgather", 1)
                    // Aggregate d_A buffers into mynodeid 0 agg buffer
                    if (mynode == k*(nnodes-1)) {
                        cudaErrorCheck( cudaMemcpyAsync(peerAggBuffer + (mynodeid*N), d_A, sizeof(dtype)*N, cudaMemcpyDeviceToDevice, Streams[mynodeid]) );
                        cudaErrorCheck( cudaDeviceSynchronize() );
                    }
                    MPI_Barrier(MPI_COMM_WORLD);
                    POP_RANGE
/*
                    // Out-node communication
                    if (ppOutCouple != MPI_COMM_NULL) {
                        if (mynode == k*(nnodes-1)) {
                            MPI_Send(d_Agg, ncouples*N, MPI_dtype, my_peer, tag1, MPI_COMM_WORLD);
                        } else {
                            MPI_Recv(d_Agg, ncouples*N, MPI_dtype, my_peer, tag1, MPI_COMM_WORLD, &stat);
                        }
                        MPI_Barrier(ppOutCouple);
                    }
                    MPI_Barrier(ppAllNodeCouples);
*/

                    PUSH_RANGE("IPCscatter", 2)
                    // Scatter agg buffer into d_B buffers
                    if (mynode == (1-k)*(nnodes-1) && mynodeid == 0) {
                        for (int i=0; i<ncouples; i++) {
                            cudaErrorCheck( cudaMemcpyAsync(peerBBuffers[i], d_Agg + (i*N), sizeof(dtype)*N, cudaMemcpyDeviceToDevice, Streams[i]) );
                        }
                        cudaErrorCheck( cudaDeviceSynchronize() );
                    }
                    MPI_Barrier(MPI_COMM_WORLD);
                    POP_RANGE
                }

                stop_time = MPI_Wtime();
                if (i>0) inner_elapsed_time[(j-fix_buff_size)*loop_count+i-1] = stop_time - start_time;

                if (rank == 0) {printf("%%"); fflush(stdout);}
            }
            if (rank == 0) {printf("#\n"); fflush(stdout);}

            fflush(stdout);
            if (mynodeid == 0) cudaErrorCheck( cudaFree(d_Agg) );
            cudaErrorCheck( cudaFree(d_A) );
            cudaErrorCheck( cudaFree(d_B) );
            free(A);
            free(B);

            for (int k=0; k<4; k++) {cudaErrorCheck(cudaStreamDestroy(Streams[k]));}
        }

        if (fix_buff_size<=30) {
            N = 1 << (fix_buff_size - 1);
        } else {
            N = 1 << 30;
            N <<= (fix_buff_size - 31);
        }

//         if(ppFirstSenders != MPI_COMM_NULL) {
            MPI_Allreduce(inner_elapsed_time, elapsed_time, buff_cycle*loop_count, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
//         }
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

            if(rank == 0) printf("[Average] Transfer size (B): %10" PRIu64 ", Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f, Error: %d\n", num_B, avg_time_per_transfer, num_GB/avg_time_per_transfer, -1 );
            fflush(stdout);
        }
//     }
    free(elapsed_time);
    free(inner_elapsed_time);
    MPI_Finalize();
    return(0);
}
