// Copyright 2009-2018 Sandia Corporation. Under the terms
// of Contract DE-NA0003525 with Sandia Corporation, the U.S.
// Government retains certain rights in this software.
//
// Copyright (c) 2009-2018, Sandia Corporation
// All rights reserved.
//
// Portions are copyright of other developers:
// See the file CONTRIBUTORS.TXT in the top level directory
// the distribution for more information.
//
// This file is part of the SST software package. For license
// information, see the LICENSE file in the top level directory of the
// distribution.

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "../../include/experiment_utils.h"
#include "../../include/debug_utils.h"
#include "../../include/helper_cuda.h"

#define BLK_SIZE 256
#define GRD_SIZE 4
#define TID_DIGITS 10000

#define dtype float
#define MPI_dtype MPI_FLOAT

static int stringCmp( const void *a, const void *b) {
     return strcmp((const char*)a,(const char*)b);

}

int  assignDeviceToProcess(MPI_Comm *nodeComm, int *nnodes, int *mynodeid)
{
      char     host_name[MPI_MAX_PROCESSOR_NAME];
      char (*host_names)[MPI_MAX_PROCESSOR_NAME];

      int myrank;
      int gpu_per_node;
      int n, namelen, color, rank, nprocs;
      size_t bytes;

      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
      MPI_Get_processor_name(host_name,&namelen);

      bytes = nprocs * sizeof(char[MPI_MAX_PROCESSOR_NAME]);
      host_names = (char (*)[MPI_MAX_PROCESSOR_NAME]) malloc(bytes);

      strcpy(host_names[rank], host_name);

      for (n=0; n<nprocs; n++)
      {
       MPI_Bcast(&(host_names[n]),MPI_MAX_PROCESSOR_NAME, MPI_CHAR,n, MPI_COMM_WORLD);
      }


      qsort(host_names, nprocs, sizeof(char[MPI_MAX_PROCESSOR_NAME]), stringCmp);

      color = 0;

      for (n=0; n<nprocs; n++)
      {
        if(n>0&&strcmp(host_names[n-1], host_names[n])) color++;
        if(strcmp(host_name, host_names[n]) == 0) break;
      }

      MPI_Comm_split(MPI_COMM_WORLD, color, 0, nodeComm);

      MPI_Comm_rank(*nodeComm, &myrank);
      MPI_Comm_size(*nodeComm, &gpu_per_node);

      MPI_Allreduce(&color, nnodes, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
      (*mynodeid) = color;
      (*nnodes) ++;

//      printf ("Assigning device %d  to process on node %s rank %d\n",*myrank,  host_name, rank );
      /* Assign device to MPI process, initialize BLAS and probe device properties */
      //cudaSetDevice(*myrank);
      return myrank;
}

__global__
void init_kernel(int n, dtype *input, int rank) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;

  dtype floattid = tid/(dtype)TID_DIGITS;
  dtype val_coord = rank + floattid;
  if (tid < n)
      input[tid] = (dtype)val_coord;

}

__global__
void test_kernel(int n, int ninputs, size_t *sizes, dtype **inputs, dtype *output) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  dtype tmp = 0.0;

  if (tid < n) {
    for (int i=0; i<ninputs; i++)
      if (tid < sizes[i])
        tmp += inputs[i][tid];
  }
  output[tid] = tmp;

}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int me = -1, mynode = -1;
  int world = -1, nnodes = -1;
  double timeTakenMPI = 0.0, timeTakenCUDA = 0.0;

  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &me);

  int repeats = 1;
  int msgsize = 1024;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "-iterations") == 0) {
      repeats = atoi(argv[i + 1]);
      i++;
    } else if (strcmp(argv[i], "-msgsize") == 0) {
      msgsize = atoi(argv[i + 1]);
      i++;
    } else {
      if (0 == me) {
        fprintf(stderr, "Unknown option: %s\n", argv[i]);
        exit(-1);
      }
    }
  }

  if (0 == me) {
    printf("# Incast Communication Benchmark\n");
    printf("# Info:\n");
    printf("# Incast Ranks:     %8d\n", (world - 1));
    printf("# Iterations:       %8d\n", repeats);
    printf("# Message Size:     %8d\n", msgsize);
  }

  // ----------------------------------------------------------------------------------------------
  // PICO asign device
  int dev, deviceCount = 0;

  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (me == 0) {
    printf("#\n");
    printf("# Number of GPUs: %d\n", deviceCount);
  }

  MPI_Comm nodeComm;
  dev = assignDeviceToProcess(&nodeComm, &nnodes, &mynode);
  cudaSetDevice(dev);

  int mynodeid = -1, mynodesize = -1;
  MPI_Comm_rank(nodeComm, &mynodeid);
  MPI_Comm_size(nodeComm, &mynodesize);

  MPI_ALL_PRINT( fprintf(fp, "mydev is %d, mynode is %d, nnodes are %d, mynodeid is %d and mynodesize is %d\n", dev, mynode, nnodes, mynodeid, mynodesize); )

  DBG_CHECK(1)

  MPI_Barrier(MPI_COMM_WORLD);
  for (int i=0; i<world; i++) {
    if (me == i)
      printf("#\tMPI process %d has device %d\n", me, dev);
    MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  // ----------------------------------------------------------------------------------------------

  dtype* recvBuffer = NULL;
  dtype* sendBuffer = NULL;

  // ---------------------------------------------------------------------------------------
  dtype *dev_recvBuffer, *dev_sendBuffer;
  // ---------------------------------------------------------------------------------------

  // ---------------------------------------------------------------------------------------
  if (me == (world - 1)) {
    checkCudaErrors( cudaMalloc(&dev_recvBuffer, sizeof(dtype) * msgsize * (world - 1)) );
    checkCudaErrors( cudaMemset(dev_recvBuffer, 0, sizeof(dtype) * msgsize * (world - 1)) );
  } else {
    checkCudaErrors( cudaMalloc(&dev_sendBuffer, sizeof(dtype) * msgsize) );

    {
        dim3 block_size(BLK_SIZE, 1, 1);
        dim3 grid_size(GRD_SIZE, 1, 1);
        init_kernel<<<grid_size, block_size>>>(msgsize, dev_sendBuffer, me);
        checkCudaErrors( cudaDeviceSynchronize() );
    }
  }
  // ---------------------------------------------------------------------------------------

  // ---------------------------------------
  {
    float *tmp0;
    srand((unsigned int)time(NULL));
    int x = rand() % (GRD_SIZE*BLK_SIZE);
    tmp0 = (dtype*)malloc(sizeof(dtype)*(msgsize));
    for (int i=0; i<(GRD_SIZE*BLK_SIZE); i++) tmp0[i] = 0.0;
    if (me != world-1)
        checkCudaErrors( cudaMemcpy(tmp0, dev_sendBuffer, msgsize*sizeof(dtype), cudaMemcpyDeviceToHost) );
    else
        checkCudaErrors( cudaMemcpy(tmp0, dev_recvBuffer, msgsize*sizeof(dtype), cudaMemcpyDeviceToHost) ); // NOTE: dev_recvBuffer is longer then msgsize
    checkCudaErrors( cudaDeviceSynchronize() );

    MPI_ALL_PRINT(
      fprintf(fp, "extracted tid = %d\n", x);
      fprintf(fp, "dev_sendBuffer = %6.4f\n", tmp0[x]);
    )
    free(tmp0);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  // ---------------------------------------

  MPI_Request* requests = (MPI_Request*) malloc(sizeof(MPI_Request) * (world - 1));
  MPI_Status* status = (MPI_Status*)malloc(sizeof(MPI_Status) * (world - 1));

  struct timeval start;
  struct timeval end;

  INIT_EXPS
  TIMER_DEF(0);
  TIMER_DEF(1);
  SET_EXPERIMENT_NAME(0, "incast")
  SET_EXPERIMENT_TYPE(0, "nvlink")
  SET_EXPERIMENT(0, "CUDA")

  SET_EXPERIMENT_NAME(1, "incast")
  SET_EXPERIMENT_TYPE(1, "nvlink")
  SET_EXPERIMENT(1, "MPI")

  SET_EXPERIMENT_NAME(2, "incast")
  SET_EXPERIMENT_TYPE(2, "nvlink")
  SET_EXPERIMENT(2, "TOTAL")

  if (nnodes > 1) {
    SET_EXPERIMENT_LAYOUT(0, "interNodes")
    SET_EXPERIMENT_LAYOUT(1, "interNodes")
    SET_EXPERIMENT_LAYOUT(2, "interNodes")
  } else {
    SET_EXPERIMENT_LAYOUT(0, "intraNode")
    SET_EXPERIMENT_LAYOUT(1, "intraNode")
    SET_EXPERIMENT_LAYOUT(2, "intraNode")
  }

  MPI_Barrier(MPI_COMM_WORLD);
  gettimeofday(&start, NULL);

  if (nnodes == 1) {
    // ---------------------------------------
    // PICO enable peers
    STR_COLL_DEF
    STR_COLL_INIT

    // Pick all the devices that can access each other's memory for this test
    // Keep in mind that CUDA has minimal support for fork() without a
    // corresponding exec() in the child process, but in this case our
    // spawnProcess will always exec, so no need to worry.
    cudaDeviceProp prop;
    int allPeers = 1, myIPC = 1, allIPC;
    checkCudaErrors(cudaGetDeviceProperties(&prop, dev));

    int* canAccesPeer = (int*) malloc(sizeof(int)*deviceCount*deviceCount);
    for (int i = 0; i < deviceCount*deviceCount; i++) canAccesPeer[i] = 0;

    // CUDA IPC is only supported on devices with unified addressing
    if (!prop.unifiedAddressing) {
      STR_COLL_APPEND( sprintf(str_coll.buff, "Device %d does not support unified addressing, skipping...\n", dev); )
      myIPC = 0;
    } else {
      STR_COLL_APPEND( sprintf(str_coll.buff, "Device %d support unified addressing\n", dev); )
    }
    // This sample requires two processes accessing each device, so we need
    // to ensure exclusive or prohibited mode is not set
    if (prop.computeMode != cudaComputeModeDefault) {
      STR_COLL_APPEND( sprintf(str_coll.buff, "Device %d is in an unsupported compute mode for this sample\n", dev); )
      myIPC = 0;
    } else {
      STR_COLL_APPEND( sprintf(str_coll.buff, "Device %d is in a supported compute mode for this sample\n", dev); )
    }

    MPI_Allreduce(&myIPC, &allIPC, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!allIPC) {
      MPI_ALL_PRINT( fprintf(fp, "%s", STR_COLL_GIVE); )
      exit(__LINE__);
    }

    if (me == 0) {
      for (int i = 0; i < deviceCount; i++) {
        for (int j = 0; j < deviceCount; j++) {
          if (j != i) {
            int canAccessPeerIJ, canAccessPeerJI;
            checkCudaErrors( cudaDeviceCanAccessPeer(&canAccessPeerJI, j, i) );
            checkCudaErrors( cudaDeviceCanAccessPeer(&canAccessPeerIJ, i, j) );

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
        if (j != dev) {
          checkCudaErrors(cudaDeviceEnablePeerAccess(j, 0));
          STR_COLL_APPEND( sprintf(str_coll.buff, "Enabled access from device %d to device %d\n", dev, j); )
        }
      }
    } else {
      if (me == 0) printf(str_coll.buff, "CUDA IPC is not supported by all the node's GPUs\n");
    }

    MPI_ALL_PRINT(
      fprintf(fp, "%s", STR_COLL_GIVE);
      FPRINT_MATRIX(fp, canAccesPeer, deviceCount, deviceCount)
    )
    STR_COLL_FREE
    MPI_Barrier(MPI_COMM_WORLD);
    // ---------------------------------------

    dtype *peerBuffer[world-1];
    cudaEvent_t sendEvent, recvEvent[world-1];
    cudaIpcMemHandle_t sendHandle, recvHandle[world-1];
    cudaIpcEventHandle_t sendEventHandle, recvEventHandle[world-1];

    for (int i = 0; i < repeats; ++i) {
      TIMER_START(0);
      if (me != world-1) {
        checkCudaErrors( cudaIpcGetMemHandle((cudaIpcMemHandle_t*)&sendHandle, dev_sendBuffer) );
        checkCudaErrors( cudaEventCreate(&sendEvent, cudaEventDisableTiming | cudaEventInterprocess) );
        checkCudaErrors( cudaIpcGetEventHandle((cudaIpcEventHandle_t*)&sendEventHandle, sendEvent) );
      }
      MPI_Barrier(MPI_COMM_WORLD);
      TIMER_STOP(0);
      timeTakenCUDA += TIMER_ELAPSED(0);


      TIMER_START(1);
      if (me == (world - 1)) {
        for (int r = 0; r < world - 1; r++) {
          MPI_Irecv(&recvHandle[r], sizeof(cudaIpcMemHandle_t), MPI_BYTE, r, 1000, MPI_COMM_WORLD, &requests[r]);
          MPI_Irecv(&recvEventHandle[r], sizeof(cudaIpcEventHandle_t), MPI_BYTE, r, 1000, MPI_COMM_WORLD, &requests[r]);
        }
      }

      if (me != (world - 1)) {
        MPI_Send(&sendHandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, (world - 1), 1000, MPI_COMM_WORLD);
        MPI_Send(&sendEventHandle, sizeof(cudaIpcEventHandle_t), MPI_BYTE, (world - 1), 1000, MPI_COMM_WORLD);
      } else {
        MPI_Waitall((world - 1), requests, status);
      }
      MPI_Barrier(MPI_COMM_WORLD);
      TIMER_STOP(1);
      timeTakenMPI += TIMER_ELAPSED(1);


      TIMER_START(0);
      if (me == world-1) {
        for (int r = 0; r < world - 1; r++) {
          checkCudaErrors( cudaIpcOpenMemHandle((void**)&(peerBuffer[r]), *(cudaIpcMemHandle_t*)&(recvHandle[r]), cudaIpcMemLazyEnablePeerAccess) );
          checkCudaErrors( cudaIpcOpenEventHandle(&recvEvent[r], *(cudaIpcEventHandle_t *)&recvEventHandle[r]) );

          checkCudaErrors( cudaMemcpy(&dev_recvBuffer[r * msgsize], peerBuffer[r], sizeof(dtype)*msgsize, cudaMemcpyDeviceToDevice) );
        }
      }
      TIMER_STOP(0);
      timeTakenCUDA += TIMER_ELAPSED(0);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    gettimeofday(&end, NULL);
    ADD_TIME_EXPERIMENT(0, timeTakenCUDA)
    ADD_TIME_EXPERIMENT(1, timeTakenMPI)
    ADD_TIME_EXPERIMENT(2, timeTakenCUDA + timeTakenMPI)

    // ---------------------------------------
    // PICO disable peers
    MPI_Barrier(MPI_COMM_WORLD);
    if (me == world-1) {
      for (int r = 0; r < world-1; r++) {
        checkCudaErrors( cudaIpcCloseMemHandle(peerBuffer[r]) );
        checkCudaErrors( cudaEventDestroy(recvEvent[r]) );
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    for (int j = 0; j < deviceCount; j++) {
      if (j != dev) {
        checkCudaErrors(cudaDeviceDisablePeerAccess(j));
        printf("[%d] Disable access from device %d to device %d\n", me, dev, j);
      }
    }
  } else {
    if (0 == me) printf("The NVLINK version is only implemented for intraNode communication\n");
  }
  // ---------------------------------------

  // ---------------------------------------
  fflush(stdout);
  MPI_Barrier(MPI_COMM_WORLD);
  {
    if (me == (world-1)) {
        size_t test_sizes[1], *dev_test_sizes;
        dtype *test_vector[1], **dev_test_vector;
        srand((unsigned int)time(NULL));
        int x = rand() % (GRD_SIZE*BLK_SIZE);
        int maxSize = msgsize*(world-1);

        dtype *dev_checkVector, *checkVector;
        checkVector = (dtype*) malloc(sizeof(dtype)*maxSize);
        checkCudaErrors( cudaMalloc(&dev_checkVector,   sizeof(dtype) * maxSize) );
        checkCudaErrors( cudaMemset(dev_checkVector, 0, sizeof(dtype) * maxSize) );

        test_sizes[0] = maxSize;
        test_vector[0] = dev_recvBuffer;

        checkCudaErrors( cudaMalloc(&dev_test_sizes,  sizeof(size_t) * 1) );
        checkCudaErrors( cudaMalloc(&dev_test_vector, sizeof(dtype*) * 1) );
        checkCudaErrors( cudaMemcpy(dev_test_sizes,  test_sizes,  sizeof(size_t) * 1, cudaMemcpyHostToDevice) );
        checkCudaErrors( cudaMemcpy(dev_test_vector, test_vector, sizeof(dtype*) * 1, cudaMemcpyHostToDevice) );

        {
        dim3 block_size(BLK_SIZE, 1, 1);
        dim3 grid_size(GRD_SIZE * (world-1), 1, 1); // NOTE the receved buffer is longer then the standard ones
        test_kernel<<<grid_size, block_size>>>(maxSize, 1, dev_test_sizes, dev_test_vector, dev_checkVector);
        checkCudaErrors( cudaDeviceSynchronize() );
        }
        checkCudaErrors( cudaMemcpy(checkVector, dev_checkVector, maxSize*sizeof(dtype), cudaMemcpyDeviceToHost) );
        checkCudaErrors( cudaDeviceSynchronize() );

        printf("------ Receved ------\n");
        printf("world = %d\n", world);
        printf("extracted tid = %d\n", x);
        for (int i=0; i<(world-1); i++)
            printf("From process %d checkVector[%d] = %6.4f\n", i, i*(GRD_SIZE*BLK_SIZE) + x, checkVector[i*(GRD_SIZE*BLK_SIZE) + x]);
        printf("---------------------\n");
        fflush(stdout);

        checkCudaErrors( cudaFree(dev_test_vector) );
        checkCudaErrors( cudaFree(dev_checkVector) );
        checkCudaErrors( cudaFree(dev_test_sizes) );
        free(checkVector);
    }
  }
  fflush(stdout);
  MPI_Barrier(MPI_COMM_WORLD);
  // ---------------------------------------

  if (recvBuffer != NULL) free(recvBuffer);
  if (sendBuffer != NULL) free(sendBuffer);

  if ((world - 1) == me) {
    const double timeTaken =
        (((double)end.tv_sec) + (double)end.tv_usec * 1.0e-6) -
        (((double)start.tv_sec) + (double)start.tv_usec * 1.0e-6);
    const double msgsRecv = ((double)(repeats * (world - 1)));
    const double dataRecv =
        (((double)repeats) * ((double)msgsize) * ((double)(world - 1))) /
        (1024.0 * 1024.0);

    printf("# Statistics:\n");
    printf("# %20s %20s %20s\n", "Time Taken", "Msgs/s", "MB/s");
    printf("  %20.6f %20f %20.6f\n", timeTaken, msgsRecv / timeTaken,
           dataRecv / timeTaken);
  }

  fflush(stdout);
  MPI_Barrier(MPI_COMM_WORLD);
  if (me == 0) PRINT_EXPARIMENT_STATS

  MPI_Finalize();
}