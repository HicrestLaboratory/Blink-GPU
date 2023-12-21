#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>

#include "../include/debug_utils.h"

#define MULTI_COMM


#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


static uint64_t getHostHash(const char* string) {
  // Based on DJB2a, result = result * 33 ^ char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}


static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
        return;
    }
  }
}

__global__
void sum_kernel(int n, char *input, int *output) {

  int tid = blockIdx.x*blockDim.x + threadIdx.x;

  if (tid < n)
    output[tid] = input[tid] + 10.0;
}


int main(int argc, char* argv[])
{
  int size = 32*1024*1024;


  int myRank, nRanks, localRank = 0;


  //initializing MPI
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));


  //calculating localRank based on hostname which is used in selecting a GPU
  uint64_t hostHashs[nRanks];
  char hostname[1024];
  getHostName(hostname, 1024);
  hostHashs[myRank] = getHostHash(hostname);
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  for (int p=0; p<nRanks; p++) {
     if (p == myRank) break;
     if (hostHashs[p] == hostHashs[myRank]) localRank++;
  }


  ncclUniqueId id;
  ncclComm_t comm;
  float *sendbuff, *recvbuff;
  float *h_sendbuff, *h_recvbuff;
  cudaStream_t s;

  h_sendbuff = (float*)malloc(sizeof(float)*size);
  h_recvbuff = (float*)malloc(sizeof(float)*size);

  int scale = 10;
  for (int i=0; i<myRank; i++)
      scale *= 10;

  for (int i=0; i<size; i++) {
    h_sendbuff[i] = 1.0/scale;
    h_recvbuff[i] = 0.0;
  }


  //get NCCL unique ID at rank 0 and broadcast it to all others
#ifndef MULTI_COMM
  if (myRank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
#else
  MPI_Comm sub_comm;
  MPI_Comm_split(MPI_COMM_WORLD, myRank/4, myRank%4, &sub_comm);
  if ((myRank%4) == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, sub_comm));
#endif


  //picking a GPU based on localRank, allocate device buffers
  CUDACHECK(cudaSetDevice(localRank));
  CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
  CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float)));
  CUDACHECK(cudaStreamCreate(&s));

  MPI_ALL_PRINT(
      for (int i=0; i<5; i++)
          fprintf(fp, "%9.8f ", h_sendbuff[i]);
      fprintf(fp, "...\n");
  )

  CUDACHECK( cudaMemcpy(sendbuff, h_sendbuff, sizeof(float)*size, cudaMemcpyHostToDevice) );

  //initializing NCCL
#ifndef MULTI_COMM
  NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));
#else
  NCCLCHECK(ncclCommInitRank(&comm, nRanks/2, id, myRank%4));
#endif

  //communicating using NCCL
  NCCLCHECK(ncclAllReduce((const void*)sendbuff, (void*)recvbuff, size, ncclFloat, ncclSum,
        comm, s));


  //completing NCCL operation by synchronizing on the CUDA stream
  CUDACHECK(cudaStreamSynchronize(s));

  CUDACHECK( cudaMemcpy(h_recvbuff, recvbuff, sizeof(float)*size, cudaMemcpyDeviceToHost) );

  MPI_ALL_PRINT(
      for (int i=0; i<5; i++)
          fprintf(fp, "%9.8f ", h_recvbuff[i]);
      fprintf(fp, "...\n");
  )

  //free device buffers
  CUDACHECK(cudaFree(sendbuff));
  CUDACHECK(cudaFree(recvbuff));


  //finalizing NCCL
  ncclCommDestroy(comm);


  //finalizing MPI
  MPICHECK(MPI_Finalize());


  printf("[MPI Rank %d] Success \n", myRank);
  return 0;
}
