#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUDA
#define GPUDIRECT

#ifdef CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#endif

#ifdef NCCL
#include <nccl.h>
#endif

#ifdef GPUDIRECT
#ifdef __cplusplus
extern "C" {
#endif
#include <nv-p2p.h>
#ifdef __cplusplus
}
#endif
#include <builtin_types.h>

// for boundary alignment requirement
#define GPU_BOUND_SHIFT   16
#define GPU_BOUND_SIZE    ((uint64_t)1 << GPU_BOUND_SHIFT)
#define GPU_BOUND_OFFSET  (GPU_BOUND_SIZE-1)
#define GPU_BOUND_MASK    (~GPU_BOUND_OFFSET)

#endif

#define DEBUG 2
#include "../include/debug_utils.h"

#define BLK_SIZE 256
#define GRD_SIZE 4

__global__
void init_kernel(int n, char *input, int scale) {

  int tid = blockIdx.x*blockDim.x + threadIdx.x;

  for (int i=0; i<n; i++) {
    int val_coord = tid * scale;
    if (tid < n)
        input[tid] = (char)val_coord;
  }
}

int main (void) {
  int msgSize = 100;
  unsigned long long int interror = 0ULL;
  double timeTaken = 0.0;
  CUdeviceptr devptrSend, devptrRecv;
  cuMemAlloc ( &devptrSend, msgSize*sizeof(char) );
  cuMemAlloc ( &devptrRecv, msgSize*sizeof(char) );

  char* dev_sendBuffer = (char*) devptrSend;
  char* dev_recvBuffer = (char*) devptrRecv;

  dim3 block_size(BLK_SIZE, 1, 1);
  dim3 grid_size(GRD_SIZE, 1, 1);

  init_kernel<<<grid_size, block_size>>>(msgSize, dev_sendBuffer, 1);
  cudaDeviceSynchronize();
  DBG_CHECK(2)


  unsigned int flag = 1;
  cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, devptrSend);
  DBG_CHECK(2)

  nvidia_p2p_page_table *page_table;
  // do proper alignment, as required by NVIDIA kernel driver
  uint64_t virt_start = devptrSend & GPU_BOUND_MASK;
  size_t pin_size = devptrSend + msgSize*sizeof(char) - virt_start;
  if (msgSize == 0)
      return (42);
  int ret = nvidia_p2p_get_pages(0, 0, virt_start, pin_size, &page_table, NULL, dev_sendBuffer);
  if (ret == 0) {
      printf("Succesfully pinned, page_table can be accessed");
  } else {
      fprintf(stderr, "Pinning failed");
      exit(42);
  }
  DBG_CHECK(2)

  return(0);
}
