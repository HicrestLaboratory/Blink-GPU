#include <stdio.h>
#include "mpi.h"
#include "cuda.h"
#include <cuda_runtime.h>
#include <string.h>
#include <unistd.h>
#include <inttypes.h>

#define MPI

#include "../include/error.h"
#include "../include/type.h"
#include "../include/gpu_ops.h"
#include "../include/device_assignment.h"
#include "../include/cmd_util.h"
#include "../include/prints.h"

#ifdef MPIX_CUDA_AWARE_SUPPORT
/* Needed for MPIX_Query_cuda_support(), below */
#include "mpi-ext.h"
#endif

#include "../include/common.h"

#define MYBENCH_CODE "test"
#define MYIMPL_CODE "kernel"

#define DELAY 10
#define SCALE 10
#define PERCENT 90

__global__ 
void kernel( float *x, int niter ) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x ;
	
	#pragma unroll
	for ( int i=0; i<niter ; i++) {
		x[tid] = x[tid] * 2 + 2 ;
		x[tid] = x[tid] / 2 - 1 ;
	}
}

int main ( int argc, const char **argv ) {
	int nblocks, nthreads, nsize;
	float *d_x;
	struct cudaDeviceProp prop;
	cudaGetDeviceProperties( &prop, 0 );
	nblocks = prop.multiProcessorCount * PERCENT;
	nthreads = prop.maxThreadsPerBlock;
	if ( nblocks < 1) nblocks = 1;
	nsize = nblocks * nthreads;
	
	cudaMalloc(&d_x, nsize*sizeof(float));

#ifdef PICODCGMI
    PICODCGMI_START( 0 , 0 , 0 )
#endif

	kernel<<<nblocks, nthreads >>>(d_x , DELAY*SCALE ) ;
	cudaDeviceSynchronize();
	usleep(DELAY*1000);

#ifdef PICODCGMI
    PICODCGMI_STOP( 0 )
#endif

	return(0);
}

