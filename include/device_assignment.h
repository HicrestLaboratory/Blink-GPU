#pragma once

#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "mpi.h"

#ifdef HIP
#include <stdlib.h>
#include <vector>
#include <sstream>
#endif

// GPU assgined for LUMI
int GPU_ASSIGN_SEQUENCE[8] = {5, 3, 7, 1, 4, 2, 6, 0};

static int stringCmp( const void *a, const void *b) {
     return strcmp((const char*)a,(const char*)b);

}

int  assignDeviceToProcess(MPI_Comm *nodeComm, int *nnodes, int *mynodeid)
{
#ifdef MPI
      char     host_name[MPI_MAX_PROCESSOR_NAME];
      char (*host_names)[MPI_MAX_PROCESSOR_NAME];

#else
      char     host_name[20];
#endif
      int myrank;
      int gpu_per_node;
      int n, namelen, color, rank, nprocs;
      size_t bytes;
/*
      if (chkseGPU()<1 && 0) {
        fprintf(stderr, "Invalid GPU Serial number\n");
	exit(EXIT_FAILURE);
      }
*/

#ifdef MPI
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

#else
     //*myrank = 0;
     cudaSetDevice(0);
     return 0;
#endif


#ifdef HIP
      // The mapping is hard-coded in the code for LUMI, the 4GCDs connnected to the network 
      // are assigned first, then the 4 GCDs not connected to the network are assgined. The sequence
      // is the same as NUMA sequence.

      // we first check if the GPU senquence is defined
      const char * GPU_ENV_NAME = "USER_HIP_GPU_MAP";
      const char * gpu_env = getenv(GPU_ENV_NAME);

      if (NULL != gpu_env) {
	  // scan GPU id, I will just use the c++ function
	  std::string gpu_env_s = gpu_env;
	  std::vector<std::string> v;	      
	  std::stringstream ss(gpu_env_s);

	  while(ss.good()) {
	      std::string substr;
	      std::getline(ss, substr, ',');
	      v.push_back(substr);
	  }
	  if (v.size() != gpu_per_node){
	    //error
	    ;
	  }
	  myrank = std::stoi(v[myrank]);
      } 
#endif
      
#ifdef HIP
      if(getenv("GPU_MICROBENCH_EXTERNAL_GPU_ASSIGNMENT") == NULL){
            hipSetDevice(myrank);      
      }else{
            printf("Rank %d ROCR_VISIBLE_DEVICES set to %s SLURM_LOCALID: %s\n", rank, getenv("ROCR_VISIBLE_DEVICES"), getenv("SLURM_LOCALID"));
            fflush(stdout);
            myrank = atoi(getenv("ROCR_VISIBLE_DEVICES"));
            assert(myrank == atoi(getenv("SLURM_LOCALID")));
      }
#else
      if(getenv("GPU_MICROBENCH_EXTERNAL_GPU_ASSIGNMENT") == NULL){
            cudaSetDevice(myrank);           
      }else{
            printf("Rank %d CUDA_VISIBLE_DEVICES set to %s SLURM_LOCALID: %s\n", rank, getenv("CUDA_VISIBLE_DEVICES"), getenv("SLURM_LOCALID"));
            fflush(stdout);
            myrank = atoi(getenv("CUDA_VISIBLE_DEVICES"));
            assert(myrank == atoi(getenv("SLURM_LOCALID")));
      }
#endif
      printf("Assigning device %d to process on node %s rank %d\n", myrank, host_name, rank);
      return myrank;
}
