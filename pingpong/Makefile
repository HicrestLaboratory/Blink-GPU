MPI_HOME  = ??
CUDA_HOME = ??

# CC=${MPI_HOME}/bin/mpicc
CC=nvcc
CFLAGS=-O3 -std=c++14
LIBS=

pingpong: old_pingpong.cu
	$(CC) $(CFLAGS) -o $@ $< $(LIBS) -I$(MPI_HOME)/include -I$(CUDA_HOME)/include -L$(MPI_HOME)/lib -lmpi -L$(CUDA_HOME)/lib64 -lmpi -lcudart -lstdc++ -lm -Wno-deprecated-gpu-targets

clean:
	rm pingpong

