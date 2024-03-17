#!/bin/bash

NAME="a2a_Nccl_nccl"

export HIPCC_COMPILE_FLAGS_APPEND="--offload-arch=gfx90a $(CC --cray-print-opts=cflags)"
export HIPCC_LINK_FLAGS_APPEND=$(CC --cray-print-opts=libs)

/opt/rocm/bin/hipcc -xhip -o test.o -std=c++14 -I/opt/rocm/include -I/opt/rocm/include/hip --offload-arch=gfx90a -O3 -DMPI_SUPPORT -I/opt/cray/pe/mpich/8.1.27/ofi/crayclang/14.0//include -I/opt/cray/pe/mpich/8.1.27/ofi/crayclang/14.0//include/mpi -c src/$NAME.cpp -DHIP -DOPEN_MPI
/opt/rocm/bin/hipcc -o bin/$NAME -std=c++14 -I/opt/rocm/include -I/opt/rocm/include/hip --offload-arch=gfx90a -O3 -DMPI_SUPPORT -I/opt/cray/pe/mpich/8.1.27/ofi/crayclang/14.0//include -I/opt/cray/pe/mpich/8.1.27/ofi/crayclang/14.0//include/mpi test.o   -L/opt/rocm/lib -lhsa-runtime64 -lrt -pthread -L/opt/cray/pe/mpich/8.1.27/ofi/crayclang/14.0//lib -L/users/demattei/EasyBuild/SW/LUMI-22.08/L/rccl/2.12.7-cpeGNU-22.08/lib/ -lmpi -lrccl
