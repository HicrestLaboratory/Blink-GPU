#! /bin/bash
module pruge
module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm

mkdir bin
mkdir sout

CC -xhip -DHIP -DOPEN_MPI -DPINNED -O3 src-intra-node/concurrent_hipmemcpy.cpp -o bin/concurrent_hipmemcpy
CC -xhip -DHIP -DOPEN_MPI -DPINNED -O3 src-intra-node/pp_hipmemcpy.cpp -o bin/pp_hipmemcpy