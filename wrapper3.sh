#!/bin/bash
ID=`date +%s`
#if [[ $OMPI_COMM_WORLD_RANK == 0 ]]; then
	nsys profile --trace=cuda,mpi --mpi-impl=openmpi --force-overwrite true -o nsys_sctNvlink-${ID} bin/sct_Nvlink -b 28
#else
#	bin/sct_Nvlink -b 28
#fi
