#!/bin/bash

ID=`date +%s`
if [[ $OMPI_COMM_WORLD_RANK == 0 ]]
then
	nsys profile --trace=cuda,mpi,nvtx -o async-1-${OMPI_COMM_WORLD_RANK}-${ID} bin/mpp_Aggregated -b 28 -l 5
else
	bin/mpp_Aggregated -b 28 -l 5
fi
