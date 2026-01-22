#!/bin/bash
#PBS -N a2a_occl_multinode
#PBS -A  EnergyOpt_PhaseFreq
#PBS -l select=2
#PBS -l walltime=00:10:00
#PBS -l filesystems=home
#PBS -o /home/lcarpent/energy-workspace/Blink-GPU/energy/logs/pbs-out/output_a2a_occl.txt
#PBS -e /home/lcarpent/energy-workspace/Blink-GPU/energy/logs/pbs-out/error_a2a_occl.txt
#PBS -q debug-scaling

echo "Executing a2a ..."  # Output: app_name
echo "Unique nodes allocated:"
sort -u $PBS_NODEFILE
source /home/lcarpent/energy-workspace/Blink-GPU/energy/scripts/aurora/env/set_env.sh

export  ZES_ENABLE_SYSMAN=1
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_LOG_LEVEL=error

# Set scaleup and scaleout mode for multi node.
export CCL_ALLTOALLV_MONOLITHIC_KERNEL=${SCALEUP_MODE} # SCALE_UP mode can be 0 (use copy engines) or 1 (use GPU compute kernel to transfer data across GPUs). 
export CCL_ALLTOALL_SCALEOUT=${ALG_NAME} # Not used in single node but set for consistency.

# MPI example w/ 12 MPI ranks per node each with access to single GPU tile
NNODES=`wc -l < $PBS_NODEFILE`

NRANKS_PER_NODE=12 # use all gpus onthe node
NDEPTH=1
NTHREADS=1

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"

MPI_ARG="-n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth "

AFFINITY=""
AFFINITY="/home/lcarpent/energy-workspace/Blink-GPU/energy/scripts/aurora/env/gpu_affinity.sh"


# mpiexec -n 12 -ppn 12  ${EXE_PATH} ${power_file} ${csv_file}



power_file="${POWER_LOG_DIR}/multi-node/occl/${COLLECTIVE}_${SCALEUP_MODE}_${ALG_NAME}_pow.csv"
csv_file="${CSV_LOG_DIR}/multi-node/occl/${COLLECTIVE}_${SCALEUP_MODE}_${ALG_NAME}.csv"

COMMAND="mpiexec ${MPI_ARG} ${EXE_PATH} ${power_file} ${csv_file}"

$COMMAND