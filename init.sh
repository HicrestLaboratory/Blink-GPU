#!/bin/bash

# --- Read inputr params ---
#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 --system <name> [--configuration <conf>]

Options:
  --system         (required) System name
  --configuration  (optional) Configuration name
  -h, --help       Show this help message
EOF
  exit 1
}

# Parse arguments
PARSED_ARGS=$(getopt -o h --long help,system:,configuration: -- "$@") || usage
eval set -- "$PARSED_ARGS"

SYSTEM=""
CONFIGURATION=""

while true; do
  case "$1" in
    --system)
      SYSTEM="${2^^}"
      shift 2
      ;;
    --configuration)
      CONFIGURATION="${2^^}"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unexpected option: $1"
      usage
      ;;
  esac
done

# Validate mandatory parameter
if [[ -z "$SYSTEM" ]]; then
  echo "Error: --system is required."
  usage
fi

# ---- Script logic below ----
echo "System: $SYSTEM"

if [[ -n "$CONFIGURATION" ]]; then
  echo "Configuration: $CONFIGURATION"
else
  CONFIGURATION="DEFAULT"
  echo "Configuration: $CONFIGURATION"
fi

#export BLINKGPU_SYSTEM=${SYSTEM}
#export BLINKGPU_CONFIGURATION=${CONFIGURATION}
#export BLINKGPU_SETUPDIR="setup_dir/"
# ---------------------------------------

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

write_setup_file() {
  local setupfile="$1"
  local cuda_home="$2"
  local nccl_home="$3"
  local mpi_home="$4"
  local mpicuda_home="$5"
  local cuda_module="$6"
  local nccl_module="$7"
  local mpi_module="$8"
  local mpicuda_module="$9"

  {
    echo "#!/bin/bash"
    echo "# Generated on $dt"
    echo
    echo "export BLINKGPU_SYSTEM=${SYSTEM}"
    echo "export BLINKGPU_CONFIGURATION=${CONFIGURATION}"
    echo "export BLINKGPU_SETUPDIR=${SETUP_DIR}"
    echo
    echo "export BLINKGPU_CUDA_HOME=\"${cuda_home}\""
    echo "export BLINKGPU_NCCL_HOME=\"${nccl_home}\""
    echo "export BLINKGPU_MPI_HOME=\"${mpi_home}\""
    echo "export BLINKGPU_MPICUDA_HOME=\"${mpicuda_home}\""
    echo
    echo "export BLINKGPU_CUDA_MODULE=\"${cuda_module}\""
    echo "export BLINKGPU_NCCL_MODULE=\"${nccl_module}\""
    echo "export BLINKGPU_MPI_MODULE=\"${mpi_module}\""
    echo "export BLINKGPU_MPICUDA_MODULE=\"${mpicuda_module}\""
  } > "$setupfile"
}


# =========== Variables to export ===========
# Home variables:   BLINKGPU_CUDA_HOME,   BLINKGPU_NCCL_HOME,   BLINKGPU_MPI_HOME
# Module variables: BLINKGPU_CUDA_MODULE, BLINKGPU_NCCL_MODULE, BLINKGPU_MPI_MODULE
# ===========================================

SETUP_DIR="configure/"
setupfile="${SETUP_DIR}${SYSTEM}_${CONFIGURATION}.conf"
mkdir -p ${SETUP_DIR}
case "${SYSTEM}:${CONFIGURATION}" in
  BALDO:DEFAULT)
    echo "Generating exports for ${SYSTEM} system and ${CONFIGURATION} configuration"
    # Home directories
    cuda_home="/opt/shares/cuda/software/CUDA/12.3.2"
    nccl_home="/opt/shares/NVHPC/nvhpc_24.7_cuda_12.5/Linux_x86_64/24.7/comm_libs/12.5/nccl"
    mpi_home="/opt/shares/openmpi-4.1.5-cuda-12.3.2"
    mpicuda_home="/opt/shares/openmpi-4.1.5-cuda-12.3.2"

    # Module name
    cuda_module="CUDA/12.3.2"
    nccl_module="NVHPC/nvhpc/24.7"
    mpi_module="OpenMpi/4.1.5-CUDA-12.3.2"
    mpicuda_module="OpenMpi/4.1.5-CUDA-12.3.2"
    ;;

  BALDO:NVHPC)
    echo "Generating exports for ${SYSTEM} system and ${CONFIGURATION} configuration"
    # Home directories
    cuda_home="/opt/shares/NVHPC/nvhpc_24.7_cuda_12.5/Linux_x86_64/24.7/cuda"
    nccl_home="/opt/shares/NVHPC/nvhpc_24.7_cuda_12.5/Linux_x86_64/24.7/comm_libs/12.5/nccl"
    mpi_home="/opt/shares/NVHPC/nvhpc_24.7_cuda_12.5/Linux_x86_64/24.7/comm_libs/12.5/openmpi4/openmpi-4.1.5"
    mpicuda_home="/opt/shares/NVHPC/nvhpc_24.7_cuda_12.5/Linux_x86_64/24.7/comm_libs/12.5/openmpi4/openmpi-4.1.5"

    # Module name
    cuda_module="NVHPC/nvhpc/24.7"
    nccl_module="NVHPC/nvhpc/24.7"
    mpi_module="NVHPC/nvhpc/24.7"
    mpicuda_module="NVHPC/nvhpc/24.7"
    ;;

  LEONARDO:DEFAULT)
    echo "Generating exports for ${SYSTEM} system and ${CONFIGURATION} configuration"
    # Home directories
    cuda_home="/leonardo/prod/spack/06/install/0.22/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.5-torlmnyzcexnrs6pq4cccabv7ehkv3xy/Linux_x86_64/24.5/cuda"
    nccl_home="/leonardo/prod/spack/06/install/0.22/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.5-torlmnyzcexnrs6pq4cccabv7ehkv3xy/Linux_x86_64/24.5/comm_libs/nccl"
    mpi_home="/leonardo/prod/spack/06/install/0.22/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.5-torlmnyzcexnrs6pq4cccabv7ehkv3xy/Linux_x86_64/24.5/comm_libs/openmpi/openmpi-3.1.5"
    mpicuda_home="/leonardo/prod/spack/06/install/0.22/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.5-torlmnyzcexnrs6pq4cccabv7ehkv3xy/Linux_x86_64/24.5/comm_libs/openmpi/openmpi-3.1.5"

    # Module name
    cuda_module="nvhpc/24.5"
    nccl_module="nvhpc/24.5"
    mpi_module="nvhpc/24.5"
    mpicuda_module="nvhpc/24.5"
    ;;

  PICOSAMSUNGPC:DEFAULT)
    # Home directories
    cuda_home="/picomodules/nvhpc/Linux_x86_64/25.11/cuda"
    nccl_home="/picomodules/nvhpc/Linux_x86_64/25.11/comm_libs/nccl"
    mpi_home="/usr/local/apps/openmpi/4.1.5"
    mpicuda_home="/usr/local/apps/openmpi/4.1.5"

    # Module name
    cuda_module="nvhpc/25.11"
    nccl_module="nvhpc/25.11"
    mpi_module="openmpi/4.1.5"
    mpicuda_module="openmpi/4.1.5"
  ;;

  *)
    echo "Error: Unknown system '$SYSTEM'"
    exit 1
    ;;
esac

write_setup_file \
      "$setupfile" \
      "$cuda_home" \
      "$nccl_home" \
      "$mpi_home" \
      "$mpicuda_home"\
      "$cuda_module" \
      "$nccl_module" \
      "$mpi_module" \
      "$mpicuda_module"

#source ${BLINKGPU_SETUPDIR}${BLINKGPU_SYSTEM}_${BLINKGPU_CONFIGURATION}.conf
