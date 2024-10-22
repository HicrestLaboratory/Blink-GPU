#!/bin/bash

#SBATCH --job-name=pp_Baseline_singlenode
#SBATCH --output=test_snellius_pp_Baseline_singlenode_%j.out
#SBATCH --error=test_snellius_pp_Baseline_singlenode_%j.err

#SBATCH --partition=gpu_a100
#SBATCH --account=vusei7310
#SBATCH --time=00:05:00
#SBATCH --qos=normal

#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8

#SBATCH --exclusive
#SBATCH --requeue

echo "-------- Topology Info --------"
echo "Nnodes = $SLURM_NNODES"
srun -l bash -c 'if [[ "$SLURM_LOCALID" == "0" ]] ; then t="$SLURM_TOPOLOGY_ADDR" ; echo "Node: $SLURM_NODEID ---> $t" ; echo "$t" > tmp_${SLURM_NODEID}_${SLURM_JOB_ID}.txt ; fi'
echo "-------------------------------"
echo "Partition = ${SLURM_JOB_PARTITION}"
echo "-------------------------------"
switchesPaths=()
for i in $( seq 0 $((SLURM_NNODES - 1)) )
do
        text=$(cat "tmp_${i}_${SLURM_JOB_ID}.txt")
        switchesPaths+=( "$text" )
        rm "tmp_${i}_${SLURM_JOB_ID}.txt"
done

echo "switchesPaths:"
for e in ${switchesPaths[@]}
do
        echo $e
done

echo "-------------------------------"
IFS='.' read -a zeroPath <<< "${switchesPaths[0]}"
# echo "zeroPath:"
# for e in ${zeroPath[@]}
# do
#         echo $e
# done

y="${#zeroPath[@]}"
zeroNode=${zeroPath[-1]}
maxDist="${#zeroPath[@]}"
for e in ${switchesPaths[@]}
do
        IFS='.' read -a tmpPath <<< "$e"
        tmpNode=${tmpPath[-1]}
        x="${#zeroPath[@]}"
        for j in ${!zeroPath[@]}
        do
                if [[ "${zeroPath[$j]}" != "${tmpPath[$j]}" && "$j" < "$x" ]]
                then
                        x="$j"
                        if [[ "$x" < "$y" ]]
                        then
                                y="$x"
                        fi
                fi
        done
        echo "$tmpNode ---> distance with node 0 ($zeroNode) = $(($maxDist - $x))"
done

echo "Max distance: $(($maxDist - $y))"
# if [[ "$(($maxDist - $y))" != "0" ]]
# then
#     echo "nodes are at the wrong distance ($(($maxDist - $y)) instead of 0); job requeued"
#     scontrol requeue ${SLURM_JOB_ID}
# fi

echo "-------------------------------"
echo ""
echo "-------------------------------"
srun -l bash -c 'export SLURM_LOCALID'
srun -l bash -c 'echo "SLURM_LOCALID = ${SLURM_LOCALID}"'
srun -l bash -c 'export UCX_NET_DEVICES=mlx5_${SLURM_LOCALID}:1 ; echo "UCX_NET_DEVICES: ${UCX_NET_DEVICES}"'
srun -l bash -c 'echo "UCX_NET_DEVICES: ${UCX_NET_DEVICES}"'

MODULE_PATH="moduleload/load_Baseline_modules.sh"
EXPORT_PATH="exportload/load_Baseline_singlenode_exports.sh"

mkdir -p sout
cat "${EXPORT_PATH}"
srun -l bash -c 'source moduleload/load_Baseline_modules.sh ; bin/pp_Baseline -l 5 -b 20'
