# Original script content
stencil_script=$(cat << 'EOF'
#!/bin/bash

#SBATCH --job-name=blinkGPU_%j
#SBATCH --output=sout/slurmInfo/slurmInfo_snellius_%j.out
#SBATCH --error=sout/slurmInfo/slurmInfo_snellius_%j.err

#SBATCH --partition=gpu_a100
#SBATCH --account=vusei7310
#SBATCH --time=01:00:00
#SBATCH --qos=normal

#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8

#SBATCH --exclusive
#SBATCH --requeue

# -------------------------------------------------------------------------------------------------------------------------
#                                                   SLURM dependent part
# -------------------------------------------------------------------------------------------------------------------------

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
# if [[ "$(($maxDist - $y))" != "<my_min_sw_distance>" ]]
# then
#     echo "nodes are at the wrong distance ($(($maxDist - $y)) instead of <my_min_sw_distance>); job requeued"
#     scontrol requeue ${SLURM_JOB_ID}
# fi

echo "-------------------------------"
echo "<sl-export>"
echo "-------------------------------"
srun -l bash -c 'export SLURM_LOCALID'
srun -l bash -c 'echo "SLURM_LOCALID = ${SLURM_LOCALID}"'
srun -l bash -c 'export UCX_NET_DEVICES=mlx5_${SLURM_LOCALID}:1 ; echo "UCX_NET_DEVICES: ${UCX_NET_DEVICES}"'
srun -l bash -c 'echo "UCX_NET_DEVICES: ${UCX_NET_DEVICES}"'

mkdir -p sout
mkdir -p sout/slurmInfo

# -------------------------------------------------------------------------------------------------------------------------
#                                                 Parameter dependent part
# -------------------------------------------------------------------------------------------------------------------------


EOF
)

stencil_script_binpart=$(cat << 'EOF'

MODULE_PATH="moduleload/load_<exp-type>_modules.sh"
EXPORT_PATH="exportload/load_<exp-type>_<exp-topo>_exports.sh"
outfile=sout/snellius_<exp-name>_<exp-type>_<exp-topo>_${SLURM_JOB_ID}.out
errfile=sout/snellius_<exp-name>_<exp-type>_<exp-topo>_${SLURM_JOB_ID}.err

cat "${EXPORT_PATH}" >${outfile} 2>${errfile}
source ${MODULE_PATH} && source ${EXPORT_PATH} && <sl-export> <sl-exp-and> mpirun -np ${SLURM_NTASKS} bin/<exp-name>_<exp-type> <exp_args> >${outfile} 2>${errfile}

EOF
)

rm sbatch/snellius/run-snellius-*

# Change it from 0 to 1 if required
fixednodeflag="1"

my_sl="1"
my_min_sw_distance="3"

names=("pp" "a2a" "ar" "hlo" "mpp")
types=("Baseline" "CudaAware" "Nccl" "Nvlink" "Aggregated")
topos=("1" "2" "4" "8")

firstiterationflag="1"
for topo in "${topos[@]}"
do
    firstiterationflag="1"
    if [[ "${topo}" == "1" ]]
    then
        topolable="singlenode"
    else
        topolable="multinode"
    fi

    for type in "${types[@]}"
    do
        for name in "${names[@]}"
        do
            if [[ "${fixednodeflag}" == "0" ]] && [[ ! -f "sbatch/snellius/run-snellius-$name-all.sh" ]]
            then
                echo "#!/bin/bash" > "sbatch/snellius/run-snellius-$name-all.sh"
            fi

            if [[
                ("$topo" == "1" || "$type" != "Nvlink") &&
                ("$type" != "Aggregated" || "$name" == "mpp") &&
                ("$name" != "mpp" || "$topo" != "1") &&
                ("$name" != "hlo" || "$type" != "Nvlink") &&
                ("$name" != "ar" || "$type" != "Nvlink")
            ]] # BUG TMP since halo and ar now implemented only in Baseline
            then

#                if [[ "${fixednodeflag}" == "0" ]] || [[ "${firstiterationflag}" == "1" ]]
#                then

                    if [[ "${fixednodeflag}" == "0" ]] || [[ "${firstiterationflag}" == "1" ]]
                    then
                        out_script_contenent=$( echo "$stencil_script $stencil_script_binpart" )
                    else
                        out_script_contenent=$( echo "$stencil_script_binpart" )
                    fi
                    tmp_script_contenent=$(echo "$out_script_contenent")

                    out_script_contenent=$(echo "$tmp_script_contenent" | sed "s/<exp-name>/$name/g" | sed "s/<exp-type>/$type/g" | sed "s/<exp-topo>/$topolable/g")
                    tmp_script_contenent=$(echo "$out_script_contenent")

                    if [[ "$topo" != "1" ]]
                    then
                        out_script_contenent=$(echo "$tmp_script_contenent" | sed "s/nodes=1/nodes=2/g")
                    fi
                    tmp_script_contenent=$(echo "$out_script_contenent")

                    if [[ "$name" == "hlo" ]]
                    then
                        if [[ "$topo" == "1" ]]
                        then
                            out_script_contenent=$(echo "$tmp_script_contenent" | sed "s/<exp_args>/-pex 2 -pey 2 -pez 1/g")
                        else
                            out_script_contenent=$(echo "$tmp_script_contenent" | sed "s/<exp_args>/-pex 2 -pey 2 -pez 2/g")
                        fi
                    else
                        out_script_contenent=$(echo "$tmp_script_contenent" | sed "s/<exp_args>//g")
                    fi

                    tmp_script_contenent=$(echo "$out_script_contenent")
                    if [[ "$topo" != "1" && "${my_sl}" != "0" ]]
                    then
                        if [[ "$type" == "Nccl" ]]
                        then
                            out_script_contenent=$(echo "$tmp_script_contenent" | sed "s/<sl-export>/export NCCL_IB_SL=${my_sl}/g" | sed "s/<sl-exp-and>/\&\&/g")
                        else
                            out_script_contenent=$(echo "$tmp_script_contenent" | sed "s/<sl-export>/export UCX_IB_SL=${my_sl}/g" | sed "s/<sl-exp-and>/\&\&/g")
                        fi
                    else
                        out_script_contenent=$(echo "$tmp_script_contenent" | sed "s/<sl-export>//g" | sed "s/<sl-exp-and>//g")
                    fi

                    tmp_script_contenent=$(echo "$out_script_contenent")
                    if [[ "$topo" != "1" ]]
                    then
                        out_script_contenent=$(echo "$tmp_script_contenent" | sed "s/<my_min_sw_distance>/${my_min_sw_distance}/g")
                    else
                        out_script_contenent=$(echo "$tmp_script_contenent" | sed "s/<my_min_sw_distance>/0/g")
                    fi

                    if [[ "${fixednodeflag}" == "0" ]]
                    then
                        # Write the new script to a file
                        out_script_file="sbatch/snellius/run-snellius-$name-$type-${topo}node.sh"
                        echo "$out_script_contenent" > "$out_script_file"
                        chmod +x "$out_script_file"

                        echo "Generated $out_script_file"

                        echo "sbatch $out_script_file" >> "sbatch/snellius/run-snellius-$name-all.sh"
                    else
                        # Write the new script to a file
                        if [[ "${firstiterationflag}" == "1" ]]
                        then
                            out_script_file="sbatch/snellius/run-snellius-${topo}node.sh"
                            echo "$out_script_contenent" > "$out_script_file"
                            chmod +x "$out_script_file"

                            echo "Generated $out_script_file"
                        else
                            echo "$out_script_contenent" >> "$out_script_file"
                            echo "$name-$type-${topo}node appended to $out_script_file"
                        fi
                    fi
                    firstiterationflag="0"
                #fi
            fi
        done
    done
    chmod +x "sbatch/snellius/run-snellius-$name-all.sh"
done
