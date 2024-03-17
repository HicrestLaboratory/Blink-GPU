# Original script content
stencil_script=$(cat << 'EOF'
#!/bin/bash

#SBATCH --job-name=<exp-name>_<exp-type>_<exp-topo>
#SBATCH --output=sout/leonardo_<exp-name>_<exp-type>_<exp-topo>_%j.out
#SBATCH --error=sout/leonardo_<exp-name>_<exp-type>_<exp-topo>_%j.err

#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_SHARP_0
#SBATCH --time=00:05:00
#SBATCH --qos=normal

#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=494000MB

#SBATCH --requeue

echo "-------- Topology Info --------"
echo "Nnodes = $SLURM_NNODES"
srun -l bash -c 'if [[ "$SLURM_LOCALID" == "0" ]] ; then t="$SLURM_TOPOLOGY_ADDR" ; echo "Node: $SLURM_NODEID ---> $t" ; echo "$t" > tmp_${SLURM_NODEID}_${SLURM_JOB_ID}.txt ; fi'
echo "-------------------------------"
switchesPaths=()
for i in $(seq 0 $SLURM_NNODES)
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



MODULE_PATH="moduleload/load_<exp-type>_modules.sh"
EXPORT_PATH="exportload/load_<exp-type>_<exp-topo>_exports.sh"

mkdir -p sout
cat "${EXPORT_PATH}"
source ${MODULE_PATH} && source ${EXPORT_PATH} && <sl-export> <sl-exp-and> srun bin/<exp-name>_<exp-type> <exp_args>
EOF
)

my_sl="1"
my_min_sw_distance="3"

names=("pp" "a2a" "ar" "hlo" "mpp")
types=("Baseline" "CudaAware" "Nccl" "Nvlink" "Aggregated")
topos=("singlenode" "multinode")

for name in "${names[@]}"
do
    echo "#!/bin/bash" > "sbatch/leonardo/run-leonardo-$name-all.sh"
    for type in "${types[@]}"
    do
        for topo in "${topos[@]}"
        do
            if [[
                ("$topo" != "multinode" || "$type" != "Nvlink") &&
                ("$type" != "Aggregated" || "$name" == "mpp") &&
                ("$name" != "mpp" || "$topo" != "singlenode") &&
                ("$name" != "hlo" || "$type" != "Nvlink") &&
                ("$name" != "ar" || "$type" != "Nvlink")
            ]] # BUG TMP since halo and ar now implemented only in Baseline
            then

                out_script_contenent=$(echo "$stencil_script" | sed "s/<exp-name>/$name/g" | sed "s/<exp-type>/$type/g" | sed "s/<exp-topo>/$topo/g")
                tmp_script_contenent=$(echo "$out_script_contenent")

                if [[ "$topo" == "multinode" ]]
                then
                    out_script_contenent=$(echo "$tmp_script_contenent" | sed "s/nodes=1/nodes=2/g")
                fi

                tmp_script_contenent=$(echo "$out_script_contenent")
                if [[ "$name" == "hlo" ]]
                then
                    if [[ "$topo" == "singlenode" ]]
                    then
                        out_script_contenent=$(echo "$tmp_script_contenent" | sed "s/<exp_args>/-pex 2 -pey 2 -pez 1/g")
                    else
                        out_script_contenent=$(echo "$tmp_script_contenent" | sed "s/<exp_args>/-pex 2 -pey 2 -pez 2/g")
                    fi
                else
                    out_script_contenent=$(echo "$tmp_script_contenent" | sed "s/<exp_args>//g")
                fi

                tmp_script_contenent=$(echo "$out_script_contenent")
                if [[ "$topo" == "multinode" && "${my_sl}" != "0" ]]
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
                if [[ "$topo" == "multinode" ]]
                then
                    out_script_contenent=$(echo "$tmp_script_contenent" | sed "s/<my_min_sw_distance>/${my_min_sw_distance}/g")
                else
                    out_script_contenent=$(echo "$tmp_script_contenent" | sed "s/<my_min_sw_distance>/0/g")
                fi

                # Write the new script to a file
                out_script_file="sbatch/leonardo/run-leonardo-$name-$type-$topo.sh"
                echo "$out_script_contenent" > "$out_script_file"
                chmod +x "$out_script_file"

                echo "Generated $out_script_file"

                echo "sbatch $out_script_file" >> "sbatch/leonardo/run-leonardo-$name-all.sh"
            fi
        done
    done
    chmod +x "sbatch/leonardo/run-leonardo-$name-all.sh"
done
