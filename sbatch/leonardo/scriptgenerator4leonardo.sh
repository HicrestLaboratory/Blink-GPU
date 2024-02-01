# Original script content
stencil_script=$(cat << 'EOF'
#!/bin/bash

#SBATCH --job-name=<exp-name>_<exp-type>_<exp-topo>
#SBATCH --output=sout/leonardo_<exp-name>_<exp-type>_<exp-topo>_%j.out
#SBATCH --error=sout/leonardo_<exp-name>_<exp-type>_<exp-topo>_%j.err

#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_SHARP_0
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:05:00

#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=494000MB

MODULE_PATH="moduleload/load_<exp-type>_modules.sh"
EXPORT_PATH="exportload/load_<exp-type>_<exp-topo>_exports.sh"

mkdir -p sout
source ${MODULE_PATH} && source ${EXPORT_PATH} && srun bin/<exp-name>_<exp-type>
EOF
)

names=("pp" "a2a" "hlo")
types=("Baseline" "CudaAware" "Nccl" "Nvlink")
topos=("singlenode" "multinode")

for name in "${names[@]}"
do
    echo "#!/bin/bash" > "sbatch/leonardo/run-leonardo-$name-all.sh"
    for type in "${types[@]}"
    do
        for topo in "${topos[@]}"
        do
            if [[ "$name" != "hlo" || "$type" == "Baseline" ]] # BUG TMP since halo now implemented only in Baseline
            then

                out_script_contenent=$(echo "$stencil_script" | sed "s/<exp-name>/$name/g" | sed "s/<exp-type>/$type/g" | sed "s/<exp-topo>/$topo/g")
                tmp_script_contenent=$(echo "$out_script_contenent")

                if [[ "$topo" == "multinode" ]]
                then
                    out_script_contenent=$(echo "$tmp_script_contenent" | sed "s/nodes=1/nodes=2/g")
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