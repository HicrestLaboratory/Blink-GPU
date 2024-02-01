# Original script content
stencil_script=$(cat << 'EOF'
#!/bin/bash

#SBATCH --job-name=<exp-name>_<exp-type>_<exp-topo>
#SBATCH --output=sout/marzola_<exp-name>_<exp-type>_<exp-topo>_%j.out
#SBATCH --error=sout/marzola_<exp-name>_<exp-type>_<exp-topo>_%j.err

#SBATCH --partition=short
#SBATCH --time=00:05:00

#SBATCH --nodes=1
#SBATCH --gres=gpu:<num-proc>
#SBATCH --ntasks-per-node=<num-proc>
#SBATCH --cpus-per-task=1

MODULE_PATH="moduleload/load_<exp-type>_modules.sh"
EXPORT_PATH="exportload/load_<exp-type>_<exp-topo>_exports.sh"

mkdir -p sout
source ${MODULE_PATH} && source ${EXPORT_PATH} && srun bin/<exp-name>_<exp-type>
EOF
)

names=("pp" "a2a")
types=("Baseline" "CudaAware" "Nccl" "Nvlink")
topos=("singlenode")

for name in "${names[@]}"
do
    echo "#!/bin/bash" > "sbatch/marzola/run-marzola-$name-all.sh"
    for type in "${types[@]}"
    do
        for topo in "${topos[@]}"
        do
#             echo "$name $type $topo"
            out_script_contenent=$(echo "$stencil_script" | sed "s/<exp-name>/$name/g" | sed "s/<exp-type>/$type/g" | sed "s/<exp-topo>/$topo/g")
            tmp_script_contenent=$(echo "$out_script_contenent")

            if [[ "$topo" == "multinode" ]]
            then
                out_script_contenent=$(echo "$tmp_script_contenent" | sed "s/nodes=1/nodes=2/g")
            fi
            tmp_script_contenent=$(echo "$out_script_contenent")

            if [[ "$name" == "pp" ]]
            then
                out_script_contenent=$(echo "$tmp_script_contenent" | sed "s/<num-proc>/2/g")
            fi

            if [[ "$name" == "a2a" ]]
            then
                out_script_contenent=$(echo "$tmp_script_contenent" | sed "s/<num-proc>/4/g")
            fi
            tmp_script_contenent=$(echo "$out_script_contenent")

            # Write the new script to a file
            out_script_file="sbatch/marzola/run-marzola-$name-$type-$topo.sh"
            echo "$out_script_contenent" > "$out_script_file"
            chmod +x "$out_script_file"

            echo "Generated $out_script_file"

            echo "sbatch $out_script_file" >> "sbatch/marzola/run-marzola-$name-all.sh"
        done
    done
    chmod +x "sbatch/marzola/run-marzola-$name-all.sh"
done