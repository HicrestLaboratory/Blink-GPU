#!/bin/bash

working_path="utils"
results_path="results"

tmp_file="${working_path}/tmp.txt"
sum_file="${results_path}/summary.txt"
tmp_file2="${working_path}/tmp2.txt"

mkdir -p "${results_path}"

glob_flag=0

# List of fixed reference folders
excluded_folders=("include" "utils" "topology-finder-test" "nccl_test" "results")


for exp_dir in *
do
    if [ -d "${exp_dir}" ]
    then
        # Check if the folder is different from the fixed reference folders
        bool_cond=true
        for ref_folder in "${excluded_folders[@]}"
        do
            if [ "${exp_dir}" == "${ref_folder}" ]
            then
                bool_cond=false
                break
            fi
        done

#         echo "exp_dir:   ${exp_dir}"
#         echo "bool_cond: ${bool_cond}"

        if ${bool_cond}
        then
            exp_name=$(echo "${exp_dir}" | sed 's|.*/||')
            out_file="${results_path}/${exp_name}.out"

            flag=0
            for file in "${exp_dir}"/out/*.out
            do
                if [ ${flag} -eq 0 ]
                then
                    cat "${file}" | tail -n 10 | head -2 > "${tmp_file}"
                    flag=1

                    if [ ${glob_flag} -eq 0 ]
                    then
                        cat "${file}" | tail -n 10 | head -2 > "${tmp_file2}"
                        glob_flag=1
                    fi
                fi

                cat "${file}" | tail -n 8 >> "${tmp_file}"
                cat "${file}" | tail -n 8 >> "${tmp_file2}"
            done

            cat "${tmp_file}" | awk '!/^     -----/' > "${out_file}"
        fi
    fi
done

cat "${tmp_file2}" | awk '!/^     -----/' > "${sum_file}"
rm "${tmp_file}" "${tmp_file2}"
