#!/bin/bash

expand_HR_strings () {

	utils_lib_systems="${1}"
	utils_lib_benchmarks="${2}"
	utils_lib_implementations="${3}"
	utils_lib_topologies="${4}"
	utils_lib_partitions="${5}"

	echo "systems: ${utils_lib_systems}"
	echo "benchmarks: ${utils_lib_benchmarks}"
	echo "implementations: ${utils_lib_implementations}"
	echo "topologies: ${utils_lib_topologies}"
	echo "partitions: ${utils_lib_partitions}"

	utils_lib_system_vec=()
	IFS=':' read -r -a utils_lib_system_vec <<< "${utils_lib_systems}"
	echo "system_vec: ${utils_lib_system_vec[*]}"

	utils_lib_benchmark_vec=()
	IFS=':' read -r -a utils_lib_benchmark_vec <<< "${utils_lib_benchmarks}"
	echo "benchmark_vec: ${benchmark_vec[*]}"

	utils_lib_implementation_vec=()
	IFS=':' read -r -a utils_lib_implementation_vec <<< "${utils_lib_implementations}"
	echo "implementation_vec: ${utils_lib_implementation_vec[*]}"

	utils_lib_topology_vec=()
	IFS=':' read -r -a utils_lib_topology_vec <<< "${utils_lib_topologies}"
	echo "topology_vec: ${utils_lib_topology_vec[*]}"

	utils_lib_partition_vec=()
	IFS=':' read -r -a utils_lib_partition_vec <<< "${utils_lib_partitions}"
	echo "partition_vec: ${utils_lib_partition_vec[*]}"

	utils_lib_result_sys=()
	utils_lib_result_ben=()
	utils_lib_result_imp=()
	utils_lib_result_top=()
	utils_lib_result_par=()
	for utils_lib_s in "${utils_lib_system_vec[@]}"
	do
        	for utils_lib_b in "${utils_lib_benchmark_vec[@]}"
	        do
                	for utils_lib_i in "${utils_lib_implementation_vec[@]}"
        	        do
	                        for utils_lib_t in "${utils_lib_topology_vec[@]}"
                        	do
                	                for utils_lib_p in "${utils_lib_partition_vec[@]}"
        	                        do	
						utils_lib_result_sys+=( "${utils_lib_s}" )
					        utils_lib_result_ben+=( "${utils_lib_b}" )
					        utils_lib_result_imp+=( "${utils_lib_i}" )
					        utils_lib_result_top+=( "${utils_lib_t}" )
					        utils_lib_result_par+=( "${utils_lib_p}" )
                                	done
                        	done
                	done
        	done
	done
}
