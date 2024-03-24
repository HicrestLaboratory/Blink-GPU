'''
#!/bin/bash

#SBATCH --job-name=<exp-name>_<exp-type>_<exp-topo>
#SBATCH --output=sout/lumi_<exp-name>_<exp-type>_<exp-topo>_%j.out
#SBATCH --error=sout/lumi_<exp-name>_<exp-type>_<exp-topo>_%j.err

#SBATCH --partition=ju-standard-g
#SBATCH --time=00:10:00

#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=<num-proc>
#SBATCH --cpus-per-task=1

#SBATCH --account=project_465000997

mkdir -p sout
srun bin/<exp-name>_<exp-type> <exp_args>
EOF
)
'''

import os
import stat
import sys
import os.path


names=["a2a", "ar", "mpp"]
types=["Baseline", "CudaAware", "Nccl"]

# nodes_tasks = [(1, 4), (1,8), (2,8)]
nodes_tasks = [(2, 8)]
cpu_list = [9, 25, 41, 57, 1, 17, 33, 49]
gpu_list = [5, 3, 7, 1, 4, 2, 6, 0]

rccl_env = """
module purge
export EBU_USER_PREFIX=/project/project_465000997/EasyBuild
module load LUMI/22.08
module load aws-ofi-rccl
module load craype-accel-amd-gfx90a
module load rocm

export NCCL_NET_GDR_LEVEL=3
export FI_CXI_ATS=0
export NCCL_BUFFSIZE=33554432
"""

standard_env = """
module purge
module load LUMI/23.09
module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm
"""

rccl_module_load = "source moduleload/load_Nccl_modules.sh\n"
baseline_module_load = "source moduleload/load_Baseline_modules.sh\n"
cudaaware_module_load = "source moduleload/load_CudaAware_modules.sh\n"
nvlink_module_load = "source moduleload/load_Nvlink_modules.sh\n"

# Update the number of nodes, and number tasks per-node from command line arguments
if len(sys.argv) == 1:
    print("#nodes and #tasks-per-node not provided.")
    print("Use default setting with {} node(s) and {} task(s) per node.".format(nodes_tasks[0][0], nodes_tasks[0][1]))
    print("")
elif len(sys.argv) == 3:
    num_nodes = int(sys.argv[1])
    num_tasks_per_node = int(sys.argv[2])
    if num_nodes < 1:
        print("Invalid number-of-nodes, should > 1, get {}".format(num_nodes))
        exit(1)
    if num_tasks_per_node < 1 or num_tasks_per_node > 8:
        print("Invalid number-of-tasks-per-node, shoud be in [1, 8], get {}.".format(num_tasks_per_node))
        exit(1)
    nodes_tasks = [(num_nodes, num_tasks_per_node)]
else:
    print("Invalid argument.")
    print("Usage: python3 script-name.py number-of-nodes number-of-tasks-per-node.")

args = ""
f_run_all_name = 'sbatch/lumi/run-lumi-all.sh'
f_run_all = open(f_run_all_name, 'w')

for cur_name in names:
    for cur_type in types:
        for cur_nodes, cur_ntasks_per_node in nodes_tasks:
            topo_name = "{}-{}".format(str(cur_nodes), str(cur_ntasks_per_node))
            sbatch_fname = "run-lumi-{}-{}-{}.sh".format(cur_name, cur_type, topo_name)
            sbatch_fname = os.path.join('sbatch/lumi', sbatch_fname)
            f = open(sbatch_fname, 'w')
            f.write("#!/bin/bash\n\n")
            f.write("#SBATCH --job-name={}_{}_{}\n".format(cur_name, cur_type, topo_name))
            f.write("#SBATCH --output=sout/lumi_{}_{}_{}_%j.out\n".format(cur_name, cur_type, topo_name))
            f.write("#SBATCH --error=sout/lumi_{}_{}_{}_%j.err\n\n".format(cur_name, cur_type, topo_name))
            f.write("#SBATCH --partition=ju-standard-g\n")
            f.write("#SBATCH --time=00:10:00\n\n")
            f.write("#SBATCH --nodes={}\n".format(str(cur_nodes)))
            f.write("#SBATCH --gpus-per-node=8\n")
            f.write("#SBATCH --ntasks-per-node={}\n".format(str(cur_ntasks_per_node)))
            f.write("#SBATCH --account=project_465000997\n\n")
            cur_cpu_list = cpu_list[0:cur_ntasks_per_node]
            cur_cpu_list = [str(x) for x in cur_cpu_list]
            cur_cpu_list = ','.join(cur_cpu_list)
            f.write('CPU_BIND="map_cpu:{}"\n\n'.format(cur_cpu_list))
            f.write("mkdir -p sout\n\n")

            # RCCL specific options
            if cur_type == "Nccl":
                f.write(rccl_module_load)
            elif cur_type == "Baseline":
                f.write(baseline_module_load)
            elif cur_type =="CudaAware":
                f.write(cudaaware_module_load)
            elif cur_type =="Nvlink":
                f.write(nvlink_module_load)
            else:
                raise RuntimeError(f"Unknown benchmark type: {cur_type}")


            # define GPU env
            cur_gpu_list = gpu_list[0:cur_ntasks_per_node]
            cur_gpu_list = [str(x) for x in cur_gpu_list]
            cur_gpu_list = ','.join(cur_gpu_list)
            f.write("export USER_HIP_GPU_MAP={}\n".format(cur_gpu_list))
            f.write("srun --cpu-bind=${}CPU_BIND{} bin/{}_{} {}\n\n".format('{', '}', cur_name, cur_type, args))
            f.close()
            
            f_run_all.write("sbatch {}\n".format(sbatch_fname))
            # a+x
            st = os.stat(sbatch_fname)
            os.chmod(sbatch_fname, st.st_mode | stat.S_IEXEC)

f_run_all.close()
st = os.stat(f_run_all_name)
os.chmod(f_run_all_name, st.st_mode | stat.S_IEXEC)
            
        
