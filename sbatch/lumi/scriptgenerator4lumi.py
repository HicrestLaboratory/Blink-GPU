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
import os.path


names=["a2a", "ar", "mpp"]
types=["Baseline", "CudaAware", "Nccl"]
nodes_tasks = [(2, 4)]



args = ""
for cur_name in names:
    for cur_type in types:
        for cur_nodes, cur_ntasks_per_node in nodes_tasks:
            topo_name = "{}_{}".format(str(cur_nodes), str(cur_ntasks_per_node))
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
            f.write("#SBATCH --cpus-per-task=1\n\n")
            f.write("#SBATCH --account=project_465000997\n\n")
            f.write("mkdir -p sout\n\n")
            if cur_type == 'CudaAware':
                f.write("export MPICH_GPU_SUPPORT_ENABLED=1\n\n")
            f.write("srun bin/{}_{} {}\n\n".format(cur_name, cur_type, args))
            f.close()
            
            # a+x
            st = os.stat(sbatch_fname)
            os.chmod(sbatch_fname, st.st_mode | stat.S_IEXEC)
            
        
