#!/bin/bash

#SBATCH --job-name=PingPong2nodes
#SBATCH --output=../outputs/PingPong_%j.out
#SBATCH --error=../outputs/PingPong_%j.err

#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_SHARP_0
#SBATCH --time=00:05:00
#SBATCH --qos=boost_qos_dbg

#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=494000MB

# if [[ $# -lt 1 ]]
# then
#     echo "usage: $0"
# else


    module load cuda/
    module load nccl/
    module load openmpi/

    echo " ------ PingPong ------ "
    echo "       <some>: $some"

#     mkdir -p "outputs/$experimentname/finished/"


    srun pingpong > ../outputs/PingPong2nodes.out 2> ../outputs/PingPong2nodes.err

    echo "------------------------"

# fi
