#!/bin/bash

#SBATCH --job-name=PingPong
#SBATCH --output=../outputs/PingPong_%j.out
#SBATCH --error=../outputs/PingPong_%j.err

#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --partition=short
#SBATCH --gres=gpu:2

# if [[ $# -lt 1 ]]
# then
#     echo "usage: $0"
# else


    module load cuda/
    module load openmpi/

    echo " ------ PingPong ------ "
    echo "       <some>: $some"

#     mkdir -p "outputs/$experimentname/finished/"


    srun --mpi=pmix pingpong > ../outputs/PingPong.out 2> ../outputs/PingPong.err

    echo "------------------------"

# fi
