#!/bin/bash

#SBATCH --job-name=TopologyFinder
#SBATCH --output=../outputs/TopologyFinder_%j.out
#SBATCH --error=../outputs/TopologyFinder_%j.err

#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_SHARP_0
#SBATCH --time=00:05:00
#SBATCH --qos=normal

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
    module load openmpi/

    echo " ------ TopologyFinder ------ "
    echo "       <some>: $some"

#     mkdir -p "outputs/$experimentname/finished/"


    srun topology-finder > ../outputs/TopologyFinder.out 2> ../outputs/TopologyFinder.err

    echo "------------------------"

# fi
