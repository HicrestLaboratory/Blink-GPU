#!/bin/bash

if [ "$#" -lt 6 ]; then
    echo "Usage: $0 R C Q P H M"
    echo "R: Number of process rows"
    echo "C: Number of process cols"
    echo "Q: One between the boost_qos_dbg/normal/boost_qos_bprod queues"
    echo "P: project-id (to put as --account value)"
    echo "H: max hours time"
    echo "M: max minutes time"
    exit 1
fi

R=$1
C=$2
Q=$3
P=$4
H=$5
M=$6

if [[ "$Q" != "boost_qos_dbg" ]] && [[ "$Q" != "normal" ]] && [[ "$Q" != "boost_qos_bprod" ]]; then
    echo "Q must be one between 'boost_qos_dbg', 'normal' and 'boost_qos_bprod'"
    exit 1
fi

nprocs=$(($R * $C))
nprocmod=$(( nprocs % 4 ))
nprocdiv=$(( nprocs / 4 ))
echo "R = $R, C = $C, nprocs = $nprocs, nprocmod = $nprocmod, nprocdiv = $nprocdiv"

if [[ "$nprocs" != "1" ]] && [[ "$nprocs" != "2" ]] && [[ "$nprocmod" != "0" ]]; then
    echo "RxC must be one, two or a multiple of 4"
    exit 1
fi

# Original script content
script_content=$(cat << 'EOF'
#!/bin/bash

#SBATCH --job-name=AXBC_1x1
#SBATCH --output=outputs/tmp/AXBC_1x1_%j.out
#SBATCH --error=outputs/tmp/AXBC_1x1_%j.err

#SBATCH --partition=boost_usr_prod
#SBATCH --account=<project-id>
#SBATCH --time=00:05:00
#SBATCH --qos=normal

#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=494000MB


if [[ $# -lt 3 ]]
then
    echo "usage: AXBC_1x1 path2graph nodesnum distribution eparameter xparameter [outname] [experimentname] [maxsamples] [Rvalue] [logical_partitioning]"
else

    path2graph=$1
    nodesnum=$2
    distribution=$3
    eparameter=$4
    xparameter=$5

    outname=tmp-$nodesnum-$distribution-$xparameter
    if [[ $# -gt 5 ]]
    then
        outname=$6
    fi

    echo "$path2graph" | sed 's|.*/||' | cut -d'.' -f1 > name_$outname.tmp
    matrixname=$(<name_$outname.tmp)
    rm name_$outname.tmp

    experimentname="tmpExp"
    if [[ $# -gt 6 ]]
    then
        experimentname=$7
    fi

    maxsamples=$((nodesnum/10))
    if [[ $# -gt 7 ]]
    then
        maxsamples=$8
    fi

    Rvalue=0
    if [[ $# -gt 8 ]]
    then
        Rvalue=$9
    fi

    logpart="1x1"
    if [[ $# -gt 9 ]]
    then
        logpart=${10}
    fi

    module load cuda/
    module load openmpi/

    echo " ------ AXBC_1x1 ------ "
    echo "    path2graph: $path2graph"
    echo "    matrixname: $matrixname"
    echo "experimentname: $experimentname"
    echo "       outname: $outname"
    echo "        Rvalue: $Rvalue"
    echo "       logpart: $logpart"

    mkdir -p "outputs/$experimentname/$matrixname/"
    mkdir -p "outputs/sout/$experimentname/$matrixname/"
    mkdir -p "outputs/serr/$experimentname/$matrixname/"
    mkdir -p "outputs/$experimentname/notSubmitted/"
    mkdir -p "outputs/$experimentname/submitted/"
    mkdir -p "outputs/$experimentname/finished/"


    echo "$outname" >> "outputs/$experimentname/submitted/$matrixname.txt"

    if [[ $xparameter > 0 ]]
    then
        srun --mpi=pmi2 src/axbc2d -p 1x1 -L $logpart -f $path2graph -n $nodesnum -c 1 -x $xparameter -z $distribution -e $eparameter -N $maxsamples -H 1 -R $Rvalue -o outputs/$experimentname/$matrixname/$outname > outputs/sout/$experimentname/$matrixname/$outname.out 2> outputs/serr/$experimentname/$matrixname/$outname.err
    else
        srun --mpi=pmi2 src/axbc2d -p 1x1 -L $logpart -f $path2graph -n $nodesnum -c 0 -z $distribution -e $eparameter -N $maxsamples -H 1 -R $Rvalue -o outputs/$experimentname/$matrixname/$outname > outputs/sout/$experimentname/$matrixname/$outname.out 2> outputs/serr/$experimentname/$matrixname/$outname.err
    fi

    if [[ $? == 0 ]]
    then
        echo "$outname" >> "outputs/$experimentname/finished/$matrixname.txt"
        acct=$(sacct --jobs=${SLURM_JOB_ID} --format=jobid,jobname,ntasks,AveCPU,MaxDiskWrite,MaxDiskRead,ConsumedEnergy,Nodelist)
        echo "${acct}" | grep "axbc2d" >> "outputs/$experimentname/finished/${matrixname}_sacct.txt"
        echo "${acct}"
    else
        echo "$outname not written in 'outputs/$experimentname/finished/$matrixname.txt' since the exit code is different form 0 ($?)"
    fi


    echo "------------------------"

fi
EOF
)

# Create a copy of the script by replacing "AXBC_1x1" with "AXBC_RxC"
if [[ "$Q" == "normal" ]]
then
    new_script_content=$(echo "$script_content" | sed "s/-p 1x1/-p "$R"x$C/g" | sed "s/AXBC_1x1/AXBC_"$R"x$C/g" | sed "s/<project-id>/$P/g" | sed "s/time=00:05:00/time="$H":"$M":00/g")
else
    new_script_content=$(echo "$script_content" | sed "s/-p 1x1/-p "$R"x$C/g" | sed "s/AXBC_1x1/AXBC_"$R"x"$C"/g" | sed "s/<project-id>/$P/g" | sed "s/normal/$Q/g" | sed "s/time=00:05:00/time="$H":"$M":00/g")
fi
script_content=$(echo "$new_script_content")

if [[ "$nprocdiv" != "0" ]]
then
    new_script_content=$(echo "$script_content" | sed "s/nodes=1/nodes=$nprocdiv/g")
fi
script_content=$(echo "$new_script_content")

if [[ "$nprocs" == "1" ]] || [[ "$nprocs" == "2" ]]
then
    new_script_content=$(echo "$script_content" | sed "s/gres=gpu:1/gres=gpu:$nprocs/g" | sed "s/ntasks-per-node=1/ntasks-per-node=$nprocs/g")
fi

# Write the new script to a file
new_script_file="axbc_leonardo_"$R"x$C$Q$H$M"
echo "$new_script_content" > "$new_script_file"

# Make the new script executable
chmod +x "$new_script_file"

echo "Generated $new_script_file"

