# Interconnect Benchmark

This repository was created to investigate and benchmark the state-of-the-art interconnections between a source and a target GPU. The code can benchmark both GPUs on the same computation
node then on different ones.

The base idea is to first initialize a message on the source device memory and then benchmark the time needed for moving the message to the target GPU/GPUs.

## Experiment set-up

For setting up the experiment we decided to always define one different MPI process for each involved GPU.

To better explain the process, we now fix some simple notations:
1. Let **m** be the message that needs to be communicated through the GPUs,
2. We name as source device (**SD**) the GPU that initially owns **m** in his device memory
3. We name as source host (**SH**) the MPI process associated with the source device **SD**
4. Likewise, we name as target device (**TD**) and target host (**TH**) the GPU and the associated MPI process that needs to receive **m**.

The investigated communication strategies are the following:
1. *Baseline*: the data **m** is first copied by **SD** to **SH** with a cudaMemcpy, then moved from **SH** to **TH** with an MPI primitive and finaly copied to **TD** with another cudaMemcpy. 
2. *CudaAware*: the source and target device pointers are directly provided to the MPI primitive which, since MPI is built with particular CudaAware support, is able to move the data from **SD** to **TD** without passing through the hosts **SH** and **TH**.
3. *NCCL*: The NVIDIA Collective Communication Library provide several primitives specialized for the GPU-GPU data exchange. As for the CudaAware strategy, the message **m** goes directly form **SD** to **TD** without involving **SH** and **TH**.
4. *NVLink* (cuda IPC): The CUDA InterProcess Communication is a family of low-level APIs that make it possible to share device pointers between GPUs associated with different MPI processes. When two GPUs which are connected through an NVLink bridge share the **m** device pointer, the target GPU **TD** is able to directly access (and copy) the message in **SD** with a DeviceToDevice cudaMemcpy call. Since that strategy requires a direct NVLink connection, it is applicable only for GPUs in the same computation node or under the same NVSwitch.

Depending on the particular experiment (i.e. communication scheme), the **SD** and the **TD** should be more than one. Moreover, each physical GPU is usually both a **SD** and a **TD** at the same time (for two different messages **m1** and **m2**). The implemented communication schemes are the following ones:
1. Peer-To-Peer (**pp**): a GPU *A* send a message **m1** to a GPU *B* that, after recived **m1**, sends a message **m2** to *A*.
2. All-To-All (**a2a**): all the **N** GPUs involved in the experiments send **N**-1 messages **m1**, **m2**, ..., **m(n-1)** (one to each of the other **N**-1 GPUs).
3. AllReduce (**ar**): given a message size **k** and a binary operation *Op*(), each GPU *i* owns a message **m(i)** of **k** elements; at the end of the communication each GPUs own the same message **M** = *Op*(**m(0)**, **m(1)**, ...,  **m(N-1)**) (where the operation *Op* is applied to the messages as an element-wise operation over the **k** elements).
4. Halo3D (**hlo**): The involved processes are divided over a 3D grid in which each process communicates with the adjacent ones; since we have three axes, each process should need to communicate with up to six other processes depending on the selected grid. Different to all the other communication schemes, the Halo3D involves non-blocking communications.
5. Multi-Peer-To-Peer (**mpp**): a set of multiple GPUs *A_0*, *A_1*, ... *A_k* send the messages **m1_0**, **m1_1**, ... **m1_k** to a disjoint set of GPUs *B_0*, *B_1*, ... *B_k* that, after recived **m1_0**, **m1_1**, ... **m1_k**, they send other messages **m2_0**, **m2_1**, ... **m2_k** to *A_0*, *A_1*, ... *A_k*.
6. Incast (**inc**): The **N-1** GPUs involved in the experiments send a message to the GPU with rank 0.
7. One-to-many (**otom**): The GPU with rank 0 sends a message to all the other **N-1** GPUs involved in the experiments.

# Repository structure

The repository is mainly divided into three different sections: Makefiles (located in the base directory), slurm scripts (located in "sbatch/<machine_name>") and source codes (located in "src/")

## Source codes

The source codes are CUDA C codes contained in the "src/" directory; for each communication primitive and each communication strategy, we have a different file named <primitive_id>_<strategy_id>.cu
...

### Base structure

...

### Command line parameters

All the source files admit several common optional parameters for custumize the execution:

1. With the flag "**-l**" you can custumize the repetition performed for each buffer-size (without setting "*-l*", the default value is *50*).
2. With the flag "**-b**" you can custumize the maximum buffer-size reached by the buffer cycle (without setting "*-b*", the default value should change between the experiments, but is usually around *30* (where we meen that the maximum buffer has a size of 2^30 B)).
3. With the flag "**-x**" you can fix a single buffer-size (as for the "*-b*" flag, the value define the 2^x B buffer size).

#### Special line parameters

Reguarding the *Halo3D* experiment, a customized 3D grid should be defined by using the falgs "**-pex**", "**-pey**" and "**-pez**".

Reguarding the *Multi-Peer-To-Peer* experiment, the number of Peer-To-Peer couples (by default as *4*) shoulb be customized with the falg "**-p**".

### Automatic correctness checks

For checking the correctness of the communication, each communication scheme implements a different check.

Regarding the Ping-pong and the All-to-all communication scheme, after initialising the send buffer with a fixed value, each process computes the sum reduction of the sending buffer and stores them inside "my_cpu_check". Once the benchmarked communication is completed, an analogue communication (a peer-to-peer or an all-to-all) is performed on "my_cpu_check" which will be stored in "recv_cpu_check". Finally, the device received buffer (owned by **TD**) is reduced inside "gpu_check" and the two elements are compared. The number reported as "Error" represents the absolute value of "recv_cpu_check" - "gpu_check".

Regarding the All-reduce communication scheme, the idea is similar, but the MPI operation performed on the "my_cpu_check" is an MPI_Allgather followed by a CPU reduction. In the end, the same reduction is performed on the device received buffer and the values are compared.

Finally, regarding the halo3d, each send buffer is initialized with the process MPI rank +1 on each entry, and, after the communication, is checked that each device received buffer reduction is equal to (source MPI rank+1)\*(message size). Since here up to 3\*2 different receive buffers (3 axes (x/y/z) times 2 directions (Up/Down)), the outputted error value compacts the comparison in a single int between 0 and 63 by using a bitwise or on a 6-bit unsigned int; the first two bits represents the correctness for the x axe, the second the one for y and the third for z.

## Building the binaries

Each machine has a dedicated Makefile contained inside the main directory and differenced by the extension ".<UPPERCASE_MACHINE_NAME>"; the binaries should be compiled by running:

  ```
  make -f Makefile.<UPPERCASE_MACHINE_NAME>
  ```

Furthermore, if we are interested in compiling only a particular experiment we can specify it by adding the experiment ID (*pp*, *a2a*, ...) at the end of the previous make command.

The Makefiles will generate several subfolders:
1. "*bin/*": which will contain all the binary files (one per each experiment-strategy couple),
2. "*moduleload/*": which will contain the files used for the module loading (one per each communication strategy),
3. "*exportload/*": which will contain the files used for setting up the environment variables (one per each communication strategy times the possible layouts (single or multiple nodes))
4. "*sout/*": which will contain the bash's standard outputs and standard errors.

For cleaning all the built directories and files the "clean" option is provided:

  ```
  make -f Makefile.<UPPERCASE_MACHINE_NAME> clean
  ```

### Building details

The Makefile was written for automatically importing the modules needed by each experiment instance and setting up the correct environment variables (defined but not really used until now).

The following code block reports the line needed for the NCCL compilation:

  ```
# ----------------- Nccl -----------------
$(NCCL_MODULES):
	mkdir -p ${MODULEFOLDER}
	@echo "module purge" > $@
	@echo "module load nvhpc/23.1" >> $@
	@echo "module load openmpi/4.1.4--nvhpc--23.1-cuda-11.8" >> $@
	@echo "module load nccl/2.14.3-1--gcc--11.3.0-cuda-11.8" >> $@

$(NCCL_SN_EXPORTS):
	mkdir -p ${EXPORTFOLDER}
	@echo "export NCCL_P2P_LEVEL=NVL" > $@

$(NCCL_MN_EXPORTS):
	mkdir -p ${EXPORTFOLDER}
	@echo "export NCCL_P2P_DISABLE=1" > $@

$(BINFOLDER)/$(NAME)_Nccl: src/$(NAME)_Nccl.cu $(NCCL_MODULES) $(NCCL_SN_EXPORTS) $(NCCL_MN_EXPORTS)
	mkdir -p $(BINFOLDER) $(OUTFOLDER) $(SOUTFOLDER)
	source $(NCCL_MODULES) ; $(CC) $(CFLAGS) -o $@ $< $(INCL) $(LIBS) $(LIBFLAGS) $(MPI) $(CUDA) $(NCCL) -lstdc++ -lm -Wno-deprecated-gpu-targets $(DBGFLAGS)

$(BINFOLDER)/$(A2ANAME)_Nccl: src/$(A2ANAME)_Nccl.cu $(NCCL_MODULES) $(NCCL_SN_EXPORTS) $(NCCL_MN_EXPORTS)
	mkdir -p $(BINFOLDER) $(OUTFOLDER) $(SOUTFOLDER)
	source $(NCCL_MODULES) ; $(CC) $(CFLAGS) -o $@ $< $(INCL) $(LIBS) $(LIBFLAGS) $(MPI) $(CUDA) $(NCCL) -lstdc++ -lm -Wno-deprecated-gpu-targets $(DBGFLAGS)

$(BINFOLDER)/$(HALONAME)_Nccl: src/$(HALONAME)_Nccl.cu $(NCCL_MODULES) $(NCCL_SN_EXPORTS) $(NCCL_MN_EXPORTS)
	mkdir -p $(BINFOLDER) $(OUTFOLDER) $(SOUTFOLDER)
	source $(NCCL_MODULES) ; $(CC) $(CFLAGS) -o $@ $< $(INCL) $(LIBS) $(LIBFLAGS) $(MPI) $(CUDA) $(NCCL) -lstdc++ -lm -Wno-deprecated-gpu-targets $(DBGFLAGS)

$(BINFOLDER)/$(ARNAME)_Nccl: src/$(ARNAME)_Nccl.cu $(NCCL_MODULES) $(NCCL_SN_EXPORTS) $(NCCL_MN_EXPORTS)
	mkdir -p $(BINFOLDER) $(OUTFOLDER) $(SOUTFOLDER)
	source $(NCCL_MODULES) ; $(CC) $(CFLAGS) -o $@ $< $(INCL) $(LIBS) $(LIBFLAGS) $(MPI) $(CUDA) $(NCCL) -lstdc++ -lm -Wno-deprecated-gpu-targets $(DBGFLAGS)
# ----------------------------------------
  ```

With "$(NCCL_MODULES)" we are creating a file containing all the modules needed by the NCCL binaries and loading with:

  ```
  source $(NCCL_MODULES) ; ...
  ```

## Slurm scripts

The slurm scripts are automatically generated by a stencil bash script that makes simple text replacements for adapting the stencil script to the different communication schemes and strategies.
Each different machine owns a different "sbatch/" subdirectory named as the machine name.

For an example, all the single bash scripts for Leonardo should be generated by running:

  ```
  ./sbatch/leonardo/scriptgenerator4leonardo.sh
  ```

which will generate the following scripts:

  ```
  user@machine:~/ping-pong-test$ ls sbatch/leonardo/
  run-leonardo-a2a-all.sh                   run-leonardo-hlo-Baseline-multinode.sh
  run-leonardo-a2a-Baseline-multinode.sh    run-leonardo-hlo-Baseline-singlenode.sh
  run-leonardo-a2a-Baseline-singlenode.sh   run-leonardo-hlo-CudaAware-multinode.sh
  run-leonardo-a2a-CudaAware-multinode.sh   run-leonardo-hlo-CudaAware-singlenode.sh
  run-leonardo-a2a-CudaAware-singlenode.sh  run-leonardo-hlo-Nccl-multinode.sh
  run-leonardo-a2a-Nccl-multinode.sh        run-leonardo-hlo-Nccl-singlenode.sh
  run-leonardo-a2a-Nccl-singlenode.sh       run-leonardo-pp-all.sh
  run-leonardo-a2a-Nvlink-multinode.sh      run-leonardo-pp-Baseline-multinode.sh
  run-leonardo-a2a-Nvlink-singlenode.sh     run-leonardo-pp-Baseline-singlenode.sh
  run-leonardo-ar-all.sh                    run-leonardo-pp-CudaAware-multinode.sh
  run-leonardo-ar-Baseline-multinode.sh     run-leonardo-pp-CudaAware-singlenode.sh
  run-leonardo-ar-Baseline-singlenode.sh    run-leonardo-pp-Nccl-multinode.sh
  run-leonardo-ar-CudaAware-multinode.sh    run-leonardo-pp-Nccl-singlenode.sh
  run-leonardo-ar-CudaAware-singlenode.sh   run-leonardo-pp-Nvlink-multinode.sh
  run-leonardo-ar-Nccl-multinode.sh         run-leonardo-pp-Nvlink-singlenode.sh
  run-leonardo-ar-Nccl-singlenode.sh        scriptgenerator4leonardo.sh
  run-leonardo-hlo-all.sh
  ```

As highlighted by the names, each experiment instance can be run with:

  ```
  sbatch sbatch/<machine_name>/run-<machine_name>-<experiment_id>-<scheme_id>-<layout>.sh
  ```

Moreover, each experiment has a dedicated bash script for submitting all the experiment instances together:

  ```
  ./sbatch/<machine_name>/run-<machine_name>-<experiment_id>-all.sh
  ```


## Output structure

Each SLURM script will collect the job results inside the "sout/" directory; the standard outputs under the extension "*.out*" and the standard errors under the extension "*.err*".

Summering up, each job will create the following outputs:

```
sout/<machine_name>_<experiment_id>_<strategy_id>_<layout>_<slurm_job_id>.out
sout/<machine_name>_<experiment_id>_<strategy_id>_<layout>_<slurm_job_id>.err
```
where the layout is one between "singlenode" and "multinode".

If no error occurs, the outfile will contain a first part with some compile and runtime information, followed by the results:

```
...
	Transfer size (B):        100, Transfer Time (s):     0.000042809, Bandwidth (GB/s):     0.002175530, Iteration 41
	Transfer size (B):        100, Transfer Time (s):     0.000043662, Bandwidth (GB/s):     0.002133028, Iteration 42
	Transfer size (B):        100, Transfer Time (s):     0.000043102, Bandwidth (GB/s):     0.002160741, Iteration 43
	Transfer size (B):        100, Transfer Time (s):     0.000043581, Bandwidth (GB/s):     0.002136992, Iteration 44
	Transfer size (B):        100, Transfer Time (s):     0.000043684, Bandwidth (GB/s):     0.002131954, Iteration 45
	Transfer size (B):        100, Transfer Time (s):     0.000044896, Bandwidth (GB/s):     0.002074400, Iteration 46
	Transfer size (B):        100, Transfer Time (s):     0.000044200, Bandwidth (GB/s):     0.002107065, Iteration 47
	Transfer size (B):        100, Transfer Time (s):     0.000043890, Bandwidth (GB/s):     0.002121947, Iteration 48
	Transfer size (B):        100, Transfer Time (s):     0.000042931, Bandwidth (GB/s):     0.002169347, Iteration 49
[Average] Transfer size (B):        100, Transfer Time (s):     0.000045054, Bandwidth (GB/s):     0.002067121, Error: 0
	Transfer size (B):        200, Transfer Time (s):     0.000043259, Bandwidth (GB/s):     0.004305798, Iteration 0
	Transfer size (B):        200, Transfer Time (s):     0.000046074, Bandwidth (GB/s):     0.004042725, Iteration 1
	Transfer size (B):        200, Transfer Time (s):     0.000045263, Bandwidth (GB/s):     0.004115161, Iteration 2
	Transfer size (B):        200, Transfer Time (s):     0.000046979, Bandwidth (GB/s):     0.003964846, Iteration 3
...
```

as we can see by the reported example, the main outputted metrics are:
1. the *Transfer size* in Byte sent by each process (this value is cumulative, so if we send multiple messages to multiple GPUs that value is the sum of all the sent messages),
2. the *Transfer Time* in seconds needed for completing the communication,
3. the *Bandwidth* in GB/s obtained by the previous values.
   
Moreover, we have two different types of result lines:
1. the *iteration lines*: which contain the values related to a single iteration (which also contain the iteration number),
2. the *average lines*: which contains the average of all the iterations performed on the same message size (and contains the aggregated error computed according to the "correctness check" section).

*Notes*:
1. Some iteration has a negative iteration number; those ones represent the warm-up iteration and are not involved in the average computation.
2. the reported times are always referred to as the max time over all the involved MPI process time.
