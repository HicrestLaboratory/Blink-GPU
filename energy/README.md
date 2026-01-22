# MPI-GPU Collective Energy Characterization
This repo contains the benchmarks used to define an energy characterization of Multi-GPU MPI collectives by using proprietary collective libraries from NVIDIA (NCCL), AMD (RCCL), and Intel (oneCCL). 
We defined an energy and power profiler calle PowerProfiler that wrap aroung existing power and energy interfaces providfed by GPU/CPU vendors (e.g., NVML for NVIDIA GPUs, RSMI for AMD GPUs, Level Zero for Intel GPUs and RAPL for Intel CPUs) to provide a unified interface to measure energy and power consumption of parallel applications running on CPU and  GPU systems.
The PowerProfiler is available as submodule in this repo and can be found at energy/energy-profiler.
Based on the PowerProfiler interface we defined a benchamrk methodology to provide accurate energy measurement of collectives. 
For each run of the benchmark a collective is executed N times by building a chain so that the total execution time of the chain is at least X seconds (this parameter depends on the precision of energy counter of the target GPU and can be measured experimentally by running the follwing test: XXX).
The energy consumption of the collective is then computed as the difference between the energy measured at the beginning and at the end of the chain execution divided by N.
The collectives are divided in three folders (nccl, rccl, occl) based on the underlying collective library used: NCCL, RCCL, and oneCCL.
The include folder contains a Logger class that can be used to log all the time, energy and info related to rank for each collective.

## How to build
To build the benchmarks, first clone the repository:
```bash
git clone <blink_gpu_repo_url> 
``` 
Then, initialize the submodules to include also the PowerProfiler interface:
```bash
git submodule update --init --recursive 
``` 
Build the PowerProfiler by follwoing the instructions in energy/energy-profiler/README.md.
