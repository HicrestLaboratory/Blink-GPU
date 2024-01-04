# Current status of development

## Key of the symbols

| Status                       | Symbol  |
|------------------------------|---------|
| To be defined or issues      | ❌      |
| Not implemented              | □       |
| In development               | ▶       |
| Implemented                  | 🔵      |
| To test                      | 🟢      |
| Tested                       | ✅      |

1. **To be defined or issues**: the communication scheme or the communication technology is not well defined yet (or there are issues in implementing it).
2. **Not implemented**: the communication scheme and the communication technology are well defined but the experiment is not implemented yet.
3. **In development**: the implementation has started.
4. **Implemented**: the core of the code is implemented but should miss some minor correctness tests, timers or data collectives.
5. **To test**: the full code is implemented and ready to be adapted and tested on each architecture.
6. **Tested**: the experiment was tested on each considered architecture.

## Status of development
**Last update**: 4/1/2024 12:30

| Benchmark   | Layout              | MPI              | NCCL              | GPUDirect       | Low-level NV-link  |
|-------------|---------------------|------------------|-------------------|-----------------|--------------------|
| Ping-pong   | IntraNode           | 🟢               | 🟢               | ❌              | 🟢                |
|             | InterNodes          | 🟢               | 🟢               | ❌              | ❌                |
| Halo3d      | IntraNode  (2x2x1)  | 🟢               | 🟢               | ❌              | 🟢                |
|             | InterNodes (2x2x2)  | 🟢               | 🟢               | ❌              | ❌                |
| Incast      | IntraNode           | 🟢               | 🟢               | ❌              | 🟢                |
|             | InterNodes          | 🟢               | 🟢               | ❌              | ❌                |
| Sweep3d     | IntraNode           | 🟢               | 🟢               | ❌              | 🟢                |
|             | InterNodes          | 🟢               | 🟢               | ❌              | ❌                |

### Short description

Short description regarding the different implementations:
1. **MPI**: this is the base implementation. The send buffer is first copied from the GPU to the CPU with 'cudaMemcpy', then the CPU send buffer is shared with the other processes with the MPI primitives and at the end, the receive buffer is copied from the CPU to the target GPU.
2. **NCCL**: this implementation uses the NVIDIA Collective Communications Library. The NCCL primitives can share data between GPUs without using any explicit CPU buffer. It works both for GPUs on the same node and GPUs on different nodes (with different performances).
3. **GPUDirect**: ...
4. Low-level **NV-link**: this implementation first uses the NVIDIA InterProcess Communication primitives (IPC) to enable peer access to other GPUs associated with other MPI tasks, then share the IPC pointers with the standard MPI collectives and, in the end, copy the data from the sender to the receiver GPU with a direct cudaMemcpy device to device. 


### Testing status

This table reports the testing status for all the experiments "*To test*" (🟢) and "*Tested*" (✅).

| Benchmark   | Layout       | MPI + CudaMemcpy |  |  | NCCL |  |  | GPUDirect |  |  | Low-level NV-link |  |  |
|-------------|--------------|----------|-----------|-------|----------|-----------|-------|-----------|----------|-------|----------|-----------|-------|
|             |              | Leonardo | UNITN_DGX | GH200 | Leonardo | UNITN_DGX | GH200 | Leonardo | UNITN_DGX | GH200 | Leonardo | UNITN_DGX | GH200 |
| Ping-pong   | Intra-node   | □        | □         | □     | □        | □         | □     | □         | □        | □     | □        | □         | □     |
|             | Inter-nodes  | □        | □         | □     | □        | □         | □     | □         | □        | □     | □        | □         | □     |
| Halo3d      | 2x2x2        | □        | □         | □     | □        | □         | □     | □         | □        | □     | □        | □         | □     |
|             | ??           | □        | □         | □     | □        | □         | □     | □         | □        | □     | □        | □         | □     |
| Incast      | Inter-nodes  | □        | □         | □     | □        | □         | □     | □         | □        | □     | □        | □         | □     |
|             | ??           | □        | □         | □     | □        | □         | □     | □         | □        | □     | □        | □         | □     |
| Sweep3d     | ??           | □        | □         | □     | □        | □         | □     | □         | □        | □     | □        | □         | □     |
|             | ??           | □        | □         | □     | □        | □         | □     | □         | □        | □     | □        | □         | □     |
| Others (?)  | ??           | □        | □         | □     | □        | □         | □     | □         | □        | □     | □        | □         | □     |
|             | ??           | □        | □         | □     | □        | □         | □     | □         | □        | □     | □        | □         | □     |


