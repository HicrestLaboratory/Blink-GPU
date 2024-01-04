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
**Last update**: 4/1/2024 10:15

| Benchmark   | Layout              | MPI + CudaMemcpy | NCCL            | GPUDirect       | Low-level NV-link |
|-------------|---------------------|------------------|-----------------|-----------------|-------------------|
| Ping-pong   | IntraNode           | 🟢               | 🟢               | ❌              | 🟢                |
|             | InterNodes          | 🟢               | 🟢               | ❌              | ❌                |
| Halo3d      | IntraNode  (2x2x1)  | 🟢               | 🟢               | ❌              | 🟢                |
|             | InterNodes (2x2x2)  | 🟢               | 🟢               | ❌              | ❌                |
| Incast      | IntraNode           | 🟢               | 🟢               | ❌              | 🟢                |
|             | InterNodes          | 🟢               | 🟢               | ❌              | ❌                |
| Sweep3d     | IntraNode           | 🔵               | 🔵               | ❌              | 🔵                |
|             | InterNodes          | 🔵               | 🔵               | ❌              | ❌                |


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


