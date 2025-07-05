# EpidemicSimulation

## Generate Graph
This script generates graphs
- To compile this code ```gcc generate_graph.c lib/cJSON.c -o generate_graph```
- To run ```./generate_graph <namegraph.json> <num_nodes> <num_edges>```

## SEQUENTIAL
This is the sequential code of the Epidemic Simulation
- To compile this code ```gcc SIR_sequential.cpp lib/cJSON.c -o SIR_sequential```
- To run ```./SIR_sequential <namegraph.json>```

## SIMD
- To compile this code ```g++ SIR_SIMD.cpp lib/cJSON.c -mavx2 -o SIR_SIMD```


## CUDA
- To compile this code ```nvcc ./kernel.cu lib/cJSON.c -o ./kernel```

### Versions:
- kernel_v1 : naïve version.
- kernel_v2 : grid dimension uptaded.
- kernel_v2_1 : *nodesPerWarp* is now a parameter.
- kernel_v2_2 : *xorshift32* introduced.
- kernel_v2_5 : pinned memory.
- kernel_opt : optimized version for multiple simulations on the same graph.

  kernel_v3_x are the same but with shared memory.
