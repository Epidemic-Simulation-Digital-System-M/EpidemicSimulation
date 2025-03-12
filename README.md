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

