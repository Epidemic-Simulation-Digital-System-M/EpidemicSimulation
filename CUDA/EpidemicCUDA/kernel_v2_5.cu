
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <device_atomic_functions.h>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include "lib/cJSON.h"
#include <emmintrin.h>
#include <immintrin.h>

#define MAX_NODES 1000
#define MAX_EDGES 10000

int *N;                 // Indici dell'inizio dei vicini per ogni nodo
int *L;                 // Lista di adiacenza compressa
int *Levels;            // Momento dell'infezione: istante in cui viene infettato
bool *Immune;           // Stato di immunità
int *active_infections; // Numero di infezioni attive

int num_nodes;
int num_edges;
int size_L;

double cpuSecond()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

char *read_file(const char *filename)
{
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        printf("Error opening file!\n");
        return NULL;
    }

    char *json_string = NULL;
    size_t size = 0;
    size_t capacity = 128; // Initial buffer size
    json_string = (char *)malloc(capacity);
    if (!json_string)
    {
        printf("Memory allocation failed!\n");
        fclose(file);
        return NULL;
    }

    int ch;
    while ((ch = fgetc(file)) != EOF)
    {
        json_string[size++] = (char)ch;
        // Resize buffer if needed
        if (size >= capacity - 1)
        {
            capacity *= 2; // Double the buffer size
            json_string = (char *)realloc(json_string, capacity);
            if (!json_string)
            {
                printf("Memory reallocation failed!\n");
                fclose(file);
                return NULL;
            }
        }
    }
    json_string[size] = '\0'; // Null-terminate the string
    fclose(file);
    return json_string;
}

void import_network(const char *filename)
{
    char filepath[256];
    snprintf(filepath, sizeof(filepath), "../../GRAPH_GENERATOR/%s", filename);
    char *json_string = read_file(filepath);
    if (!json_string)
    {
        exit(1);
    }

    cJSON *root = cJSON_Parse(json_string);
    free(json_string); // Free memory after parsing
    if (!root)
    {
        printf("Error parsing JSON!\n");
        return;
    }

    cJSON *json_numNodes = cJSON_GetObjectItem(root, "num_nodes");
    cJSON *json_numEdges = cJSON_GetObjectItem(root, "num_edges");
    num_nodes = json_numNodes->valueint;
    num_edges = json_numEdges->valueint;

    // Extract arrays
    cJSON *json_N = cJSON_GetObjectItem(root, "N");
    cJSON *json_L = cJSON_GetObjectItem(root, "L");

    int size_N = cJSON_GetArraySize(json_N);
    size_L = cJSON_GetArraySize(json_L);

    cudaMallocHost((void **)&N, size_N * sizeof(int));
    cudaMallocHost((void **)&L, size_L * sizeof(int));
    cudaMallocHost((void **)&Levels, num_nodes * sizeof(int));
    cudaMallocHost((void **)&Immune, num_nodes * sizeof(bool));

    for (int i = 0; i < size_N; i++)
    {
        N[i] = cJSON_GetArrayItem(json_N, i)->valueint;
    }
    for (int i = 0; i < size_L; i++)
    {
        L[i] = cJSON_GetArrayItem(json_L, i)->valueint;
    }

    for (int i = 0; i < num_nodes; i++)
    {
        Levels[i] = -1;    // Non infetto
        Immune[i] = false; // Non immune
    }
    Levels[0] = 0; // Nodo inizialmente infetto al tempo 0
}

void print_network()
{
    printf("Network:\n");
    for (int i = 0; i < num_nodes; i++)
    {
        printf("%d: ", i);
        for (int j = N[i]; j < N[i + 1]; j++)
        {
            printf("%d ", L[j]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_status(int step, int active_infections, int *d_Levels)
{
    printf("Step %d: %d active infections\n", step, active_infections);
    if (active_infections > 0)
    {
        cudaMemcpy(Levels, d_Levels, num_nodes * sizeof(int), cudaMemcpyDeviceToHost);
        printf("Infected nodes: ");
        for (int i = 0; i < num_nodes; i++)
        {
            if (Levels[i] == step)
            {
                printf("%d ", i);
            }
        }
        printf("\n");
    }
}

__device__ uint32_t xorshift32(uint32_t &state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

__device__ float rand_uniform(uint32_t &state)
{
    return (xorshift32(state) & 0xFFFFFF) / float(0x1000000);
}

__global__ void simulate_step(int *d_N, int *d_L, int *d_Levels, bool *d_Immune, int num_nodes, double p, double q, int step, int *d_active_infections, int nodesPerWarp)
{
    int tid_in_warp = (threadIdx.x) % 32;

    int warp_id = (threadIdx.x + blockIdx.x * blockDim.x) / 32;

    int start_index = warp_id * nodesPerWarp;

    int final_index = start_index + nodesPerWarp;
    if (final_index > num_nodes)
    {
        final_index = num_nodes;
    }

    uint32_t prng_state = ((uintptr_t)&prng_state) + (threadIdx.x + blockIdx.x * blockDim.x) + (step + 1);

    for (int i = start_index; i < final_index; i++)
    {
        if (d_Levels[i] == step)
        { // Il nodo è infetto

            for (int j = d_N[i] + tid_in_warp; j < d_N[i + 1]; j += 32)
            {
                int neighbor = d_L[j];
                // printf("Thread %d: Nodo %d Vicino %d\n", tid_in_warp,i, neighbor);
                // printf("Random %f\n", rand_uniform(prng_state));
                if (d_Levels[neighbor] == -1 && !d_Immune[neighbor] && (rand_uniform(prng_state) < p))
                {
                    // Infetto al prossimo step
                    // Usa atomicCAS per evitare doppie infezioni
                    int old_level = atomicCAS(&d_Levels[neighbor], -1, step + 1);
                    if (old_level == -1)
                    { // Solo il primo thread che infetta il nodo lo conta
                        atomicAdd(d_active_infections, 1);
                        // printf("Thread %d Blocco %d: Nodo %d infetta %d\n", threadIdx.x,blockIdx.x, i, neighbor);
                    }
                }
            }

            if (tid_in_warp == 0)
            {
                if (rand_uniform(prng_state) < q)
                {
                    d_Immune[i] = true; // Nodo recuperato
                    atomicSub(d_active_infections, 1);
                    // printf("Thread %d Blocco %d: Nodo %d guarito\n", threadIdx.x, blockIdx.x, i);
                }
                else
                {
                    d_Levels[i] = step + 1; // Nodo può infettare anche al prossimo step
                    // printf("Thread %d Blocco %d: Nodo %d rimane infetto\n", threadIdx.x, blockIdx.x, i);
                }
            }
        }
    }
}

void simulate(double p, double q)
{
    int step = 0;
    *active_infections = 1;

    // Device variables
    int *d_N, *d_L, *d_Levels;
    bool *d_Immune;
    int *d_active_infections;

    cudaMalloc(&d_N, (num_nodes + 1) * sizeof(int));
    cudaMalloc(&d_L, size_L * sizeof(int));
    cudaMalloc(&d_Levels, num_nodes * sizeof(int));
    cudaMalloc(&d_Immune, num_nodes * sizeof(bool));
    cudaMalloc(&d_active_infections, sizeof(int));

    cudaMemcpy(d_N, N, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_L, L, size_L * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Levels, Levels, num_nodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Immune, Immune, num_nodes * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_active_infections, active_infections, sizeof(int), cudaMemcpyHostToDevice);

    // print_status(step, active_infections, d_Levels);

    int threadsPerBlock = 64;
    int nodesPerWarp = 3;
    // int gridSize = (num_nodes + threadsPerBlock - 1) / threadsPerBlock;
    int warpNumber = (num_nodes + nodesPerWarp - 1) / nodesPerWarp;
    int gridSize = (warpNumber + threadsPerBlock / 32 - 1) * 32 / threadsPerBlock;

    while (*active_infections > 0)
    {

        simulate_step<<<gridSize, threadsPerBlock>>>(d_N, d_L, d_Levels, d_Immune, num_nodes, p, q, step, d_active_infections, nodesPerWarp);
        cudaDeviceSynchronize();
        cudaMemcpy(active_infections, d_active_infections, sizeof(int), cudaMemcpyDeviceToHost);

        step++;
        // print_status(step, *active_infections, d_Levels);
    }

    //cudaMemcpy(Levels, d_Levels, num_nodes * sizeof(int), cudaMemcpyDeviceToHost);
    //cudaMemcpy(Immune, d_Immune, num_nodes * sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_N);
    cudaFree(d_L);
    cudaFree(d_Levels);
    cudaFree(d_Immune);
    cudaFree(d_active_infections);
    cudaFreeHost(active_infections);
}

int main(int argc, char *argv[])
{
    // Selezionando p=1 e q=1 otteniamo una ricerca in ampiezza
    double p = 1; // Probabilità di infezione
    double q = 1; // Probabilità di guarigione

    double start_import = cpuSecond();
    cudaMallocHost((void **)&active_infections, sizeof(int));
    import_network(argv[1]);
    double end_import = cpuSecond();
    printf("Import time: %f seconds\n", end_import - start_import);

    // print_network();

    printf("#1------------\n");
    printf("Simulating with p=%f, q=%f\n", p, q);
    double start = cpuSecond();
    simulate(p, q);
    double end = cpuSecond();
    printf("Elapsed time: %f seconds\n", end - start);
    printf("-------------\n");

    printf("#2\n");
    printf("Simulating with p=%f, q=%f\n", p, q);
    start = cpuSecond();
    simulate(p, q);
    end = cpuSecond();
    printf("Elapsed time: %f seconds\n", end - start);
    printf("-------------\n");

    printf("#3\n");
    printf("Simulating with p=%f, q=%f\n", p, 0.8);
    start = cpuSecond();
    simulate(p, 0.8);
    end = cpuSecond();
    printf("Elapsed time: %f seconds\n", end - start);
    printf("-------------\n");

    printf("#4\n");
    printf("Simulating with p=%f, q=%f\n", 0.8, 0.6);
    start = cpuSecond();
    simulate(0.8, 0.6);
    end = cpuSecond();
    printf("Elapsed time: %f seconds\n", end - start);
    printf("-------------\n");

    cudaFreeHost(N);
    cudaFreeHost(L);
    cudaFreeHost(Levels);
    cudaFreeHost(Immune);

    return 0;
}
