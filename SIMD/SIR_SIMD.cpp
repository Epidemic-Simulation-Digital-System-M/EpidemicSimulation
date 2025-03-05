#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include "lib/cJSON.h"
#include <emmintrin.h>
#include <immintrin.h>

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

int *N;       // Indici dell'inizio dei vicini per ogni nodo
int *L;       // Lista di adiacenza compressa
int *Levels;  // Momento dell'infezione: istante in cui viene infettato
bool *Immune; // Stato di immunità

int num_nodes;
int num_edges;

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
    char *json_string = read_file(filename);
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
    int size_L = cJSON_GetArraySize(json_L);

    N = (int *)malloc(size_N * sizeof(int));
    L = (int *)malloc(size_L * sizeof(int));
    Levels = (int *)malloc(num_nodes * sizeof(int));
    Immune = (bool *)malloc(num_nodes * sizeof(bool));

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

void print_status(int step, int active_infections)
{
    printf("Step %d: %d active infections\n", step, active_infections);
    if (active_infections > 0)
    {
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

void print__mm_register_epi32(__m128i reg){
    printf("XMM[3]: %d\n", _mm_extract_epi32(reg, 3));
    printf("XMM[2]: %d\n", _mm_extract_epi32(reg, 2));
    printf("XMM[1]: %d\n", _mm_extract_epi32(reg, 1));
    printf("XMM[0]: %d\n", _mm_extract_epi32(reg, 0));
}

void simulate(double p, double q)
{
    int active_infections = 1;
    int step = 0;

    // print_status(step, active_infections);

    while (active_infections > 0)
    {
        for (int i = 0; i < num_nodes; i++)
        {
            if (Levels[i] == step)
            { // Nodo infetto al passo corrente
                int num_neighbors = N[i + 1] - N[i];              

                __m128i neighbor = _mm_loadu_si128((__m128i *)&L[N[i]]);

                int remainder = num_neighbors % 4;
                
                if(remainder > 0){
                    __m128i mask = _mm_set_epi32(0, remainder > 2 ? -1 : 0, remainder > 1 ? -1 : 0, -1);
                    neighbor = _mm_and_si128(neighbor, mask);
                }

                print__mm_register_epi32(neighbor);

               /*
               TODO
                -scatter gather per Levels e Immune
                -calcolare se il nodo è infetto o immmune -> maschera 
                -vettore di rand-> aggiornare Levels 
                -gesitire iterazioni                 
               */

            }
        }
        step++;
        // print_status(step, active_infections);
    }
}

int main(int argc, char *argv[])
{
    // Selezionando p=1 e q=1 otteniamo una ricerca in ampiezza
    double p = 1; // Probabilità di infezione
    double q = 1; // Probabilità di guarigione

    import_network(argv[1]);

    print_network();
    uint64_t clock_counter_start = __rdtsc();
    simulate(p, q);
    uint64_t clock_counter_end = __rdtsc();
    printf("Elapsed time: %lu\n", clock_counter_end - clock_counter_start);
    return 0;
}
