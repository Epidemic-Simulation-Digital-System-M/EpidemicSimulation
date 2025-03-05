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
int *Immune; // Stato di immunità passa da bool a int

#define TRUE -1
#define FALSE 0

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
    Immune = (int *)malloc(num_nodes * sizeof(int));

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
        Immune[i] = FALSE; // Non immune
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

void print__mm_register_ps(__m128 reg){
    printf("XMM[3]: %f\n",(float) _mm_extract_ps(reg, 3));
    printf("XMM[2]: %f\n",(float) _mm_extract_ps(reg, 2));
    printf("XMM[1]: %f\n",(float) _mm_extract_ps(reg, 1));
    printf("XMM[0]: %f\n",(float) _mm_extract_ps(reg, 0));
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
            if (Levels[i] == step && Immune[i] == FALSE)//controllo per il nodo 0
            { // Nodo infetto al passo corrente
                int num_neighbors = N[i + 1] - N[i];                
                int remainder = num_neighbors % 4;
                printf("num_neighbors: %d, remainder: %d\n", num_neighbors, remainder);

                __m128i neighbors;

                __m128 probability = _mm_set1_ps(p);

                for (int j = 0; j < num_neighbors; j+=4){

                    neighbors = _mm_loadu_si128((__m128i *)&L[N[i] + j]);

                    if(4*(j+1)>num_neighbors && remainder > 0){ //ultimo vettore
                        __m128i mask = _mm_set_epi32(0, remainder > 2 ? -1 : 0, remainder > 1 ? -1 : 0, -1 );
                        printf("Debug\n");

                        neighbors = _mm_and_si128(neighbors, mask);
                    }

                    printf("neighbors\n");
                    print__mm_register_epi32(neighbors);

                    __m128i Levels_SIMD = _mm_i32gather_epi32(Levels, neighbors, 4);
                    __m128i Immune_SIMD = _mm_i32gather_epi32(Immune, neighbors, 4);

                    printf("Levels_SIMD\n");
                    print__mm_register_epi32(Levels_SIMD);
                    printf("Immune_SIMD\n");
                    print__mm_register_epi32(Immune_SIMD);
                    
                    __m128i mask_susceptible = _mm_cmpeq_epi32(Levels_SIMD, _mm_set1_epi32(-1));
                    __m128i mask_not_immune = _mm_cmpeq_epi32(Immune_SIMD, _mm_setzero_si128());
                    __m128i infection_mask = _mm_and_si128(mask_susceptible, mask_not_immune);

                    printf("infection_mask\n");
                    print__mm_register_epi32(infection_mask);

                    __m128 random = _mm_set_ps((float)rand()/RAND_MAX,(float)rand()/RAND_MAX,(float)rand()/RAND_MAX,(float)rand()/RAND_MAX);
   
                    printf("random\n");
                    print__mm_register_ps(random);
   
                    __m128 infection_probability = _mm_cmplt_ps(random, probability);
   
                    printf("infection_probability\n");
                    print__mm_register_ps(infection_probability);

   
                    __m128i final_mask = _mm_and_si128(_mm_castps_si128(infection_probability), infection_mask);
   
                    printf("final_mask\n");
                    print__mm_register_epi32(final_mask);

                    __m128i neighbors_infected = _mm_and_si128(neighbors, final_mask);
                    //FIXME
                    _mm_i32scatter_epi32(Levels, neighbors_infected, _mm_set1_epi32(step + 1), 4);
                }
            }
            if (((double)rand() / RAND_MAX) < q) {
                Immune[i] = TRUE; // Nodo recuperato
                active_infections--;
            } else {
                Levels[i]=step+1; // Nodo può infettare anche al prossimo step
            }
        }
        step++;
        print_status(step, active_infections);
    }
}

int main(int argc, char *argv[])
{
    // Selezionando p=1 e q=1 otteniamo una ricerca in ampiezza
    double p = 0.5; // Probabilità di infezione
    double q = 1; // Probabilità di guarigione

    import_network(argv[1]);

    print_network();
    uint64_t clock_counter_start = __rdtsc();
    simulate(p, q);
    uint64_t clock_counter_end = __rdtsc();
    printf("Elapsed time: %lu\n", clock_counter_end - clock_counter_start);
    return 0;
}
