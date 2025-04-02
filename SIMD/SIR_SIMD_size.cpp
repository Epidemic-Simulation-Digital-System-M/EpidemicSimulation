#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include "lib/cJSON.h"
#include <emmintrin.h>
#include <immintrin.h>
#include <random>
#include <cstdint>

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

int *N;      // Indici dell'inizio dei vicini per ogni nodo
uint8_t *L;      // Lista di adiacenza compressa
//int8_t  *Levels; // Momento dell'infezione: istante in cui viene infettato
int* Levels;
int* Immune;
//int8_t  *Immune; // Stato di immunità passa da bool a int

#define TRUE -1
#define FALSE 0
#define AVX_DATALANE 32
#define INTSIZE 8 
#define REGISTER_LENGTH 256

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
    char filepath[256];
    snprintf(filepath, sizeof(filepath), "../GRAPH_GENERATOR/%s", filename);
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
    int size_L = cJSON_GetArraySize(json_L);

    // N = (int *)malloc(size_N * sizeof(int));
    // L = (int *)malloc(size_L * sizeof(int));

    // Levels = (int *)malloc(num_nodes * sizeof(int));
    // Immune = (int *)malloc(num_nodes * sizeof(int));

    N = (int *)_mm_malloc(size_N * sizeof(int), AVX_DATALANE);
    L = (uint8_t  *)_mm_malloc(size_L * sizeof(uint8_t ), AVX_DATALANE);

    Levels = (int  *)_mm_malloc(num_nodes * sizeof(int), AVX_DATALANE);
    Immune = (int  *)_mm_malloc(num_nodes * sizeof(int), AVX_DATALANE);

    for (int i = 0; i < size_N; i++)
    {
        N[i] = cJSON_GetArrayItem(json_N, i)->valueint;
    }
    for (int i = 0; i < size_L; i++)
    {
        L[i] = (uint8_t)cJSON_GetArrayItem(json_L, i)->valueint;
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
        printf("Infected nodes: \n");
        for (int i = 0; i < num_nodes; i++)
        {
            if (Levels[i] == step)
            {
                printf("%d ", i);
            }
            else{}
        }
        printf("\n");
        /*
        printf("Other Nodes: \n");
        for (int i = 0; i < num_nodes; i++)
        {
            if (Levels[i] != step)
            {
                printf("%d ", i);
            }
        }
        printf("\n");
        */
    }
}

void print__mm_register_ps(__m256 reg)
{
    float values[8];
    _mm256_storeu_ps(values, reg); // Memorizza il registro in un array
    for (int i = 0; i < 8; i++)
    {
        printf("XMM[%d]: %f\n", i, values[i]);
    }
}

void print__mm_register_epi32(__m256i reg)
{
    printf("XMM[7]: %d\n", _mm256_extract_epi32(reg, 7));
    printf("XMM[6]: %d\n", _mm256_extract_epi32(reg, 6));
    printf("XMM[5]: %d\n", _mm256_extract_epi32(reg, 5));
    printf("XMM[4]: %d\n", _mm256_extract_epi32(reg, 4));
    printf("XMM[3]: %d\n", _mm256_extract_epi32(reg, 3));
    printf("XMM[2]: %d\n", _mm256_extract_epi32(reg, 2));
    printf("XMM[1]: %d\n", _mm256_extract_epi32(reg, 1));
    printf("XMM[0]: %d\n", _mm256_extract_epi32(reg, 0));
}

void print__mm_register_epi8(__m256i reg)
{
    printf("XMM[31]: %d\n", _mm256_extract_epi8(reg, 31));
    printf("XMM[30]: %d\n", _mm256_extract_epi8(reg, 30));
    printf("XMM[29]: %d\n", _mm256_extract_epi8(reg, 29));
    printf("XMM[28]: %d\n", _mm256_extract_epi8(reg, 28));
    printf("XMM[27]: %d\n", _mm256_extract_epi8(reg, 27));
    printf("XMM[26]: %d\n", _mm256_extract_epi8(reg, 26));
    printf("XMM[25]: %d\n", _mm256_extract_epi8(reg, 25));
    printf("XMM[24]: %d\n", _mm256_extract_epi8(reg, 24));
    printf("XMM[23]: %d\n", _mm256_extract_epi8(reg, 23));
    printf("XMM[22]: %d\n", _mm256_extract_epi8(reg, 22));
    printf("XMM[21]: %d\n", _mm256_extract_epi8(reg, 21));
    printf("XMM[20]: %d\n", _mm256_extract_epi8(reg, 20));
    printf("XMM[19]: %d\n", _mm256_extract_epi8(reg, 19));
    printf("XMM[18]: %d\n", _mm256_extract_epi8(reg, 18));
    printf("XMM[17]: %d\n", _mm256_extract_epi8(reg, 17));
    printf("XMM[16]: %d\n", _mm256_extract_epi8(reg, 16));
    printf("XMM[15]: %d\n", _mm256_extract_epi8(reg, 15));
    printf("XMM[14]: %d\n", _mm256_extract_epi8(reg, 14));
    printf("XMM[13]: %d\n", _mm256_extract_epi8(reg, 13));
    printf("XMM[12]: %d\n", _mm256_extract_epi8(reg, 12));
    printf("XMM[11]: %d\n", _mm256_extract_epi8(reg, 11));
    printf("XMM[10]: %d\n", _mm256_extract_epi8(reg, 10));
    printf("XMM[9]: %d\n", _mm256_extract_epi8(reg, 9));
    printf("XMM[8]: %d\n", _mm256_extract_epi8(reg, 8));
    printf("XMM[7]: %d\n", _mm256_extract_epi8(reg, 7));
    printf("XMM[6]: %d\n", _mm256_extract_epi8(reg, 6));
    printf("XMM[5]: %d\n", _mm256_extract_epi8(reg, 5));
    printf("XMM[4]: %d\n", _mm256_extract_epi8(reg, 4));
    printf("XMM[3]: %d\n", _mm256_extract_epi8(reg, 3));
    printf("XMM[2]: %d\n", _mm256_extract_epi8(reg, 2));
    printf("XMM[1]: %d\n", _mm256_extract_epi8(reg, 1));
    printf("XMM[0]: %d\n", _mm256_extract_epi8(reg, 0));
}

void print__m128i(__m128i reg) {
    int8_t* data = (int8_t*)&reg; // Treat the register as an array of 8-bit integers
    for (int i = 0; i < 16; i++) {
        printf("%d ", data[i]); // Print each element
    }
    printf("\n");
}


__m256i convert_epi8_to_epi32(__m256i reg, int numberOfElements) {
    __m256i out;
    __m128i low_128, high_128, shifted;

    switch (numberOfElements) {
        case 1:
            low_128 = _mm256_extracti128_si256(reg, 0); // Extract lower 128 bits
            out = _mm256_cvtepi8_epi32(low_128);
            break;

        case 2:
            low_128 = _mm256_extracti128_si256(reg, 0); // Extract lower 128 bits
            shifted = _mm_srli_si128(low_128, 8); // Shift right by 8 bytes
            out = _mm256_cvtepi8_epi32(shifted);
            break;

        case 3:
            high_128 = _mm256_extracti128_si256(reg, 1); // Extract upper 128 bits
            out = _mm256_cvtepi8_epi32(high_128);
            break;

        case 4:
            high_128 = _mm256_extracti128_si256(reg, 1); // Extract upper 128 bits
            shifted = _mm_srli_si128(high_128, 8); // Shift right by 8 bytes
            out = _mm256_cvtepi8_epi32(shifted);
            break;

        default:
            // Handle invalid input (return zero vector)
            out = _mm256_setzero_si256();
            break;
    }

    return out;
}

__m256i 
convert_and_merge(__m256i reg1, __m256i reg2, __m256i reg3, __m256i reg4) {
    // Step 1: Convert int32_t to int16_t (packing)
    __m256i packed16_1 = _mm256_packs_epi32(reg1, reg2); // Packs first two registers
    __m256i packed16_2 = _mm256_packs_epi32(reg3, reg4); // Packs last two registers

    // Step 2: Convert int16_t to int8_t (final packing)
    __m256i packed8 = _mm256_packs_epi16(packed16_1, packed16_2);

    // Step 3: Reorder bytes correctly
    packed8 = _mm256_permutevar8x32_epi32(packed8, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7));

    return packed8;
}

void updateLevels(__m256i neighbors_infected, int step, int active_infections, int neighbours_toUpdate)
{
                    alignas(32) int indices[8]; // Store extracted indices
                    _mm256_store_si256((__m256i*)indices, neighbors_infected); // Extract all at once

                    for (int i = 0; i < neighbours_toUpdate; i++)
                    {
                        if (indices[i] != -1)
                        {
                            Levels[indices[0]] = step + 1;
                            active_infections++;
                        }
                    }

}

void simulate(int p, int q)
{
    int active_infections = 1;
    int step = 0;

    __m256i probability = _mm256_set1_epi8(p);
    __m256i minus1 = _mm256_set1_epi8(-1);
    __m256i zeros = _mm256_set1_epi8(0);

    print_status(step, active_infections);

    while (active_infections > 0)
    {
        for (int i = 0; i < num_nodes; i++)
        {
            if (Levels[i] == step && Immune[i] == FALSE) // controllo per il nodo 0
            {                                            // Nodo infetto al passo corrente
                int num_neighbors = N[i + 1] - N[i];
                int remainder = num_neighbors % (REGISTER_LENGTH/INTSIZE);
                //printf("num_neighbors: %d, remainder: %d\n", num_neighbors, remainder);


                __m256i neighbors;

                for (int j = 0; j < num_neighbors; j += (REGISTER_LENGTH/INTSIZE))
                {
                    int num_nighbors_visited = 0;
                    neighbors = _mm256_loadu_si256((__m256i *)&L[N[i] + j]);

                    if ((j+(REGISTER_LENGTH/INTSIZE)) > num_neighbors && remainder > 0)
                    { // ultimo vettore
                        __m256i mask = _mm256_set_epi8(
                            0,
                            remainder > 30 ? -1 : 0,
                            remainder > 29 ? -1 : 0,
                            remainder > 28 ? -1 : 0,
                            remainder > 27 ? -1 : 0,
                            remainder > 26 ? -1 : 0,
                            remainder > 25 ? -1 : 0,
                            remainder > 24 ? -1 : 0,
                            remainder > 23 ? -1 : 0,
                            remainder > 22 ? -1 : 0,
                            remainder > 21 ? -1 : 0,
                            remainder > 20 ? -1 : 0,
                            remainder > 19 ? -1 : 0,
                            remainder > 18 ? -1 : 0,
                            remainder > 17 ? -1 : 0,
                            remainder > 16 ? -1 : 0,
                            remainder > 15 ? -1 : 0,
                            remainder > 14 ? -1 : 0,
                            remainder > 13 ? -1 : 0,
                            remainder > 12 ? -1 : 0,
                            remainder > 11 ? -1 : 0,
                            remainder > 10 ? -1 : 0,
                            remainder > 9 ? -1 : 0,
                            remainder > 8 ? -1 : 0,
                            remainder > 7 ? -1 : 0,
                            remainder > 6 ? -1 : 0,
                            remainder > 5 ? -1 : 0,
                            remainder > 4 ? -1 : 0,
                            remainder > 3 ? -1 : 0,
                            remainder > 2 ? -1 : 0,
                            remainder > 1 ? -1 : 0,
                            -1
                        );
                        
                        neighbors = _mm256_and_si256(neighbors, mask);
                    }

                    // printf("Neighbors after mask: \n");
                    // print__mm_register_epi8(neighbors);

                    __m256i n_out1, n_out2, n_out3, n_out4;
                    n_out1 = convert_epi8_to_epi32(neighbors, 1);

                    __m256i Levels_SIMD1, Levels_SIMD2, Levels_SIMD3, Levels_SIMD4;
                    __m256i Immune_SIMD1, Immune_SIMD2, Immune_SIMD3, Immune_SIMD4;
                    
                    Levels_SIMD1 = _mm256_i32gather_epi32(Levels, n_out1, 4);
                    Immune_SIMD1 = _mm256_i32gather_epi32(Immune, n_out1, 4);
                    Levels_SIMD2 = _mm256_setzero_si256();
                    Immune_SIMD2 = _mm256_setzero_si256();
                    Levels_SIMD3 = _mm256_setzero_si256();
                    Immune_SIMD3 = _mm256_setzero_si256();
                    Levels_SIMD4 = _mm256_setzero_si256();
                    Immune_SIMD4 = _mm256_setzero_si256();

                    if(num_neighbors - num_nighbors_visited>8){
                        n_out2 = convert_epi8_to_epi32(neighbors, 2);
                        Levels_SIMD2 = _mm256_i32gather_epi32(Levels, n_out2, 4);
                        Immune_SIMD2 = _mm256_i32gather_epi32(Immune, n_out2, 4);                        
                    }
                    if(num_neighbors - num_nighbors_visited>16){
                        n_out3 = convert_epi8_to_epi32(neighbors, 3);        
                        Levels_SIMD3 = _mm256_i32gather_epi32(Levels, n_out3, 4);
                        Immune_SIMD3 = _mm256_i32gather_epi32(Immune, n_out3, 4);                
                    }
                    if(num_neighbors - num_nighbors_visited>24){
                        n_out4 = convert_epi8_to_epi32(neighbors, 4);         
                        Levels_SIMD4 = _mm256_i32gather_epi32(Levels, n_out4, 4);
                        Immune_SIMD4 = _mm256_i32gather_epi32(Immune, n_out4, 4);                
                    }
                    
                //     __m256i extract_high_mask = _mm256_setr_epi8(
                //         -1, 0, 0, 0,  -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0,
                //         -1, 0, 0, 0,  -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0
                //    );

                    //Levels_SIMD1 = _mm256_and_si256(Levels_SIMD1, extract_high_mask);

                    printf("n_out1: \n");
                    print__mm_register_epi32(n_out1);
                    printf("Levels_SIMD1 after mask: \n");
                    print__mm_register_epi8(Levels_SIMD1);
                    // print__mm_register_epi32(Immune_SIMD1);

                       // Convert and merge
                    __m256i Levels_SIMD_merged = convert_and_merge(Levels_SIMD1, Levels_SIMD2, Levels_SIMD3, Levels_SIMD4);
                    __m256i Immune_SIMD_merged = convert_and_merge(Immune_SIMD1, Immune_SIMD2, Immune_SIMD3, Immune_SIMD4);



                    __m256i mask_susceptible = _mm256_cmpeq_epi8(Levels_SIMD_merged, minus1);
                    __m256i mask_not_immune = _mm256_cmpeq_epi8(Immune_SIMD_merged, zeros);
                    __m256i infection_mask = _mm256_and_si256(mask_susceptible, mask_not_immune);
                    
                    // printf("Infection Mask: \n");
                    // print__mm_register_epi32(infection_mask);

                    // __m256 random = random_m256_01();
                    __m256i random = _mm256_set_epi8(
                        (int8_t)rand()/RAND_MAX*1000,
                        (int8_t)rand()/RAND_MAX*1000,
                        (int8_t)rand()/RAND_MAX*1000,
                        (int8_t)rand()/RAND_MAX*1000,
                        (int8_t)rand()/RAND_MAX*1000,
                        (int8_t)rand()/RAND_MAX*1000,
                        (int8_t)rand()/RAND_MAX*1000,
                        (int8_t)rand()/RAND_MAX*1000,
                        (int8_t)rand()/RAND_MAX*1000,
                        (int8_t)rand()/RAND_MAX*1000,
                        (int8_t)rand()/RAND_MAX*1000,
                        (int8_t)rand()/RAND_MAX*1000,
                        (int8_t)rand()/RAND_MAX*1000,
                        (int8_t)rand()/RAND_MAX*1000,
                        (int8_t)rand()/RAND_MAX*1000,
                        (int8_t)rand()/RAND_MAX*1000,
                        (int8_t)rand()/RAND_MAX*1000,
                        (int8_t)rand()/RAND_MAX*1000,
                        (int8_t)rand()/RAND_MAX*1000,
                        (int8_t)rand()/RAND_MAX*1000,
                        (int8_t)rand()/RAND_MAX*1000,
                        (int8_t)rand()/RAND_MAX*1000,
                        (int8_t)rand()/RAND_MAX*1000,
                        (int8_t)rand()/RAND_MAX*1000,
                        (int8_t)rand()/RAND_MAX*1000,
                        (int8_t)rand()/RAND_MAX*1000,
                        (int8_t)rand()/RAND_MAX*1000,
                        (int8_t)rand()/RAND_MAX*1000,
                        (int8_t)rand()/RAND_MAX*1000,
                        (int8_t)rand()/RAND_MAX*1000,
                        (int8_t)rand()/RAND_MAX*1000,
                        (int8_t)rand()/RAND_MAX*1000
                    );
   
                    
                    __m256i infection_probability = _mm256_cmpgt_epi8(probability, random);
                    // printf("Infection Probability: \n");
                    // print__mm_register_ps(infection_probability);
                    __m256i final_mask = _mm256_and_si256(infection_probability, infection_mask);
                    __m256i neighbors_infected = _mm256_and_si256(neighbors, final_mask);
                    // printf("FinalMask: \n");
                    // print__mm_register_epi32(final_mask);
                    // printf("Neighbors Infected: \n");
                    // print__mm_register_epi32(neighbors_infected);
                    //_mm256_maskstore_epi32(Levels, neighbors_infected, _mm256_set1_epi32(step + 1));

                    // Update Levels and Immune arrays
                    updateLevels(Levels_SIMD_merged, step, active_infections, (num_neighbors-num_nighbors_visited));
                    num_nighbors_visited += 32;
                }
            }

            if (Levels[i] == step && ((double)rand() / RAND_MAX) < q)
            {
                Immune[i] = TRUE; // Nodo recuperato
                active_infections--;
            }
            else if (Levels[i] == step)
            {
                Levels[i] = step + 1; // Nodo può infettare anche al prossimo step
            }
        }
        step++;
        print_status(step, active_infections);
        if(step == 1){
            return;
        }
    }

    // FREE
    _mm_free(N);
    _mm_free(L);
    _mm_free(Levels);
    _mm_free(Immune);
}

int main(int argc, char *argv[])
{
    // Selezionando p=1 e q=1 otteniamo una ricerca in ampiezza
    int p = 1000; // Probabilità di infezione
    int q = 1000; // Probabilità di guarigione

    import_network(argv[1]);

    print_network();
    uint64_t clock_counter_start = __rdtsc();
    simulate(p, q);
    uint64_t clock_counter_end = __rdtsc();
    printf("Elapsed time: %lu\n", clock_counter_end - clock_counter_start);
    return 0;
}