#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include "lib/cJSON.h"
#include <emmintrin.h>
#include <immintrin.h>
#include <random>
#include <unistd.h>  

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

int8_t** Matrix; // Matrice di adiacenza
int8_t* Levels;  // Momento dell'infezione: istante in cui viene infettato
int8_t* Immune;  // Stato di immunità passa da bool a int

#define TRUE -1
#define FALSE 0
#define AVX_DATALANE 32

int num_nodes;
int num_nodes_32;
int num_edges;

double cpuSecond() {
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

    num_nodes_32 = ((num_nodes + 31) / 32) * 32;

    Matrix = (int8_t **)_mm_malloc(num_nodes * sizeof(int8_t *), AVX_DATALANE);
    for (int i = 0; i < num_nodes; i++)
    {
        Matrix[i] = (int8_t *)_mm_malloc(num_nodes_32 * sizeof(int8_t), AVX_DATALANE);
    }

    Levels = (int8_t *)_mm_malloc(num_nodes_32 * sizeof(int8_t), AVX_DATALANE);
    Immune = (int8_t *)_mm_malloc(num_nodes_32 * sizeof(int8_t), AVX_DATALANE);

    for (int i = 0; i < num_nodes; i++)
    {
        for (int j = 0; j < num_nodes_32; j++)
        {
            Matrix[i][j] = 0;
        }
    }

    for (int i = 0; i < num_nodes; i++)
    {
        int first_neighbor = cJSON_GetArrayItem(json_N, i)->valueint;
        int last_neighbor = cJSON_GetArrayItem(json_N, i + 1)->valueint;

        for (int j = first_neighbor; j < last_neighbor; j++)
        {
            int neighbors_index = cJSON_GetArrayItem(json_L, j)->valueint;
            Matrix[i][neighbors_index] = (int8_t)-1;
        }
    }

    for (int i = 0; i < num_nodes; i++)
    {
        Levels[i] = -1;    // Non infetto
        Immune[i] = FALSE; // Non immune
    }
    for (int i = num_nodes; i < num_nodes_32; i++)
    {
        Levels[i] = -1;
        Immune[i] = FALSE; // Non immune
    }
    Levels[0] = 0; // Nodo inizialmente infetto al tempo 0
}

void print_array(int8_t *array, int length)
{
    for (int i = 0; i < length; i++)
    {
        printf("%d ", array[i]);
    }
    printf("\n");
}

void print_network()
{
    printf("Network:\n");
    for (int i = 0; i < num_nodes; i++)
    {
        for (int j = 0; j < num_nodes; j++)
        {
            printf("%d ", Matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_status(int step, int active_infections, __m256i* Levels_array)
{

    for (int i = 0; i < num_nodes_32 / 32; i++)
    {
        _mm256_store_si256((__m256i *)Levels + i, Levels_array[i]);
    }

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
        }
        printf("\n");
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

int sum_epu8(__m256i v)
{
    // Set zero vector for unpacking
    __m256i zero = _mm256_setzero_si256();

    // Unpack 8-bit elements into 16-bit integers (split into lo and hi)
    __m256i lo16 = _mm256_unpacklo_epi8(v, zero); // Lower 16 bytes expanded to 16-bit
    __m256i hi16 = _mm256_unpackhi_epi8(v, zero); // Upper 16 bytes expanded to 16-bit

    // Sum 16-bit integers
    __m256i sum16 = _mm256_add_epi16(lo16, hi16);

    // Unpack 16-bit elements into 32-bit integers
    __m256i lo32 = _mm256_unpacklo_epi16(sum16, zero);
    __m256i hi32 = _mm256_unpackhi_epi16(sum16, zero);

    // Sum 32-bit integers
    __m256i sum32 = _mm256_add_epi32(lo32, hi32);

    // Extract lower and upper 128-bit halves
    __m128i sum_low = _mm256_extracti128_si256(sum32, 0);
    __m128i sum_high = _mm256_extracti128_si256(sum32, 1);

    // Sum final 128-bit halves
    __m128i total = _mm_add_epi32(sum_low, sum_high);

    // Horizontally sum the four 32-bit values
    total = _mm_hadd_epi32(total, total);
    total = _mm_hadd_epi32(total, total);

    return _mm_cvtsi128_si32(total);
}

void simulate(int8_t p, int8_t q)
{
    __m256i index_mask[32];
    for (int i = 0; i < 32; i++)
    {
        // uint8_t mask[32] = {0}; // Inizializza tutto a 0
        alignas(32) uint8_t mask[32] = {0};

        mask[i] = 0xFF;            // Per Impostare il byte i-esimo a -1 sostituisci con (0xFF)
        index_mask[i] = _mm256_load_si256((__m256i *)mask);
        //print__mm_register_epi8(index_mask[i]);
    }

    int active_infections = 1;
    int step = 0;

    __m256i probability = _mm256_set1_epi8(p);
    __m256i minus1 = _mm256_set1_epi8(-1);
    __m256i zeros = _mm256_set1_epi8(0);
    __m256i ones = _mm256_set1_epi8(1);

    __m256i step_simd = zeros;
    __m256i step_simd_p1 = zeros;

    int remainder = num_nodes % 32;
    __m256i remainder_mask = _mm256_set_epi8(0, remainder > 30 ? -1 : 0, remainder > 29 ? -1 : 0, remainder > 28 ? -1 : 0,
            remainder > 27 ? -1 : 0, remainder > 26 ? -1 : 0, remainder > 25 ? -1 : 0, remainder > 24 ? -1 : 0, remainder > 23 ? -1 : 0,
            remainder > 22 ? -1 : 0, remainder > 21 ? -1 : 0, remainder > 20 ? -1 : 0, remainder > 19 ? -1 : 0, remainder > 18 ? -1 : 0,
            remainder > 17 ? -1 : 0, remainder > 16 ? -1 : 0, remainder > 15 ? -1 : 0, remainder > 14 ? -1 : 0, remainder > 13 ? -1 : 0,
            remainder > 12 ? -1 : 0, remainder > 11 ? -1 : 0, remainder > 10 ? -1 : 0, remainder > 9 ? -1 : 0, remainder > 8 ? -1 : 0,
            remainder > 7 ? -1 : 0, remainder > 6 ? -1 : 0, remainder > 5 ? -1 : 0, remainder > 4 ? -1 : 0, remainder > 3 ? -1 : 0,
            remainder > 2 ? -1 : 0, remainder > 1 ? -1 : 0, -1
    );

    // Array di registri per Levels e Immune
    __m256i *Levels_array = (__m256i *)_mm_malloc(num_nodes_32 / 32 * sizeof(__m256i), AVX_DATALANE);
    //__m256i *Immune_array = (__m256i *)_mm_malloc(num_nodes_32/32 * sizeof(__m256i), AVX_DATALANE);

    //print_status(step, active_infections, Levels_array);


    for (int i = 0; i < num_nodes_32 / 32; i++)
    {
        Levels_array[i] = _mm256_loadu_si256((__m256i *)Levels + i);
        // Immune_array[i] = _mm256_loadu_si256((__m256i *)Immune+i);
    }

    while (active_infections > 0)
    {
        step_simd_p1 = _mm256_add_epi8(step_simd, ones);
        for (int i = 0; i < num_nodes; i++)
        {
            /*
                Levels_array[i/32] AND mask-iesimo 
                mask_step_iesimo = and mask-iesimo e tutti step
            */
            __m256i Levels_simd_i = _mm256_and_si256(Levels_array[i / 32], index_mask[i % 32]);
            __m256i levels_step_i = _mm256_and_si256(index_mask[i % 32], step_simd);
            __m256i sub_step_i = _mm256_sub_epi8(Levels_simd_i, levels_step_i);
            // if(step==1){

            //     printf("Levels_simd[%d]: \n", i/32);
            //     print__mm_register_epi8(Levels_array[i / 32]);
            //     printf("Levels_simd_i [%d]: \n", i);
            //     print__mm_register_epi8(Levels_simd_i);
            //     printf("levels_step_i[%d]: \n", i);
            //     print__mm_register_epi8(levels_step_i);
            //     printf("sub_step_i[%d]: \n", i);
            //     print__mm_register_epi8(sub_step_i);
            // }
            
            if (_mm256_testz_si256(sub_step_i, sub_step_i)) // controllo per il nodo 0 -> Levels[i] == step
            {                      // Nodo infetto al passo corrente
                // printf("Nodo infetto [%d]: \n", i);
                __m256i neighbors;
                for (int j = 0; j < num_nodes; j += 32)
                {
                    neighbors = _mm256_loadu_si256((__m256i *)&Matrix[i][j]);
                    __m256i Levels_SIMD = Levels_array[j / 32];
                    __m256i Immune_SIMD = _mm256_loadu_si256((__m256i *)Immune + (j / 32));

                    if ((j + 32) > num_nodes && remainder > 0)
                    { // ultimo vettore
                        neighbors = _mm256_and_si256(neighbors, remainder_mask);
                        Levels_SIMD = _mm256_and_si256(Levels_SIMD, remainder_mask);
                        Immune_SIMD = _mm256_and_si256(Immune_SIMD, remainder_mask);
                    }

                    __m256i mask_susceptible = _mm256_cmpeq_epi8(Levels_SIMD, minus1);
                    __m256i mask_not_immune = _mm256_cmpeq_epi8(Immune_SIMD, zeros);
                    __m256i infection_mask = _mm256_and_si256(mask_susceptible, mask_not_immune);

                    __m256i random = _mm256_set_epi8((int8_t)rand() % 100, (int8_t)rand() % 100, (int8_t)rand() % 100, 
                        (int8_t)rand() % 100, (int8_t)rand() % 100, (int8_t)rand() % 100, (int8_t)rand() % 100, (int8_t)rand() % 100,
                        (int8_t)rand() % 100, (int8_t)rand() % 100, (int8_t)rand() % 100, (int8_t)rand() % 100, (int8_t)rand() % 100,
                        (int8_t)rand() % 100, (int8_t)rand() % 100, (int8_t)rand() % 100, (int8_t)rand() % 100, (int8_t)rand() % 100,
                        (int8_t)rand() % 100, (int8_t)rand() % 100, (int8_t)rand() % 100, (int8_t)rand() % 100, (int8_t)rand() % 100,
                        (int8_t)rand() % 100, (int8_t)rand() % 100, (int8_t)rand() % 100, (int8_t)rand() % 100, (int8_t)rand() % 100,
                        (int8_t)rand() % 100, (int8_t)rand() % 100, (int8_t)rand() % 100, (int8_t)rand() % 100
                    );

                    __m256i infection_probability = _mm256_cmpgt_epi8(probability, random);
                    __m256i final_mask = _mm256_and_si256(infection_probability, infection_mask);

                    __m256i neighbors_infected = _mm256_and_si256(neighbors, final_mask);
                    __m256i old_Levels = _mm256_andnot_si256(neighbors_infected, Levels_SIMD);
                    __m256i step_simd_AND_neighbors_infected = _mm256_and_si256(step_simd_p1, neighbors_infected);

                    Levels_array[j / 32] = _mm256_add_epi8(step_simd_AND_neighbors_infected, old_Levels);
                    // _mm256_store_si256((__m256i *)Levels+j/32, step_simd_p1); // Extract all at once

                    __m256i mask_256 = _mm256_set1_epi8(254);
                    __m256i num_infected = _mm256_sub_epi8(neighbors_infected, mask_256);
                    num_infected = _mm256_and_si256(num_infected, neighbors_infected);
                    active_infections += sum_epu8(num_infected);
                }

                if ((int8_t)rand() % 100 < q)
                {
                    Immune[i] = TRUE; // Nodo recuperato
                    active_infections--;
                }
                else
                {
                    // Levels[i] = step + 1; // Nodo può infettare anche al prossimo step
                    __m256i simd_1_i = _mm256_and_si256(index_mask[i % 32], ones);
                    Levels_array[i / 32] = _mm256_add_epi8(Levels_array[i / 32], simd_1_i);
                }
            }
        }
        // for (int i = 0; i < num_nodes_32 / 32; i++)
        // {
        //     _mm256_store_si256((__m256i *)Levels + i, Levels_array[i]);
        // }
        step++;
        step_simd = step_simd_p1;
        //print_status(step, active_infections, Levels_array);
        // if (step == 2)
        // {
        //     return;
        // }
    }

    for (int i = 0; i < num_nodes_32 / 32; i++)
    {
        _mm256_store_si256((__m256i *)Levels + i, Levels_array[i]);
    }
}

int main(int argc, char *argv[])
{
    // Selezionando p=1 e q=1 otteniamo una ricerca in ampiezza
    int8_t p = 100; // Probabilità di infezione
    int8_t q = 100; // Probabilità di guarigione

    double start_import = cpuSecond();
    import_network(argv[1]);
    double end_import = cpuSecond();
    printf("Import time: %f seconds\n", end_import - start_import);

    
    printf("Simulating with p=%f, q=%f\n", p, q);
    double start = cpuSecond();
    simulate(p, q);
    double end = cpuSecond();
    printf("Elapsed time: %f seconds\n", end - start);

    // FREE
    for (int i = 0; i < num_nodes; i++)
    {
        _mm_free(Matrix[i]);
    }
    _mm_free(Matrix);
    _mm_free(Immune);
    _mm_free(Levels);
    return 0;
}