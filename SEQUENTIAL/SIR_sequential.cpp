#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include "lib/cJSON.h"
#include <emmintrin.h>
#include <immintrin.h>
#include <unistd.h>  


#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#define MAX_NODES 1000
#define MAX_EDGES 10000

int *N; // Indici dell'inizio dei vicini per ogni nodo
int *L; // Lista di adiacenza compressa
int *Levels; // Momento dell'infezione: istante in cui viene infettato
bool *Immune; // Stato di immunità

int num_nodes;
int num_edges;

double cpuSecond() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

char *read_file(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error opening file!\n");
        return NULL;
    }

    char *json_string = NULL;
    size_t size = 0;
    size_t capacity = 128;  // Initial buffer size
    json_string = (char *)malloc(capacity);
    if (!json_string) {
        printf("Memory allocation failed!\n");
        fclose(file);
        return NULL;
    }

    int ch;
    while ((ch = fgetc(file)) != EOF) {
        json_string[size++] = (char)ch;
        // Resize buffer if needed
        if (size >= capacity - 1) {
            capacity *= 2;  // Double the buffer size
            json_string = (char *)realloc(json_string, capacity);
            if (!json_string) {
                printf("Memory reallocation failed!\n");
                fclose(file);
                return NULL;
            }
        }
    }
    json_string[size] = '\0';  // Null-terminate the string
    fclose(file);
    return json_string;
}

void reset_Levels_and_Immune() {
    for (int i = 0; i < num_nodes; i++) {
        Levels[i] = -1; // Inizializza tutti i nodi come non infetti
        Immune[i] = false; // Inizializza tutti i nodi come non immuni
    }
    Levels[0] = 0; // Il nodo 0 è infetto al passo 0
    Immune[0] = false; // Il nodo 0 non è immune
}

void import_network(const char *filename) {
    char filepath[256];
    snprintf(filepath, sizeof(filepath), "../GRAPH_GENERATOR/%s", filename);
    char *json_string = read_file(filepath);
    if (!json_string) {
        exit(1);
    }

    cJSON *root = cJSON_Parse(json_string);
    free(json_string);  // Free memory after parsing
    if (!root) {
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

    for (int i = 0; i < size_N; i++) {
        N[i] = cJSON_GetArrayItem(json_N, i)->valueint;
    }
    for (int i = 0; i < size_L; i++) {
        L[i] = cJSON_GetArrayItem(json_L, i)->valueint;
    }    

    reset_Levels_and_Immune();
}

void print_network(){
    printf("Network:\n");
    for (int i = 0; i < num_nodes; i++) {
        printf("%d: ", i);
        for (int j = N[i]; j < N[i + 1]; j++) {
            printf("%d ", L[j]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_status(int step, int active_infections) {
    printf("Step %d: %d active infections\n", step, active_infections);
    if (active_infections > 0) {
        printf("Infected nodes: ");
        for (int i = 0; i < num_nodes; i++) {
            if (Levels[i] == step) {
                printf("%d ", i);
            }
        }
        printf("\n");
    }    
}

void simulate(double p, double q) {
    int active_infections = 1;
    int step = 0;

    //print_status(step, active_infections);

    while (active_infections > 0) {
        //double start = cpuSecond();
        for (int i = 0; i < num_nodes; i++) {
            if (Levels[i] == step) { // Nodo infetto al passo corrente
                for (int j = N[i]; j < N[i + 1]; j++) {
                    int neighbor = L[j];
                    if (Levels[neighbor] == -1 && !Immune[neighbor] && ((double)rand() / RAND_MAX) < p) {
                        Levels[neighbor] = step + 1; // Infetto al prossimo step
                        active_infections++;
                    }
                }
                if (((double)rand() / RAND_MAX) < q) {
                    Immune[i] = true; // Nodo recuperato
                    active_infections--;
                } else {
                    Levels[i]=step+1; // Nodo può infettare anche al prossimo step
                }
            }
        }
        step++;
        //double end = cpuSecond();
        //printf("Step %d: %fs elapsed time\n", step, end - start);
        //print_status(step, active_infections);
    }
}

int main(int argc, char *argv[]) {
    //Selezionando p=1 e q=1 otteniamo una ricerca in ampiezza
    double p = 1; // Probabilità di infezione
    double q = 1; // Probabilità di guarigione

    double start_import = cpuSecond();
    import_network(argv[1]);
    double end_import = cpuSecond();
    printf("Import time: %f seconds\n", end_import - start_import);

    //print_network();

    printf("#1------------\n");
    printf("Simulating with p=%f, q=%f\n", p, q);
    double start = cpuSecond();
    simulate(p, q);
    double end = cpuSecond();
    printf("Elapsed time: %f seconds\n", end - start);
    printf("-------------\n");

    reset_Levels_and_Immune(); // Reset Levels and Immune arrays for next simulation

    printf("#2\n");
    printf("Simulating with p=%f, q=%f\n", p, q);
    start = cpuSecond();
    simulate(p, q);
    end = cpuSecond();
    printf("Elapsed time: %f seconds\n", end - start);
    printf("-------------\n");

    reset_Levels_and_Immune(); 

    printf("#3\n");
    printf("Simulating with p=%f, q=%f\n", p, 0.8);
    start = cpuSecond();
    simulate(p, 0.8);
    end = cpuSecond();
    printf("Elapsed time: %f seconds\n", end - start);
    printf("-------------\n");

    reset_Levels_and_Immune();

    printf("#4\n");
    printf("Simulating with p=%f, q=%f\n", 0.8, 0.6);
    start = cpuSecond();
    simulate(0.8, 0.6);
    end = cpuSecond();
    printf("Elapsed time: %f seconds\n", end - start);
    printf("-------------\n");

    free(N);
    free(L);
    free(Levels);
    free(Immune);
    return 0;
}
