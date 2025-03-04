#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include "lib/cJSON.h"

#define MAX_NODES 1000
#define MAX_EDGES 10000

int *N; // Indici dell'inizio dei vicini per ogni nodo
int *L; // Lista di adiacenza compressa
int *Levels; // Momento dell'infezione: istante in cui viene infettato
bool *Immune; // Stato di immunità

int num_nodes=10;
int num_edges=3;


bool is_valid_neighbor(int *neighbors, int count, int new_node, int node) {
    if(node==new_node) return false;
    for (int i = 0; i < count; i++) {
        if (neighbors[i] == new_node) return false;
    }
    return true;
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

void import_network(const char *filename) {
    char *json_string = read_file(filename);
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

    for(int i=0;i<num_nodes;i++){
        Levels[i] = -1; // Non infetto
        Immune[i] = false;  // Non immune
    }
    Levels[0] = 0; // Nodo inizialmente infetto al tempo 0
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

    print_status(step, active_infections);

    while (active_infections > 0) {
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
        print_status(step, active_infections);
    }
}


int main(int argc, char *argv[]) {
    double p = 0.5; // Probabilità di infezione
    double q = 0.2; // Probabilità di guarigione

    import_network(argv[1]);

    print_network();
    simulate(p,q);
    return 0;
}
