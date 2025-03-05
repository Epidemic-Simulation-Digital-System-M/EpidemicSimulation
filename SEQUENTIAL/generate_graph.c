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

int num_nodes;
int num_edges;
int edge_count = 0;

bool is_valid_neighbor(int *neighbors, int count, int new_node, int node) {
    if(node==new_node) return false;
    for (int i = 0; i < count; i++) {
        if (neighbors[i] == new_node) return false;
    }
    return true;
}

void initialize_network() {
    srand(time(NULL));
    N = (int *)malloc((num_nodes + 1) * sizeof(int));
    L = (int *)malloc(num_edges*num_nodes * sizeof(int));
    Levels = (int *)malloc(num_nodes * sizeof(int));
    Immune = (bool *)malloc(num_nodes * sizeof(bool));


    N[0] = 0;
    for (int i = 0; i < num_nodes; i++) {

        int num_neighbors = rand() % num_edges + 1; // Fino a num_edges vicini per nodo
        int *neighbors = (int *)malloc(num_edges * sizeof(int));
        if (neighbors == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }
        int neighbor_count = 0;
        
        for (int j = 0; j < num_neighbors && edge_count < MAX_EDGES; j++) {
            int new_neighbor;
            do {
                new_neighbor = rand() % num_nodes;
            } while (!is_valid_neighbor(neighbors, neighbor_count, new_neighbor, i));
            
            neighbors[neighbor_count++] = new_neighbor;
            L[edge_count++] = new_neighbor;
        }

        N[i + 1] = edge_count;
        // Levels[i] = -1; // Non infetto
        // Immune[i] = false;  // Non immune
    }
    // Levels[0] = 0; // Nodo inizialmente infetto al tempo 0
}

void save_graph(char *filename){
    cJSON *root = cJSON_CreateObject();
    
    cJSON *json_N = cJSON_CreateArray();
    cJSON *json_L = cJSON_CreateArray();

    cJSON *json_numNodes = cJSON_CreateNumber(num_nodes);
    cJSON *json_numEdges = cJSON_CreateNumber(num_edges);

    for(int i=0; i<num_nodes+1; i++){
        cJSON_AddItemToArray(json_N, cJSON_CreateNumber(N[i]));
    }

    for(int i=0; i<edge_count; i++){
        cJSON_AddItemToArray(json_L, cJSON_CreateNumber(L[i]));
    }    

    cJSON_AddItemToObject(root, "num_nodes", json_numNodes);
    cJSON_AddItemToObject(root, "num_edges", json_numEdges);
    cJSON_AddItemToObject(root, "N", json_N);
    cJSON_AddItemToObject(root, "L", json_L);

    char *json_string = cJSON_Print(root);

    // Write JSON to file
    FILE *file = fopen(filename, "w");
    if (file) {
        fputs(json_string, file);
        fclose(file);
        printf("JSON saved successfully!\n");
    } else {
        printf("Error opening file!\n");
    }

    // Clean up
    free(json_string);
    cJSON_Delete(root);

}

int main(int argc, char *argv[]) {

    num_nodes = atoi(argv[2]);
    num_edges = atoi(argv[3]);

    initialize_network();
    save_graph(argv[1]);
    printf("Graph generated successfully!\n");
    return 0;
}