#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include "lib/cJSON.h"


int *N; // Indici dell'inizio dei vicini per ogni nodo
int *L; // Lista di adiacenza compressa
int *Levels; // Momento dell'infezione: istante in cui viene infettato
bool *Immune; // Stato di immunità

int num_nodes;
int num_edges;
int edge_count = 0;

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
    char filepath[256];
    snprintf(filepath, sizeof(filepath), "../GRAPH_GENERATOR/%s", filename);
    char *json_string = read_file(filepath);    if (!json_string) {
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
    edge_count = size_L;

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

// Funzione per stampare gli elementi dell'array
void stampaArray(int array[], int dimensione) {
    printf("Array: ");
    for (int i = 0; i < dimensione; i++) {
        printf("%d ", array[i]);
    }
    printf("\n");
}

// Funzione per eseguire il partitioning e trovare il pivot
int partition(int array[], int basso, int alto) {
    int pivot = array[alto];
    int i = (basso - 1);

    for (int j = basso; j <= alto - 1; j++) {
        if (array[j] < pivot) {
            i++;
            int temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }
    int temp = array[i + 1];
    array[i + 1] = array[alto];
    array[alto] = temp;
    return (i + 1);
}

// Funzione principale per eseguire l'ordinamento quick sort
void quickSort(int array[], int basso, int alto) {
    if (basso < alto) {
        int pivot = partition(array, basso, alto);

        quickSort(array, basso, pivot - 1);
        quickSort(array, pivot + 1, alto);
    }
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

char *find_newFileName(char *fileName) {
    // Make a copy of the input string (so we don't modify the original)
    char *nameWithoutExt = (char *)malloc(strlen(fileName) + 10);
    if (nameWithoutExt == NULL) {
        return NULL; // Memory allocation failed
    }

    strcpy(nameWithoutExt, fileName);

    // Find the ".json" extension
    char *ext = strstr(nameWithoutExt, ".json");
    if (ext != NULL) {
        *ext = '\0'; // Truncate the string
    }
    strcat(nameWithoutExt, "_sorted.json");

    return nameWithoutExt;
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

int main(int argc, char *argv[]) {

    import_network(argv[1]);
    printf("Network imported successfully!\n");
    //print_network();
    
    for(int i=0;i<num_nodes;i++){
        int start_index = N[i];
        int end_index = N[i+1];

        int* neighbors = (int*)malloc((end_index-start_index)*sizeof(int));
        for (int j=start_index;j<end_index;j++){
            neighbors[j-start_index] = L[j];
        }

        quickSort(neighbors, 0, end_index-start_index-1);

        for(int j=start_index;j<end_index;j++){
            L[j] = neighbors[j-start_index];
        }
    }

    printf("Neighbours sorted\n");
    char* newFileName = find_newFileName(argv[1]);
    save_graph(newFileName);
    
    printf("Graph sorted successfully!\n");
    free(newFileName);
    free(N);
    free(L);
    free(Levels);
    free(Immune);
    return 0;
}