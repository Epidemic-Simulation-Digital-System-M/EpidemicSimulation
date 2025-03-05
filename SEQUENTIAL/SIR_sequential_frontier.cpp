#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <stdint.h>
#include <intrin.h>
#include "lib/cJSON.h"

#define MAX_NODES 1000
#define MAX_EDGES 10000

#define SUSCEPTIBLE 0
#define INFECTED 1
#define RECOVERED 2

int *N; // Indici dell'inizio dei vicini per ogni nodo
int *L; // Lista di adiacenza compressa
int *Status; // Stato del nodo

int num_nodes;
int num_edges;

//CODA 

typedef struct Queue {
    int * data;
    int front, rear;
} Queue;

void initQueue(Queue *q) {
    q->data = (int *)malloc(num_nodes * sizeof(int));
    q->front = 0;
    q->rear = 0;
}

int getSize(Queue *q) {
    return (q->rear - q->front + num_nodes) % num_nodes;
}

void enqueue(Queue *q, int value) {
    if (q->rear < num_nodes) {
        q->data[q->rear] = value;
        q->rear=(q->rear+1)%num_nodes;
    }
}

int dequeue(Queue *q) {
    if (getSize(q) > 0) {
        int res = q->data[q->front];
        q->front=(q->front+1)%num_nodes;
        return res;
    }
    return -1;
}

void printQueue(Queue *q) {
    printf("Queue: ");
    for (int i = q->front; i != q->rear; i=(i+1)%num_nodes) {
        printf("%d ", q->data[i]);
    }
    printf(" -- rear: %d, front: %d\n", q->rear, q->front);
}

//READ FILE
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
    Status = (int *)malloc(num_nodes * sizeof(int));

    for (int i = 0; i < size_N; i++) {
        N[i] = cJSON_GetArrayItem(json_N, i)->valueint;
    }
    for (int i = 0; i < size_L; i++) {
        L[i] = cJSON_GetArrayItem(json_L, i)->valueint;
    }    

    for(int i=0;i<num_nodes;i++){
        Status[i]=SUSCEPTIBLE;
    }
    Status[0] = INFECTED; // Nodo inizialmente infetto al tempo 0
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
            if (Status[i] == INFECTED) {
                printf("%d ", i);
            }
        }
        printf("\n");
    }    
}

void simulate(double p, double q) {
    int step = 0;

    Queue frontier;
    initQueue(&frontier);
    enqueue(&frontier, 0);

    int queue_size = getSize(&frontier);
    //printQueue(&frontier);
    //print_status(step, queue_size);

    while (queue_size > 0) {
        for (int i = 0; i < queue_size; i++) {
            int node = dequeue(&frontier);
            for (int j = N[node]; j < N[node + 1]; j++) {
                int neighbor = L[j];
                //printf("Neighbor: %d\n", neighbor);
                if (Status[neighbor] == SUSCEPTIBLE && ((double)rand() / RAND_MAX) < p) {
                    Status[neighbor] = INFECTED; // Infetto al prossimo step
                    enqueue(&frontier, neighbor);
                }
            }
            if (((double)rand() / RAND_MAX) < q) {
                Status[node] = RECOVERED; // Nodo recuperato
            } else {
                // Nodo può infettare anche al prossimo step
                enqueue(&frontier, node);
            }    
        }
        step++;
        queue_size = getSize(&frontier);
        //print_status(step, queue_size);
        //printQueue(&frontier);
    }
}


int main(int argc, char *argv[]) {
    //Selezionando p=1 e q=1 otteniamo una ricerca in ampiezza
    double p = 1; // Probabilità di infezione
    double q = 1; // Probabilità di guarigione

    import_network(argv[1]);

    //print_network();
    uint64_t clock_counter_start = __rdtsc();  
    simulate(p,q);
    uint64_t clock_counter_end = __rdtsc();  
    printf("Elapsed time: %lu\n", clock_counter_end - clock_counter_start);
    return 0;
}
