#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>

#define MAX_NODES 1000
#define MAX_EDGES 10000

int N[MAX_NODES + 1]; // Indici dell'inizio dei vicini per ogni nodo
int L[MAX_EDGES]; // Lista di adiacenza compressa
int Levels[MAX_NODES]; // Momento dell'infezione
bool Immune[MAX_NODES]; // Stato di immunità

int num_nodes=10;
int num_edges=3;


bool is_valid_neighbor(int *neighbors, int count, int new_node, int node) {
    if(node==new_node) return false;
    for (int i = 0; i < count; i++) {
        if (neighbors[i] == new_node) return false;
    }
    return true;
}

void initialize_network() {
    srand(time(NULL));
    int edge_count = 0;
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

int main() {
    double p = 0.5; // Probabilità di infezione
    double q = 0.2; // Probabilità di guarigione

    initialize_network();
    print_network();
    simulate(p,q);
    return 0;
}
