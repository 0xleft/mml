//
// Created by plusleft on 10/25/2023.
//

#include "test.h"
#include "nn.h"

int main() {
    Layer *layer;
    layer = create_layer(10, 10, SIGMOID);
    destroy_layer(layer);

    Network *network;
    network = create_network(3, (int[]){2, 10, 10, 1}, (Activation[]){SIGMOID, SIGMOID, SIGMOID});
    if (network == NULL) {
        printf("Network is null\n");
        return 1;
    }

    randomize_weights(network);
    print_network(network);

    destroy_network(network);

    return 0;
}