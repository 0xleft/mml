//
// Created by plusleft on 10/30/2023.
//

#include "nn.h"

int main() {
    Network *network;
    network = create_network(1);
    add_layer(network, create_conv2d_layer_l(1, 2, 5, 32, RELU, 0.0f, 0.0f));

    destroy_network(network);

    return 0;
}