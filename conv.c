//
// Created by plusleft on 10/30/2023.
//

#include "nn.h"

int main() {
    Network *network;
    network = create_network(4);
    // we are going for mnist
    add_layer(network, create_conv2d_layer_l(1, 32, 1, 2, 5, 32, RELU, 0.01, 0.01));
    // output is now 32x32x32
    add_layer(network, create_maxpool_layer_l(32, 32, 2, 2));
    // output is now 16x16x32

    Matrix *input = create_matrix(32, 32);

    Matrix *output = forward(network, input);

    destroy_network(network);
    return 0;
}