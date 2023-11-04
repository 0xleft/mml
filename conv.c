//
// Created by plusleft on 10/30/2023.
//

#include "nn.h"

int main() {
    Network *network;
    srand(time(NULL));
    network = create_network(3);
    // we are going for mnist
    add_layer(network, create_conv2d_layer_l(5, 1, 1, 1, 1, 3, 7, RELU, 0.01, 0.01));
    // output is now 1x5x5
    add_layer(network, create_flatten_layer_l(5, 1));
    // now its 1x25
    add_layer(network, create_dense_layer_l(25, 2, SIGMOID, 0.1, 0.1));
    // now its 1x1

    Matrix *input = from_image("tests/small.png");
    print_matrix(input);

    Matrix *output = forward(network, input);
    print_matrix(output);

    for (int i = 0; i < 1000; i++) {
        float loss = train_input(network, input, create_matrix_from_array(1, 1, (float[]) {0.5, 0.5}), 0.01);
        if (i % 100 == 0) {
            printf("loss: %f\n", loss);
        }
    }

    destroy_network(network);
    return 0;
}