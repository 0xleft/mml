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
    add_layer(network, create_flatten_layer_l(16, 32));
    // now its 1x(32 * 16 * 16)
    add_layer(network, create_dense_layer_l(8192, 10, SOFTMAX, 0.0, 0.0));
    // now its 1x10

    Matrix *input = from_image("tests/smile.png");
    print_matrix(input);

    Matrix *output = forward(network, input);
    print_matrix(output);

    for (int i = 0; i < 1000; i++) {
        float loss = train_input(network, input, create_matrix_from_array(1, 10, (float[]) {1, 0, 0, 0, 0, 0, 0, 0, 0, 0}), 0.01);
        if (i % 100 == 0) {
            printf("loss: %f\n", loss);
        }
    }

    destroy_network(network);
    return 0;
}