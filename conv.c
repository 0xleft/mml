//
// Created by plusleft on 10/30/2023.
//

#include "nn.h"

int main() {
    Network *network;
    network = create_network(1);
    Layer *conv_layer = create_conv2d_layer_l(1, 0, 3, 7, RELU, 0.0f, 0.0f);

    add_layer(network, conv_layer);

    Matrix *input = from_image("tests/small.png");
    printf("input size: %d\n", input->rows);

    Matrix *output = forward(network, input);
    printf("output size: %d\n", output->rows);
    print_matrix(output);

    destroy_matrix(input);
    destroy_matrix(output);
    destroy_network(network);
    return 0;
}