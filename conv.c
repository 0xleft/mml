//
// Created by plusleft on 10/30/2023.
//

#include "nn.h"

int main() {
    Network *network;
    network = create_network(1);
    Layer *conv_layer = create_conv2d_layer_l(1, 2, 5, 32, RELU, 0.0f, 0.0f);

    printf("outut size: %d\n", conv_layer->layer.conv2d->output_size);

    add_layer(network, conv_layer);

    Matrix *input = from_image("tests/cross.png");
    printf("input size: %d\n", input->rows);


    Matrix *output = forward(network, input);
    print_matrix(output);

    destroy_network(network);

    return 0;
}