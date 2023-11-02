//
// Created by plusleft on 10/30/2023.
//

#include "nn.h"

int main() {
    Network *network;
    network = create_network(2);
    add_layer(network, create_maxpool_layer_l(7, 2, 3));
    add_layer(network, create_flatten_layer_l(3));

    Matrix *input = from_image("tests/small.png");
    Matrix *expected = from_image("tests/small_expected.png");
    printf("input %dx%d\n", input->rows, input->cols);
    print_matrix(input);
    printf("expected %dx%d\n", expected->rows, expected->cols);
    print_matrix(expected);

    Matrix *output = forward(network, input);
    print_matrix(output);

    Matrix *a = backward_flatten(network->layers[1]->layer.flatten, output);
    print_matrix(a);

    Matrix *b = backward_maxpool(network->layers[0]->layer.maxpool, a);
    print_matrix(b);

    destroy_matrix(b);
    destroy_matrix(a);
    destroy_matrix(input);
    destroy_matrix(output);
    destroy_network(network);
    return 0;
}