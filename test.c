//
// Created by plusleft on 10/25/2023.
//

#include "test.h"
#include "nn.h"

int main() {
    Network *network;
    int layer_sizes[] = {2, 3, 2, 1};
    Activation activations[] = {RELU, RELU, SIGMOID};
    network = create_network(3, layer_sizes, activations);

    Matrix *input = create_matrix(1, 2);

    input->data[0] = 0.1;
    input->data[1] = 0.2;

    Matrix *desired_output = create_matrix(1, 1);
    desired_output->data[0] = 0.3;

    train(network, input, desired_output, 1000);

    Matrix *output = forward(network, input);

    printf("Output: %f\n", output->data[0]);

    destroy_matrix(output);
    destroy_matrix(input);
    destroy_matrix(desired_output);
    destroy_network(network);

    return 0;
}