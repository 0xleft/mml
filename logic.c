//
// Created by plusleft on 10/29/2023.
//

#include "nn.h"

void and() {
    Network *network;
    int layer_sizes[] = {2, 1};
    Activation activations[] = {SIGMOID};
    network = create_network(1, layer_sizes, activations);

    srand(66);
    initialize_weights_xavier_norm(network);

    Matrix *input1 = create_matrix_from_array(1, 2, (float[]) {0, 0});
    Matrix *expected1 = create_matrix_from_array(1, 1, (float[]) {0});

    Matrix *input2 = create_matrix_from_array(1, 2, (float[]) {0, 1});
    Matrix *expected2 = create_matrix_from_array(1, 1, (float[]) {0});

    Matrix *input3 = create_matrix_from_array(1, 2, (float[]) {1, 0});
    Matrix *expected3 = create_matrix_from_array(1, 1, (float[]) {0});

    Matrix *input4 = create_matrix_from_array(1, 2, (float[]) {1, 1});
    Matrix *expected4 = create_matrix_from_array(1, 1, (float[]) {1});

    Dataset *dataset = create_dataset(4);
    add_data(dataset, input1, expected1);
    add_data(dataset, input2, expected2);
    add_data(dataset, input3, expected3);
    add_data(dataset, input4, expected4);

    // print_neural_network(network);

    train_dataset(network, dataset, 10000, 1.0f);

    // print_neural_network(network);

    destroy_network(network);
    destroy_dataset(dataset);
}

int main() {
    and();
    return 0;
}