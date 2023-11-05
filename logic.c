//
// Created by plusleft on 10/29/2023.
//

#include "nn.h"
#include <stdio.h>
#include <stdlib.h>

#define RED "\033[0;31m"
#define RESET "\033[0m"

void and() {
    Network *network;
    network = create_network(4);
    add_layer(network, create_dense_layer_l(2, 100, RELU, 0.0001f, 0.00000001f));
    add_layer(network, create_dense_layer_l(100, 100, RELU, 0.0001f, 0.00000001f));
    add_layer(network, create_dense_layer_l(100, 10, RELU, 0.0001f, 0.00000001f));
    add_layer(network, create_dense_layer_l(10, 1, RELU, 0.0001f, 0.00000001f));

    Matrix *input1 = create_matrix_from_array(1, 2, (float[]) {0, 0});
    Matrix *input2 = create_matrix_from_array(1, 2, (float[]) {0, 1});
    Matrix *input3 = create_matrix_from_array(1, 2, (float[]) {1, 0});
    Matrix *input4 = create_matrix_from_array(1, 2, (float[]) {1, 1});

    Matrix *expected1 = create_matrix_from_array(1, 1, (float[]) {0});
    Matrix *expected2 = create_matrix_from_array(1, 1, (float[]) {0});
    Matrix *expected3 = create_matrix_from_array(1, 1, (float[]) {0});
    Matrix *expected4 = create_matrix_from_array(1, 1, (float[]) {1});

    Dataset *dataset = create_dataset(4);
    add_data(dataset, input1, expected1);
    add_data(dataset, input2, expected2);
    add_data(dataset, input3, expected3);
    add_data(dataset, input4, expected4);

    train_dataset(network, dataset, 3000, 0.1f);

    Matrix *output1 = forward(network, input1);
    Matrix *output2 = forward(network, input2);
    Matrix *output3 = forward(network, input3);
    Matrix *output4 = forward(network, input4);

    printf("0 0: %f\n", output1->data[0][0]);
    printf("0 1: %f\n", output2->data[0][0]);
    printf("1 0: %f\n", output3->data[0][0]);
    printf("1 1: %f\n", output4->data[0][0]);

    destroy_matrix(output1);
    destroy_matrix(output2);
    destroy_matrix(output3);
    destroy_matrix(output4);
    destroy_dataset(dataset);
    destroy_network(network);
}

// TODO fix and explain why tf it no workey :(
void or() {
    Network *network;
    network = create_network(1);
    srand(time(NULL));
    add_layer(network, create_dense_layer_l(2, 1, SIGMOID, 0.0f, 0.0f));

    Matrix *input1 = create_matrix_from_array(1, 2, (float[]) {0, 0});
    Matrix *input2 = create_matrix_from_array(1, 2, (float[]) {0, 1});
    Matrix *input3 = create_matrix_from_array(1, 2, (float[]) {1, 0});
    Matrix *input4 = create_matrix_from_array(1, 2, (float[]) {1, 1});

    Matrix *expected1 = create_matrix_from_array(1, 1, (float[]) {0});
    Matrix *expected2 = create_matrix_from_array(1, 1, (float[]) {1});
    Matrix *expected3 = create_matrix_from_array(1, 1, (float[]) {1});
    Matrix *expected4 = create_matrix_from_array(1, 1, (float[]) {1});

    Dataset *dataset = create_dataset(4);
    add_data(dataset, input1, expected1);
    add_data(dataset, input2, expected2);
    add_data(dataset, input3, expected3);
    add_data(dataset, input4, expected4);

    train_dataset(network, dataset, 10000, 1.0f);

    Matrix *output1 = forward(network, input1);
    Matrix *output2 = forward(network, input2);
    Matrix *output3 = forward(network, input3);
    Matrix *output4 = forward(network, input4);

    printf("0 0: %f\n", output1->data[0][0]);
    printf("0 1: %f\n", output2->data[0][0]);
    printf("1 0: %f\n", output3->data[0][0]);
    printf("1 1: %f\n", output4->data[0][0]);

    destroy_matrix(output1);
    destroy_matrix(output2);
    destroy_matrix(output3);
    destroy_matrix(output4);
    destroy_dataset(dataset);
    destroy_network(network);
}

void xor() {
    Network *network;
    network = create_network(2);
    add_layer(network, create_dense_layer_l(2, 4, SIGMOID, 0.0f, 0.0f));
    add_layer(network, create_dense_layer_l(4, 1, SIGMOID, 0.0f, 0.0f));

    Matrix *input1 = create_matrix_from_array(1, 2, (float[]) {0, 0});
    Matrix *input2 = create_matrix_from_array(1, 2, (float[]) {0, 1});
    Matrix *input3 = create_matrix_from_array(1, 2, (float[]) {1, 0});
    Matrix *input4 = create_matrix_from_array(1, 2, (float[]) {1, 1});

    Matrix *expected1 = create_matrix_from_array(1, 1, (float[]) {0});
    Matrix *expected2 = create_matrix_from_array(1, 1, (float[]) {1});
    Matrix *expected3 = create_matrix_from_array(1, 1, (float[]) {1});
    Matrix *expected4 = create_matrix_from_array(1, 1, (float[]) {0});

    Dataset *dataset = create_dataset(4);
    add_data(dataset, input1, expected1);
    add_data(dataset, input2, expected2);
    add_data(dataset, input3, expected3);
    add_data(dataset, input4, expected4);

    train_dataset(network, dataset, 10000, 0.1f);

    Matrix *output1 = forward(network, input1);
    Matrix *output2 = forward(network, input2);
    Matrix *output3 = forward(network, input3);
    Matrix *output4 = forward(network, input4);

    printf("0 0: %f\n", output1->data[0][0]);
    printf("0 1: %f\n", output2->data[0][0]);
    printf("1 0: %f\n", output3->data[0][0]);
    printf("1 1: %f\n", output4->data[0][0]);

    destroy_matrix(output1);
    destroy_matrix(output2);
    destroy_matrix(output3);
    destroy_matrix(output4);
    destroy_dataset(dataset);
    destroy_network(network);
}

int main() {
    printf("running and\n");
    and();
    printf("running or\n");
    or();
    printf("running xor\n");
    xor();
    return 0;
}