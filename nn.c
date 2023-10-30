//
// Created by plusleft on 10/25/2023.
//

#include "nn.h"
#include <stdlib.h>
#include <stddef.h>
#include <math.h>

#define EPSILON 0.000000001f
#define DECAY_RATE 0.00001f

// using the xavier initialization weight = U [-(1/sqrt(n)), 1/sqrt(n)] where n is the number of inputs
void initialize_weights_xavier(Network *network) {
    // TODO
}

Network *create_network(int max_layer_count) {
    Network *network = malloc(sizeof(Network));
    network->layer_count = 0;
    network->max_layer_count = max_layer_count;
    network->layers = malloc(sizeof(Layer) * max_layer_count);
    return network;
}

void add_layer(Network *network, Layer *layer) {
    if (network->layer_count == network->max_layer_count) {
        printf("layer count is equal to max layer count\n");
        return;
    }

    network->layers[network->layer_count] = layer;
    network->layer_count++;
}

void destroy_network(Network *network) {
    if (network == NULL) {
        printf("network is null\n");
        return;
    }
    for (int i = 0; i < network->layer_count; i++) {
        // TODO
    }
    free(network->layers);
    free(network);
    network = NULL;
}

Matrix *forward(Network *network, Matrix *input) {
    Matrix *result = input;
    for (int i = 0; i < network->layer_count; i++) {
        // TODO
    }
    return result;
}

void backward(Network *network, Matrix *expected) {

}

void update(Network *network, float learning_rate) {

}

// 2f
Matrix *calc_loss_gradient(Matrix *output, Matrix *expected) {
    Matrix *loss_gradient = subtract(output, expected);
    Matrix *result = multiply_s(loss_gradient, 2.0f);

    destroy_matrix(loss_gradient);

    return result;
}

float calc_loss(Matrix *output, Matrix *expected) {
    Matrix *loss = subtract(output, expected);
    Matrix *squared_loss = power(loss, 2);
    float result = sum(squared_loss);
    destroy_matrix(loss);
    destroy_matrix(squared_loss);
    return result;
}

float train_input(Network *network, Matrix *input, Matrix *expected, float learning_rate) {
    Matrix *output = forward(network, input);
    backward(network, expected);
    update(network, learning_rate);
    float loss = calc_loss(output, expected);

    destroy_matrix(output);

    return loss;
}

void train_dataset(Network *network, Dataset *dataset, int epochs, float learning_rate) {
    for (int i = 0; i < epochs; i++) {
        float total_loss = 0.0f;

        for (int j = 0; j < dataset->size; j++) {
            Matrix *input = dataset->inputs[j];
            Matrix *expected = dataset->expected[j];
            float loss = train_input(network, input, expected, learning_rate);
            total_loss += loss;
        }

        if (i % 1000 == 0)
            printf("epoch %d loss %f\n", i, total_loss);
    }
}