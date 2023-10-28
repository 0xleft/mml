//
// Created by plusleft on 10/25/2023.
//

#include "nn.h"
#include <stdlib.h>
#include <stddef.h>

Layer *create_layer(int input_size, int output_size, Activation activation) {
    Layer *layer = malloc(sizeof(Layer));
    layer->input_size = input_size;
    layer->output_size = output_size;
    layer->activation = activation;
    layer->weights = create_matrix(input_size, output_size);
    layer->bias = create_matrix(1, output_size);
    layer->input = NULL;
    layer->output = NULL;
    return layer;
}

Network *create_network(int layer_count, int *layer_sizes, Activation *activations) {
    Network *network = malloc(sizeof(Network));
    network->layer_count = layer_count;
    network->layers = malloc(sizeof(Layer *) * layer_count);
    for (int i = 0; i < layer_count; i++) {
        network->layers[i] = create_layer(layer_sizes[i], layer_sizes[i + 1], activations[i]);
    }
    return network;
}

void destroy_layer(Layer *layer) {
    destroy_matrix(layer->weights);
    destroy_matrix(layer->bias);
    free(layer);
}

void destroy_network(Network *network) {
    for (int i = 0; i < network->layer_count; i++) {
        destroy_layer(network->layers[i]);
    }
    free(network->layers);
    free(network);
}

Matrix *forward(Network *network, Matrix *input) {
    Matrix *result = input;
    for (int i = 0; i < network->layer_count; i++) {
        Layer *layer = network->layers[i];
        result = dot(result, layer->weights);
        result = add(result, layer->bias);
        result = apply(result, layer->activation);

        layer->input = input;
        layer->output = result;
    }
    return result;
}

Matrix *calc_loss_gradient(Matrix *output, Matrix *expected) {
    Matrix *loss_gradient = subtract(output, expected);
    Matrix *result = multiply_s(loss_gradient, 2.0f);

    destroy_matrix(loss_gradient);

    return result;
}

float calc_loss(Matrix *output, Matrix *expected) {
    Matrix *sub = subtract(output, expected);
    Matrix *power_result = power(sub, 2.0f);
    float sum_result = sum(power_result);

    destroy_matrix(sub);
    destroy_matrix(power_result);

    return sum_result;
}

Matrix *backward_layer(Layer *layer, Matrix *loss) {
    return NULL;
}

void train(Network *network, Matrix *input, Matrix *expected, int epochs) {
    for (int i = 0; i < epochs; i++) {
        Matrix *output = forward(network, input);
        Matrix *loss = calc_loss_gradient(output, expected);

        print_matrix(loss);

        destroy_matrix(output);
        destroy_matrix(loss);
    }
}