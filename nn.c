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

Matrix *backward_layer(Layer *layer, Matrix *input, Matrix *loss_gradient) {
    Matrix *gradient_activation = create_matrix(loss_gradient->rows, loss_gradient->cols);
    for (int i = 0; i < loss_gradient->rows * loss_gradient->cols; i++) {
        gradient_activation->data[i] = derivative(loss_gradient->data[i], layer->activation);
    }

    Matrix *gradient = dot(transpose(loss_gradient), gradient_activation);

    Matrix *gradient_weight = dot(transpose(gradient), transpose(input));
    Matrix *gradient_bias = multiply_s(gradient, -1.0f);
    Matrix *gradient_x = transpose(layer->weights);

    Matrix *gradient_x_result = dot(gradient, gradient_x);
    Matrix *gradient_result = transpose(gradient_x_result);

    // upgrade: the Negative gradient direction
    Matrix *multiply_weight = multiply_s(gradient_weight, 0.1);
    destroy_matrix(layer->weights);
    layer->weights = subtract(layer->weights, multiply_weight);
    Matrix *multiply_bias = multiply_s(gradient_bias, 0.1);
    destroy_matrix(layer->bias);
    layer->bias = subtract(layer->bias, multiply_bias);

    destroy_matrix(multiply_weight);
    destroy_matrix(multiply_bias);
    destroy_matrix(gradient_activation);
    destroy_matrix(gradient);
    destroy_matrix(gradient_weight);
    destroy_matrix(gradient_bias);
    destroy_matrix(gradient_x);
    destroy_matrix(gradient_x_result);
    destroy_matrix(gradient_result);

    return gradient_result;
}

void train(Network *network, Matrix *input, Matrix *expected, int epochs) {
    for (int i = 0; i < epochs; i++) {
        Matrix *output = forward(network, input);

        float loss = calc_loss(output, expected);
        Matrix *loss_gradient;
        loss_gradient = calc_loss_gradient(output, expected);

        destroy_matrix(output);

        for (int i = 0; i < network->layer_count - 1; i++) {
            loss_gradient = backward_layer(network->layers[i], input, loss_gradient);
        }

        destroy_matrix(loss_gradient);

        printf("Loss: %f\n", loss);
    }
}