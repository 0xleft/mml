//
// Created by plusleft on 10/29/2023.
//

#include "layers.h"

DenseLayer *create_dense_layer(int input_size, int output_size, Activation activation, float epsilon, float decay_rate) {
    DenseLayer *layer = malloc(sizeof(DenseLayer));
    layer->input_size = input_size;
    layer->output_size = output_size;
    layer->activation = activation;
    layer->weights = create_matrix(input_size, output_size);
    layer->bias = create_matrix(1, output_size);
    layer->input = NULL;
    layer->output = NULL;
    layer->delta = NULL;
    layer->epsilon = epsilon;
    layer->decay_rate = decay_rate;
    return layer;
}

void destroy_dense_layer(DenseLayer *layer) {
    if (layer == NULL) {
        return;
    }
    destroy_matrix(layer->weights);
    destroy_matrix(layer->bias);
    if (layer->input != NULL)
        destroy_matrix(layer->input);
    if (layer->output != NULL)
        destroy_matrix(layer->output);
    if (layer->delta != NULL)
        destroy_matrix(layer->delta);
    free(layer);
    layer = NULL;
}

Matrix *forward_dense(DenseLayer *layer, Matrix *input) {
    Matrix *result = input;
    result = dot(result, layer->weights);
    result = add(result, layer->bias);
    result = apply(result, layer->activation);
    return result;
}

void backward_dense(DenseLayer *layer, Matrix *errors) {
    Matrix *weights_t = transpose(layer->weights);
    Matrix *errors_t = transpose(errors);
    Matrix *input_t = transpose(layer->input);
    Matrix *delta = dot(errors_t, weights_t);
    delta = multiply(delta, input_t);
    destroy_matrix(layer->delta);
    layer->delta = delta;
    destroy_matrix(weights_t);
    destroy_matrix(errors_t);
    destroy_matrix(input_t);
}

void update_dense(DenseLayer *layer, float learning_rate) {
    Matrix *weights_delta = multiply_s(layer->delta, layer->epsilon);
    Matrix *bias_delta = multiply_s(layer->delta, layer->epsilon);
    layer->weights = subtract(layer->weights, weights_delta);
    layer->bias = subtract(layer->bias, bias_delta);
    destroy_matrix(weights_delta);
    destroy_matrix(bias_delta);
}