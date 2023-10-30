//
// Created by plusleft on 10/29/2023.
//

#include "layers.h"
#include <math.h>

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

void initialize_dense_layer_xavier_norm(DenseLayer *layer) {
    int input_size = layer->input_size;
    int output_size = layer->output_size;

    float lower_bound = -(sqrt(6) / sqrt(input_size + output_size));
    float upper_bound = sqrt(6) / sqrt(input_size + output_size);
    for (int j = 0; j < layer->weights->rows; j++) {
        for (int k = 0; k < layer->weights->cols; k++) {
            layer->weights->data[j][k] = (float) rand() / (float) (RAND_MAX / (upper_bound - lower_bound)) + lower_bound;
        }
    }
}

Layer *create_dense_layer_l(int input_size, int output_size, Activation activation, float epsilon, float decay_rate) {
    Layer *layer = malloc(sizeof(Layer));
    layer->type = DENSE;
    layer->layer.dense = create_dense_layer(input_size, output_size, activation, epsilon, decay_rate);
    initialize_dense_layer_xavier_norm(layer->layer.dense);
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
    print_matrix(layer->weights);
    Matrix *res_dot = dot(input, layer->weights);
    Matrix *res_add = add(res_dot, layer->bias);
    Matrix *result = apply(res_add, layer->activation);
    destroy_matrix(res_dot);
    destroy_matrix(res_add);
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
    float learning_rate_decay = 1.0f - learning_rate * layer->decay_rate;
    Matrix *weights_delta_ad = multiply_s(weights_delta, learning_rate_decay);
    Matrix *bias_delta_ad = multiply_s(bias_delta, learning_rate_decay);
    layer->weights = subtract(layer->weights, weights_delta_ad);
    layer->bias = subtract(layer->bias, weights_delta_ad);
    destroy_matrix(weights_delta);
    destroy_matrix(bias_delta);
    destroy_matrix(weights_delta);
    destroy_matrix(bias_delta);
    destroy_matrix(weights_delta_ad);
    destroy_matrix(bias_delta_ad);
}