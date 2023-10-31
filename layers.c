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
    Matrix *res_dot = dot(input, layer->weights);
    Matrix *res_add = add(res_dot, layer->bias);
    Matrix *result = apply(res_add, layer->activation);
    destroy_matrix(res_dot);
    destroy_matrix(res_add);

    layer->input = copy_matrix(input);
    layer->output = copy_matrix(result);

    return result;
}

Matrix *backward_dense(DenseLayer *layer, Matrix *loss_gradient) {
    Matrix *transfer_derivative = derivative_m(layer->output, layer->activation);
    Matrix *delta = multiply(loss_gradient, transfer_derivative);

    destroy_matrix(loss_gradient);
    destroy_matrix(transfer_derivative);

    layer->delta = delta;

    // downstream gradient
    Matrix *downstream_gradient = dot(delta, transpose(layer->weights));

    return downstream_gradient;
}

void update_dense(DenseLayer *layer, float learning_rate) {
    for (int j = 0; j < layer->output_size; j++) {
        for (int k = 0; k < layer->input_size; k++) {
            float delta = layer->delta->data[0][j];
            float input = layer->input->data[0][k];
            float weight = layer->weights->data[k][j];

            float learning_rate_adjusted = learning_rate / (1 + layer->decay_rate);

            float new_weight = weight - learning_rate_adjusted * delta * input;
            layer->weights->data[k][j] = new_weight;

            // update bias
            float bias = layer->bias->data[0][j];
            float new_bias = bias - learning_rate_adjusted * delta;
            layer->bias->data[0][j] = new_bias;
        }
    }

    destroy_matrix(layer->delta);
    layer->delta = NULL;

    // update decay rate
    layer->decay_rate += layer->epsilon;
}