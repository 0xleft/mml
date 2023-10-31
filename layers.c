//
// Created by plusleft on 10/29/2023.
//

#include "layers.h"
#include <math.h>

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

Layer *create_conv2d_layer_l(int stride, int padding, int kernel_size, int input_size, Activation activation, float epsilon, float decay_rate) {
    Layer *layer = malloc(sizeof(Layer));
    layer->type = CONV2D;
    layer->layer.conv2d = create_conv2d_layer(stride, padding, kernel_size, input_size, activation, epsilon, decay_rate);
    return layer;
}