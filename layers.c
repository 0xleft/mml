//
// Created by plusleft on 10/29/2023.
//

#include "layers.h"
#include <math.h>

Layer *create_dense_layer_l(int input_size, int output_size, Activation activation, float epsilon, float decay_rate) {
    Layer *layer = malloc(sizeof(Layer));
    layer->type = DENSE;
    layer->layer.dense = create_dense_layer(input_size, output_size, activation, epsilon, decay_rate);
    return layer;
}

Layer *create_conv2d_layer_l(int stride, int padding, int kernel_size, int input_size, Activation activation, float epsilon, float decay_rate) {
    Layer *layer = malloc(sizeof(Layer));
    layer->type = CONV2D;
    layer->layer.conv2d = create_conv2d_layer(stride, padding, kernel_size, input_size, activation, epsilon, decay_rate);
    return layer;
}

Layer *create_maxpool_layer_l(int input_size, int stride, int kernel_size) {
    Layer *layer = malloc(sizeof(Layer));
    layer->type = MAXPOOL;
    layer->layer.maxpool = create_maxpool_layer(input_size, stride, kernel_size);
    return layer;
}