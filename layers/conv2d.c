//
// Created by plusleft on 10/31/2023.
//

#include "conv2d.h"
#include "../matrix.h"

Conv2DLayer *create_conv2d_layer(int stride, int padding, int kernel_size, int input_size, int output_size, Activation activation, float epsilon, float decay_rate) {
    Conv2DLayer *layer = malloc(sizeof(Conv2DLayer));
    layer->stride = stride;
    layer->padding = padding;
    layer->kernel_size = kernel_size;
    layer->input_size = input_size;
    layer->output_size = output_size;
    layer->activation = activation;
    layer->epsilon = epsilon;
    layer->decay_rate = decay_rate;
    layer->weights = create_matrix(output_size, input_size);
    layer->bias = create_matrix(output_size, 1);
    return layer;
}

void destroy_conv2d_layer(Conv2DLayer *layer) {
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

Matrix *forward_conv2d(Conv2DLayer *layer, Matrix *input) {
    return input;
}

Matrix *backward_conv2d(Conv2DLayer *layer, Matrix *loss_gradient) {
    return loss_gradient;
}

void update_conv2d(Conv2DLayer *layer, float learning_rate) {

}

