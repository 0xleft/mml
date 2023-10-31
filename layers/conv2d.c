//
// Created by plusleft on 10/31/2023.
//

#include "conv2d.h"
#include "../matrix.h"

Conv2DLayer *create_conv2d_layer(int stride, int padding, int kernel_size, int input_size, Activation activation, float epsilon, float decay_rate) {
    Conv2DLayer *layer = malloc(sizeof(Conv2DLayer));
    layer->stride = stride;
    layer->padding = padding;
    layer->kernel_size = kernel_size;
    layer->input_size = input_size;
    layer->activation = activation;
    layer->epsilon = epsilon;
    layer->decay_rate = decay_rate;
    // input size means nxn input n for input_size
    layer->output_size = (input_size - kernel_size + 2 * padding) / stride + 1;
    layer->weights = create_matrix(kernel_size * kernel_size, layer->output_size);
    layer->bias = create_matrix(1, layer->output_size);
    layer->input = NULL;
    layer->output = NULL;
    layer->delta = NULL;
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

// basically we are computing the dot product of the kernel passing over the input
// stanford talk:
// https://www.youtube.com/watch?v=bNb2fEVKeEo&
Matrix *forward_conv2d(Conv2DLayer *layer, Matrix *input) {
    Matrix *padded_input = pad(input, layer->padding);
    Matrix *result = create_matrix(layer->output_size, layer->output_size);

    for (int i = 0; i < layer->output_size; i++) {
        for (int j = 0; j < layer->output_size; j++) {
            Matrix *input_slice = get_slice(padded_input, i * layer->stride, j * layer->stride, layer->kernel_size, layer->kernel_size);

            // dot product
            float sum = 0;
            for (int k = 0; k < layer->kernel_size; k++) {
                for (int l = 0; l < layer->kernel_size; l++) {
                    sum += input_slice->data[k][l] * layer->weights->data[k][l];
                }
            }

            result->data[i][j] = sum + layer->bias->data[0][0];

            destroy_matrix(input_slice);

            // activation
            result->data[i][j] = activate(result->data[i][j], layer->activation);
        }
    }

    destroy_matrix(padded_input);

    layer->input = copy_matrix(input);
    layer->output = copy_matrix(result);

    return result;
}

Matrix *backward_conv2d(Conv2DLayer *layer, Matrix *loss_gradient) {
    return loss_gradient;
}

void update_conv2d(Conv2DLayer *layer, float learning_rate) {

}