//
// Created by plusleft on 10/31/2023.
//

#include "conv2d.h"
#include "../matrix.h"

Conv2DLayer *create_conv2d_layer(int input_count, int filter_count, int stride, int padding, int kernel_size, int input_size, Activation activation, float epsilon, float decay_rate) {
    Conv2DLayer *layer = malloc(sizeof(Conv2DLayer));
    layer->stride = stride;
    layer->padding = padding;
    layer->kernel_size = kernel_size;
    layer->input_size = input_size;
    layer->activation = activation;
    layer->filter_count = filter_count;
    layer->input_count = input_count;
    layer->epsilon = epsilon;
    layer->decay_rate = decay_rate;
    // input size means nxn input n for input_size
    layer->output_size = (input_size - kernel_size + 2 * padding) / stride + 1;
    for (int i = 0; i < filter_count; i++) {
        layer->weights[i] = create_matrix(kernel_size, kernel_size);
        layer->bias[i] = create_matrix(1, layer->output_size);

        initialize_weights_xavier_norm(layer->input_size, layer->output_size, layer->weights[i]);
        initialize_weights_xavier_norm(1, layer->output_size, layer->bias[i]);
    }

    layer->input = NULL;
    layer->output = NULL;
    layer->delta = NULL;
    return layer;
}

void destroy_conv2d_layer(Conv2DLayer *layer) {
    if (layer == NULL) {
        return;
    }
    for (int i = 0; i < layer->filter_count; i++) {
        destroy_matrix(layer->weights[i]);
        destroy_matrix(layer->bias[i]);
        destroy_matrix(layer->input[i]);
        if (layer->output != NULL)
            destroy_matrix(layer->output[i]);
        if (layer->delta != NULL)
            destroy_matrix(layer->delta[i]);
    }

    free(layer);
    layer = NULL;
}

// basically we are computing the dot product of the kernel passing over the input
// stanford talk:
// https://www.youtube.com/watch?v=bNb2fEVKeEo&
Matrix **forward_conv2d(Conv2DLayer *layer, Matrix **input) {

    Matrix **output = malloc(sizeof(Matrix) * layer->input_count);

    for (int i = 0; i < layer->input_count; i++) {
        for (int j = 0; j < layer->filter_count; j++) {
            Matrix *padded_input = pad(input[i], layer->padding);

            Matrix *flipped_weights = flip(layer->weights[j]);

            Matrix *convolved = convolve(padded_input, flipped_weights, layer->stride, layer->kernel_size, layer->padding, layer->output_size);

            destroy_matrix(flipped_weights);

            for (int i = 0; i < layer->output_size; i++) {
                for (int j = 0; j < layer->output_size; j++) {
                    convolved->data[i][j] += layer->bias[j]->data[0][j];
                }
            }

            Matrix *activated = apply(convolved, layer->activation);

            destroy_matrix(padded_input);
            destroy_matrix(convolved);

            layer->input[j] = copy_matrix(input[i]);
            layer->output[j] = copy_matrix(activated);

            output[i] = activated;
        }
    }

    return output;
}

Matrix **backward_conv2d(Conv2DLayer *layer, Matrix **loss_gradient) {
}

void update_conv2d(Conv2DLayer *layer, float learning_rate) {

}