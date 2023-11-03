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
    layer->weights = malloc(sizeof(Matrix) * filter_count);
    layer->bias = malloc(sizeof(Matrix) * filter_count);
    for (int i = 0; i < filter_count; i++) {
        layer->weights[i] = create_matrix(kernel_size, kernel_size);
        layer->bias[i] = create_matrix(1, layer->output_size);

        initialize_weights_xavier_norm(layer->input_size, layer->output_size, layer->weights[i]);
        initialize_weights_xavier_norm(1, layer->output_size, layer->bias[i]);
    }
    printf("Created weights and bias for %d filters\n", filter_count);

    layer->input = malloc(sizeof(Matrix) * input_count * filter_count);
    layer->output = malloc(sizeof(Matrix) * input_count * filter_count);

    printf("Created input, output, and delta for %d inputs\n", input_count);
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
    }

    free(layer);
    layer = NULL;
}

Matrix *forward_conv2d_single(Conv2DLayer *layer, Matrix *input, int index) {
    Matrix *padded_input = pad(input, layer->padding);
    Matrix *flipped_weights = flip(layer->weights[index]);

    Matrix *convolved = convolve(padded_input, flipped_weights, layer->stride, layer->kernel_size, layer->padding, layer->output_size);

    destroy_matrix(flipped_weights);

    for (int i = 0; i < layer->output_size; i++) {
        for (int j = 0; j < layer->output_size; j++) {
            convolved->data[i][j] += layer->bias[index]->data[0][j];
        }
    }

    Matrix *activated = apply(convolved, layer->activation);

    destroy_matrix(padded_input);
    destroy_matrix(convolved);

    layer->input[index] = copy_matrix(input);
    layer->output[index] = copy_matrix(activated);

    return activated;
}

// basically we are computing the dot product of the kernel passing over the input
// stanford talk:
// https://www.youtube.com/watch?v=bNb2fEVKeEo&
Matrix **forward_conv2d(Conv2DLayer *layer, Matrix **input) {
    for (int i = 0; i < layer->input_count; i++) {
        for (int j = 0; j < layer->filter_count; j++) {
            Matrix *result = forward_conv2d_single(layer, input[i], i * layer->filter_count + j);
            destroy_matrix(result);
        }
    }

    return layer->output;
}

Matrix *backward_conv2d_single(Conv2DLayer *layer, Matrix *loss_gradient, int filter_index) {
    // wi* = wi - a * dL/dwi
    // calculate dL/dwi (matrix of partial derivatives)

    Matrix *padded_input = pad(layer->input[filter_index], layer->padding);

    // for every w in kernel
    for (int i = 0; i < layer->output_size; i++) {
        for (int j = 0; j < layer->output_size; j++) {
            Matrix *a_m = get_slice(padded_input, i * layer->stride, j * layer->stride, layer->kernel_size, layer->kernel_size);

            Matrix *dL_dz = get_slice(loss_gradient, i, j, 1, 1);

            Matrix *dL_dw = multiply_s(a_m, dL_dz->data[0][0]);

            // update weights
            for (int k = 0; k < layer->kernel_size; k++) {
                for (int l = 0; l < layer->kernel_size; l++) {
                    layer->weights[filter_index]->data[k][l] -= 1 * dL_dw->data[k][l];
                }
            }

            // update bias
            for (int k = 0; k < layer->output_size; k++) {
                layer->bias[filter_index]->data[0][k] -= 1 * dL_dz->data[0][k];
            }

            destroy_matrix(dL_dw);
            destroy_matrix(a_m);
            destroy_matrix(dL_dz);
        }
    }

    destroy_matrix(padded_input);

    return copy_matrix(loss_gradient);
}

Matrix **backward_conv2d(Conv2DLayer *layer, Matrix **loss_gradient) {
    Matrix **dout = malloc(sizeof(Matrix) * layer->input_count * layer->filter_count);

    for (int i = 0; i < layer->input_count; i++) {
        for (int j = 0; j < layer->filter_count; j++) {
            Matrix *result = backward_conv2d_single(layer, loss_gradient[i], i * layer->filter_count + j);
            destroy_matrix(result);
        }
        destroy_matrix(loss_gradient[i]);
    }

    return dout;
}