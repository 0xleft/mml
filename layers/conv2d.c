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
    layer->weights = create_matrix(layer->kernel_size, layer->kernel_size);
    layer->bias = create_matrix(1, layer->output_size);
    layer->input = NULL;
    layer->output = NULL;
    layer->delta = NULL;
    initialize_weights_xavier_norm(layer->input_size, layer->output_size, layer->weights);
    initialize_weights_xavier_norm(1, layer->output_size, layer->bias);
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

    Matrix *flipped_weights = flip(layer->weights);

    Matrix *convolved = convolve(padded_input, flipped_weights, layer->stride, layer->kernel_size, layer->padding, layer->output_size);

    destroy_matrix(flipped_weights);

    for (int i = 0; i < layer->output_size; i++) {
        for (int j = 0; j < layer->output_size; j++) {
            convolved->data[i][j] += layer->bias->data[0][j];
        }
    }

    Matrix *activated = apply(convolved, layer->activation);

    destroy_matrix(padded_input);
    destroy_matrix(convolved);

    layer->input = copy_matrix(input);
    layer->output = copy_matrix(activated);

    return activated;
}

Matrix *backward_conv2d(Conv2DLayer *layer, Matrix *loss_gradient) {
    // The backward computation for a convolution function
    //
    //    Arguments:
    //    dH -- gradient of the cost with respect to output of the conv layer (H), numpy array of shape (n_H, n_W) assuming channels = 1
    //    cache -- cache of values needed for the conv_backward(), output of conv_forward()
    //
    //    Returns:
    //    dX -- gradient of the cost with respect to input of the conv layer (X), numpy array of shape (n_H_prev, n_W_prev) assuming channels = 1
    //    dW -- gradient of the cost with respect to the weights of the conv layer (W), numpy array of shape (f,f) assuming single filter
    //    '''
    //
    //    # Retrieving information from the "cache"
    //    (X, W) = cache
    //
    //    # Retrieving dimensions from X's shape
    //    (n_H_prev, n_W_prev) = X.shape
    //
    //    # Retrieving dimensions from W's shape
    //    (f, f) = W.shape
    //
    //    # Retrieving dimensions from dH's shape
    //    (n_H, n_W) = dH.shape
    //
    //    # Initializing dX, dW with the correct shapes
    //    dX = np.zeros(X.shape)
    //    dW = np.zeros(W.shape)
    //
    //    # Looping over vertical(h) and horizontal(w) axis of the output
    //    for h in range(n_H):
    //        for w in range(n_W):
    //            dX[h:h+f, w:w+f] += W * dH(h,w)
    //            dW += X[h:h+f, w:w+f] * dH(h,w)
    //

    Matrix *dX = create_matrix(layer->input->rows, layer->input->cols);
    Matrix *dW = create_matrix(layer->weights->rows, layer->weights->cols);

    for (int h = 0; h < layer->output_size; h++) {
        for (int w = 0; w < layer->output_size; w++) {
            dX->data[h][w] += layer->weights->data[h][w] * loss_gradient->data[h][w];
            dW->data[h][w] += layer->input->data[h][w] * loss_gradient->data[h][w];
        }
    }

    return dX;
}

void update_conv2d(Conv2DLayer *layer, float learning_rate) {

    if (layer->delta == NULL) {
        return;
    }

    for (int j = 0; j < layer->kernel_size; j++) {
        for (int k = 0; k < layer->kernel_size; k++) {
            float delta = layer->delta->data[0][j];
            float weight = layer->weights->data[j][k];
            float weight_gradient = delta * layer->input->data[0][k];
            float bias_gradient = delta;
            layer->weights->data[j][k] = weight - learning_rate * weight_gradient;
            // layer->bias->data[0][j] = weight - learning_rate * bias_gradient;
        }
    }

    destroy_matrix(layer->delta);
    layer->delta = NULL;
}