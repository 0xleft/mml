//
// Created by plusleft on 10/31/2023.
//

#include "conv2d.h"
#include "../matrix.h"

Conv2DLayer *create_conv2d_layer(int output_size, int input_count, int filter_count, int stride, int padding, int kernel_size, int input_size, Activation activation, float epsilon, float decay_rate) {
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
    layer->output_size = output_size;
    layer->weights = create_matrix_3d(filter_count);
    layer->bias = create_matrix_3d(filter_count);
    for (int i = 0; i < filter_count; i++) {
        layer->weights->data[i] = create_matrix(kernel_size, kernel_size);
        layer->bias->data[i] = create_matrix(1, layer->output_size);

        initialize_weights_xavier_norm(layer->input_size, layer->output_size, layer->weights->data[i]);
        initialize_weights_xavier_norm(1, layer->output_size, layer->bias->data[i]);
    }
    printf("Created weights and bias for %d filters\n", filter_count);

    layer->input = create_matrix_3d(input_count * filter_count);
    layer->output = create_matrix_3d(input_count * filter_count);

    printf("Created input, output, and delta for %d inputs\n", input_count * filter_count);
    return layer;
}

void destroy_conv2d_layer(Conv2DLayer *layer) {
    if (layer == NULL) {
        return;
    }
    destroy_matrix_3d(layer->weights);
    destroy_matrix_3d(layer->bias);
    destroy_matrix_3d(layer->input);
    destroy_matrix_3d(layer->output);
    free(layer);
    layer = NULL;
}

void forward_conv2d_single(Conv2DLayer *layer, Matrix *input, int index) {
    Matrix *padded_input = pad(input, layer->padding);
    Matrix *flipped_weights = flip(layer->weights->data[index]);

    Matrix *convolved = convolve(padded_input, flipped_weights, layer->stride, layer->kernel_size, layer->padding, layer->output_size);

    destroy_matrix(flipped_weights);

    for (int i = 0; i < layer->output_size; i++) {
        for (int j = 0; j < layer->output_size; j++) {
            convolved->data[i][j] += layer->bias->data[index]->data[0][j];
        }
    }

    Matrix *activated = apply(convolved, layer->activation);

    destroy_matrix(padded_input);
    destroy_matrix(convolved);

    layer->input->data[index] = copy_matrix(input);
    layer->output->data[index] = copy_matrix(activated);
}

// basically we are computing the dot product of the kernel passing over the input
// stanford talk:
// https://www.youtube.com/watch?v=bNb2fEVKeEo&
Matrix3D *forward_conv2d(Conv2DLayer *layer, Matrix3D *input) {
    //destroy_matrix_3d(layer->output);
    //destroy_matrix_3d(layer->input);
    //destroy_matrix_3d(layer->delta);

    for (int i = 0; i < layer->input_count; i++) {
        for (int j = 0; j < layer->filter_count; j++) {
            forward_conv2d_single(layer, input->data[i], i * layer->filter_count + j);
        }
    }

    return layer->output;
}

Matrix *backward_conv2d_single(Conv2DLayer *layer, Matrix *loss_gradient, int filter_index) {




    Matrix *dw = create_matrix(layer->kernel_size, layer->kernel_size);
    for (int i = 0; i < layer->kernel_size; i++) {
        for (int j = 0; j < layer->kernel_size; j++) {
            for (int x = 0; x < layer->output_size; x++) {
                for (int y = 0; y < layer->output_size; y++) {
                    // Compute the gradient for filter weight dw[i][j]
                    dw->data[i][j] += loss_gradient->data[x][y] * layer->input->data[filter_index]->data[x + i][y + j];
                }
            }
        }
    }

    /* Compute bias gradient db */
    Matrix *db = create_matrix(1, layer->output_size);
    for (int x = 0; x < layer->output_size; x++) {
        for (int y = 0; y < layer->output_size; y++) {
            // Compute the gradient for the bias term db[x]
            db->data[0][x] += loss_gradient->data[x][y];
        }
    }

    /* Compute loss gradient for this layer (dout) */
    Matrix *dout = create_matrix(layer->input_size, layer->input_size);
    for (int i = 0; i < layer->kernel_size; i++) {
        for (int j = 0; j < layer->kernel_size; j++) {
            for (int x = 0; x < layer->output_size; x++) {
                for (int y = 0; y < layer->output_size; y++) {
                    // Compute the gradient for the input of the layer
                    dout->data[x + i][y + j] += loss_gradient->data[x][y] * layer->weights->data[filter_index]->data[i][j];
                }
            }
        }
    }

    /*
        * Update weights and bias
     */

    for (int i = 0; i < layer->kernel_size; i++) {
        for (int j = 0; j < layer->kernel_size; j++) {
            for (int x = 0; x < layer->output_size; x++) {
                for (int y = 0; y < layer->output_size; y++) {
                    // Compute the gradient for filter weight dw[i][j]
                    layer->weights->data[filter_index]->data[i][j] = 0.00001 * dw->data[i][j];
                }
            }
        }
    }

    for (int x = 0; x < layer->output_size; x++) {
        for (int y = 0; y < layer->output_size; y++) {
            // Compute the gradient for the bias term db[x]
            layer->bias->data[filter_index]->data[0][x] = 0.00001 * db->data[0][x];
        }
    }

    /* Clean up and return gradients */
    destroy_matrix(db);
    destroy_matrix(dw);

    return dout;
}

Matrix3D *backward_conv2d(Conv2DLayer *layer, Matrix3D *loss_gradient) {
    Matrix3D *dout = create_matrix_3d(layer->input_count * layer->filter_count);

    for (int i = 0; i < layer->input_count; i++) {
        for (int j = 0; j < layer->filter_count; j++) {
            Matrix *result = backward_conv2d_single(layer, loss_gradient->data[i], i * layer->filter_count + j);
            dout->data[i * layer->filter_count + j] = result;
        }
    }

    return dout;
}