//
// Created by plusleft on 11/2/2023.
//

#include "maxpool.h"

MaxPoolLayer *create_maxpool_layer(int input_size, int stride, int kernel_size) {
    MaxPoolLayer *layer = malloc(sizeof(MaxPoolLayer));
    layer->input_size = input_size;
    layer->output_size = (input_size - kernel_size + 2 * 0) / stride + 1;
    layer->stride = stride;
    layer->kernel_size = kernel_size;
    layer->input = NULL;
    layer->output = NULL;
    return layer;
}

void destroy_maxpool_layer(MaxPoolLayer *layer) {
    if (layer == NULL) {
        return;
    }
    if (layer->input != NULL)
        destroy_matrix(layer->input);
    if (layer->output != NULL)
        destroy_matrix(layer->output);
    free(layer);
    layer = NULL;
}

Matrix *forward_maxpool(MaxPoolLayer *layer, Matrix *input) {
    Matrix *result = create_matrix(layer->output_size, layer->output_size);

    for (int i = 0; i < layer->output_size; i++) {
        for (int j = 0; j < layer->output_size; j++) {
            Matrix *input_slice = get_slice(input, i * layer->stride, j * layer->stride, layer->kernel_size, layer->kernel_size);

            float max_value = max(input_slice);

            result->data[i][j] = max_value;

            destroy_matrix(input_slice);
        }
    }

    return result;
}

Matrix *backward_maxpool(MaxPoolLayer *layer, Matrix *loss_gradient) {
    // todo
    return loss_gradient;
}