//
// Created by plusleft on 11/2/2023.
//

#include "flatten.h"

FlattenLayer *create_flatten_layer(int input_size) {
    FlattenLayer *layer = malloc(sizeof(FlattenLayer));
    layer->input_size = input_size;
    layer->output = NULL;
    return layer;
}

void destroy_flatten_layer(FlattenLayer *layer) {
    if (layer == NULL) {
        return;
    }
    if (layer->output != NULL) {
        destroy_matrix(layer->output);
        layer->output = NULL;
    }
    free(layer);
    layer = NULL;
}

Matrix *forward_flatten(FlattenLayer *layer, Matrix *input) {
    Matrix *result = create_matrix(1, input->cols*input->rows);

    for (int i = 0; i < input->rows; i++) {
        for (int j = 0; j < input->cols; j++) {
            result->data[0][i * input->rows + j] = input->data[i][j];
        }
    }

    layer->output = result;

    return result;
}

Matrix *backward_flatten(FlattenLayer *layer, Matrix *loss_gradient) {
    Matrix *result = create_matrix(layer->input_size, layer->input_size);

    for (int i = 0; i < layer->input_size; i++) {
        for (int j = 0; j < layer->input_size; j++) {
            result->data[i][j] = loss_gradient->data[0][i * layer->input_size + j];
        }
    }

    return result;
}