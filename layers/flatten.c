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
    return loss_gradient;
}