//
// Created by plusleft on 11/2/2023.
//

#include "flatten.h"

FlattenLayer *create_flatten_layer(int input_size, int input_count) {
    FlattenLayer *layer = malloc(sizeof(FlattenLayer));
    layer->input_size = input_size;
    layer->input_count = input_count;
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

Matrix *forward_flatten(FlattenLayer *layer, Matrix **input) {
    Matrix *result = create_matrix(1, layer->input_size * layer->input_size * layer->input_count);

    int index = 0;
    for (int i = 0; i < layer->input_count; i++) {
        for (int j = 0; j < layer->input_size; j++) {
            for (int k = 0; k < layer->input_size; k++) {
                result->data[0][index] = input[i]->data[j][k];
                index++;
            }
        }
    }

    return result;
}

Matrix **backward_flatten(FlattenLayer *layer, Matrix *loss_gradient) {
    Matrix **result = malloc(sizeof(Matrix *) * layer->input_count);

    int index = 0;
    for (int i = 0; i < layer->input_count; i++) {
        result[i] = create_matrix(layer->input_size, layer->input_size);
        for (int j = 0; j < layer->input_size; j++) {
            for (int k = 0; k < layer->input_size; k++) {
                result[i]->data[j][k] = loss_gradient->data[0][index];
                index++;
            }
        }
    }

    return result;
}