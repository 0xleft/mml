#pragma once

//
// Created by plusleft on 11/2/2023.
//

#include "../matrix.h"

struct FlattenLayer {
    int input_size;
    Matrix *output;
    int input_count;
};

typedef struct FlattenLayer FlattenLayer;

FlattenLayer *create_flatten_layer(int input_size, int input_count);
void destroy_flatten_layer(FlattenLayer *layer);
Matrix *forward_flatten(FlattenLayer *layer, Matrix **input);
Matrix **backward_flatten(FlattenLayer *layer, Matrix *loss_gradient);