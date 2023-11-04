#pragma once

//
// Created by plusleft on 11/2/2023.
//

#include "../matrix.h"

struct MaxPoolLayer {
    int stride;
    int kernel_size;
    int input_size;
    int output_size;
    Matrix **input;
    Matrix **output;
    Matrix **mask;
    int input_count;
};

typedef struct MaxPoolLayer MaxPoolLayer;

MaxPoolLayer *create_maxpool_layer(int input_size, int input_count, int stride, int kernel_size);
void destroy_maxpool_layer(MaxPoolLayer *layer);
Matrix **forward_maxpool(MaxPoolLayer *layer, Matrix **input);
Matrix **backward_maxpool(MaxPoolLayer *layer, Matrix **loss_gradient);