#pragma once

//
// Created by plusleft on 10/31/2023.
//

#include "../matrix.h"

struct Conv2DLayer {
    int stride;
    int padding;
    int kernel_size; // the filter size
    int input_size;
    int output_size;
    Matrix *weights;
    Matrix *bias;
    Activation activation;
    Matrix *input;
    Matrix *output;
    Matrix *delta;
    float epsilon;
    float decay_rate;
};

typedef struct Conv2DLayer Conv2DLayer;