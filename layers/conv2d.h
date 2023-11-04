#pragma once

//
// Created by plusleft on 10/31/2023.
//

#include "../matrix.h"
#include "common.h"

struct Conv2DLayer {
    int stride;
    int padding;
    int kernel_size; // the filter size
    int filter_count;
    int input_size;
    int output_size;
    int input_count;
    Matrix3D *weights;
    Matrix3D *bias;
    Activation activation;
    Matrix3D *input;
    Matrix3D *output;
    Matrix3D *delta;
    float epsilon;
    float decay_rate;
};

typedef struct Conv2DLayer Conv2DLayer;

Conv2DLayer *create_conv2d_layer(int output_size, int input_count, int filter_count, int stride, int padding, int kernel_size, int input_size, Activation activation, float epsilon, float decay_rate);
void destroy_conv2d_layer(Conv2DLayer *layer);
Matrix3D *forward_conv2d(Conv2DLayer *layer, Matrix3D *input);
Matrix3D *backward_conv2d(Conv2DLayer *layer, Matrix3D *loss_gradient);