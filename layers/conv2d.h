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

Conv2DLayer *create_conv2d_layer(int stride, int padding, int kernel_size, int input_size, int output_size, Activation activation, float epsilon, float decay_rate);
void destroy_conv2d_layer(Conv2DLayer *layer);
Matrix *forward_conv2d(Conv2DLayer *layer, Matrix *input)
Matrix *backward_conv2d(Conv2DLayer *layer, Matrix *loss_gradient);
void update_conv2d(Conv2DLayer *layer, float learning_rate);