#pragma once

//
// Created by plusleft on 10/31/2023.
//

#include "../matrix.h"
#include "common.h"

struct DenseLayer {
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

typedef struct DenseLayer DenseLayer;

DenseLayer *create_dense_layer(int input_size, int output_size, Activation activation, float epsilon, float decay_rate);
void destroy_dense_layer(DenseLayer *layer);
Matrix *forward_dense(DenseLayer *layer, Matrix *input);
Matrix *backward_dense(DenseLayer *layer, Matrix *loss_gradient);
void update_dense(DenseLayer *layer, float learning_rate);