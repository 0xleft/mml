#pragma once

//
// Created by plusleft on 10/29/2023.
//

#include "matrix.h"

enum LayerType {
    DENSE,
    CONV,
};

typedef enum LayerType LayerType;

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

struct ConvLayer {
    int stride;
    int padding;
    int kernel_size; // the filter size
    int input_size;
    int output_size;
    int depth;
    Activation activation;
    Matrix3D *input;
    Matrix3D *output;
    Matrix3D *delta;
    float epsilon;
    float decay_rate;
    Matrix3D *weights;
    Matrix *bias;
};

typedef struct ConvLayer ConvLayer;

union LayerUnion {
    DenseLayer *dense;
    ConvLayer *conv;
};

typedef union LayerUnion LayerUnion;

struct Layer {
    LayerType type;
    LayerUnion layer;
};

typedef struct Layer Layer;

// implement forward and backward pass for each layer type
// implement update weights for each layer type

DenseLayer *create_dense_layer(int input_size, int output_size, Activation activation, float epsilon, float decay_rate);
void destroy_dense_layer(DenseLayer *layer);
Matrix *forward_dense(DenseLayer *layer, Matrix *input);
void backward_dense(DenseLayer *layer, Matrix *errors);
void update_dense(DenseLayer *layer, float learning_rate);

// conv