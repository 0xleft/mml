#pragma once

//
// Created by plusleft on 10/29/2023.
//

enum LayerType {
    DENSE,
    CONV,
    MAXPOOL,
};

typedef enum LayerType LayerType;

union Layer {
    DenseLayer *dense;
    ConvLayer *conv;
};

typedef union Layer Layer;

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
    Matrix *weights;
    Matrix *bias;
    Activation activation;
    Matrix3D *input;
    Matrix3D *output;
    Matrix3D *delta;
    float epsilon;
    float decay_rate;
};

typedef struct ConvLayer ConvLayer;

// implement forward and backward pass for each layer type
// implement update weights for each layer type
