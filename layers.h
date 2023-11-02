#pragma once

//
// Created by plusleft on 10/29/2023.
//

#include "matrix.h"
#include "layers/dense.h"
#include "layers/conv2d.h"
#include "layers/maxpool.h"

enum LayerType {
    DENSE,
    CONV2D,
    MAXPOOL,
};

typedef enum LayerType LayerType;

union LayerUnion {
    DenseLayer *dense;
    Conv2DLayer *conv2d;
    MaxPoolLayer *maxpool;
};

typedef union LayerUnion LayerUnion;

struct Layer {
    LayerType type;
    LayerUnion layer;
};

typedef struct Layer Layer;

// implement forward and backward pass for each layer type
// implement update weights for each layer type

Layer *create_dense_layer_l(int input_size, int output_size, Activation activation, float epsilon, float decay_rate);

// conv

Layer *create_conv2d_layer_l(int stride, int padding, int kernel_size, int input_size, Activation activation, float epsilon, float decay_rate);

// maxpool

Layer *create_maxpool_layer_l(int input_size, int stride, int kernel_size);