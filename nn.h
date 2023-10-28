#pragma once

//
// Created by plusleft on 10/25/2023.
//
#include <stddef.h>
#include "matrix.h"

struct Layer {
    int input_size;
    int output_size;
    Matrix *weights;
    Matrix *bias;
    Activation activation;
    Matrix *input;
    Matrix *output;
    Matrix *delta;
};

typedef struct Layer Layer;

struct Network {
    int layer_count;
    struct Layer **layers;
};

typedef struct Network Network;

Layer *create_layer(int input_size, int output_size, Activation activation);
Network *create_network(int layer_count, int *layer_sizes, Activation *activations);
void destroy_layer(Layer *layer);
void destroy_network(Network *network);
Matrix *forward(Network *network, Matrix *input);
Matrix *calc_loss_gradient(Matrix *output, Matrix *expected);
void train(Network *network, Matrix *input, Matrix *expected, int epochs, float learning_rate);
Matrix *backward(Network *network, Matrix *expected);
void randomize_network(Network *network);
void initialize_weights_xavier(Network *network);
void initialize_weights_xavier_norm(Network *network);