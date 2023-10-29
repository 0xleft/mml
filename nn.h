#pragma once

//
// Created by plusleft on 10/25/2023.
//
#include <stddef.h>
#include "matrix.h"
#include "data.h"

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

struct Network {
    int layer_count;
    struct DenseLayer **layers;
};

typedef struct Network Network;

DenseLayer *create_layer(int input_size, int output_size, Activation activation, float epsilon, float decay_rate);
Network *create_network(int layer_count, int *layer_sizes, Activation *activations);
void destroy_layer(DenseLayer *layer);
void destroy_network(Network *network);
Matrix *forward(Network *network, Matrix *input);
Matrix *calc_loss_gradient(Matrix *output, Matrix *expected);
float train_input(Network *network, Matrix *input, Matrix *expected, float learning_rate);
// maybe return final loss?
void train_dataset(Network *network, Dataset *dataset, int epochs, float learning_rate);
Matrix *backward(Network *network, Matrix *expected);
void randomize_network(Network *network);
void initialize_weights_xavier(Network *network);
void initialize_weights_xavier_norm(Network *network);
void print_neural_network(Network *network);