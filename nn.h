#pragma once

//
// Created by plusleft on 10/25/2023.
//
#include <stddef.h>
#include "matrix.h"
#include "data.h"
#include "layers.h"

struct Network {
    int layer_count;
    int max_layer_count;
    Layer **layers;
};

typedef struct Network Network;

Network *create_network(int max_layer_count);
void add_layer(Network *network, Layer *layer);
Matrix *forward(Network *network, Matrix *input);
void destroy_network(Network *network);
// maybe return final loss?
void train_dataset(Network *network, Dataset *dataset, int epochs, float learning_rate);
float train_input(Network *network, Matrix *input, Matrix *expected, float learning_rate);
void initialize_weights_xavier(Network *network);