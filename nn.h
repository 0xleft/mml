//
// Created by plusleft on 10/25/2023.
//
#include <stddef.h>

#ifndef MML_NN_H
#define MML_NN_H

#endif //MML_NN_H

struct WeightArray {
    int size;
    float *data;
};

typedef struct WeightArray WeightArray;

enum Activation {
    SIGMOID,
    RELU,
    TANH,
    SOFTMAX
};

typedef enum Activation Activation;

struct Layer {
    int input_size;
    int output_size;
    struct WeightArray *weights;
    struct WeightArray *bias;
    enum Activation activation;
};

typedef struct Layer Layer;

struct Network {
    int layer_count;
    struct Layer **layers;
};

typedef struct Network Network;

WeightArray *create_weight_array(int size);
void destroy_weight_array(struct WeightArray *array);

Layer *create_layer(int input_size, int output_size, enum Activation activation);
void destroy_layer(struct Layer *layer);

Network *create_network(int layer_count, int *layer_sizes, enum Activation *activations);
void destroy_network(struct Network *network);
void randomize_weights(Network *network);
void print_network(Network *network);