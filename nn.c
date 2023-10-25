//
// Created by plusleft on 10/25/2023.
//

#include "nn.h"
#include <stdlib.h>
#include <stddef.h>

WeightArray *create_weight_array(int size) {
    struct WeightArray *array = malloc(sizeof(struct WeightArray));
    array->size = size;
    array->data = malloc(sizeof(float) * size);
    return array;
}

void destroy_weight_array(struct WeightArray *array) {
    free(array->data);
    free(array);
}

Layer *create_layer(int input_size, int output_size, enum Activation activation) {
    struct Layer *layer = malloc(sizeof(struct Layer));
    layer->input_size = input_size;
    layer->output_size = output_size;
    layer->weights = create_weight_array(input_size * output_size);
    layer->bias = create_weight_array(output_size);
    layer->activation = activation;
    return layer;
}

void destroy_layer(struct Layer *layer) {
    destroy_weight_array(layer->weights);
    destroy_weight_array(layer->bias);
    free(layer);
}

Network *create_network(int layer_count, int *layer_sizes, enum Activation *activations) {
    struct Network *network = malloc(sizeof(struct Network));
    network->layer_count = layer_count;
    network->layers = malloc(sizeof(struct Layer *) * layer_count);
    for (int i = 0; i < layer_count; ++i) {
        network->layers[i] = create_layer(layer_sizes[i], layer_sizes[i + 1], activations[i]);
    }
    return network;
}

void destroy_network(struct Network *network) {
    for (int i = 0; i < network->layer_count; ++i) {
        destroy_layer(network->layers[i]);
    }
    free(network->layers);
    free(network);
}

void randomize_weights(Network *network) {
    for (int i = 0; i < network->layer_count; ++i) {
        Layer *layer = network->layers[i];
        for (int j = 0; j < layer->weights->size; ++j) {
            layer->weights->data[j] = (float) rand() / RAND_MAX;
        }
        for (int j = 0; j < layer->bias->size; ++j) {
            layer->bias->data[j] = (float) rand() / RAND_MAX;
        }
    }
}

void print_network(Network *network) {
    for (int i = 0; i < network->layer_count; ++i) {
        Layer *layer = network->layers[i];
        printf("Layer %d: %d -> %d\n", i, layer->input_size, layer->output_size);
    }
}