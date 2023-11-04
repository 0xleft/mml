//
// Created by plusleft on 10/25/2023.
//

#include "nn.h"
#include <stdlib.h>
#include <stddef.h>
#include <math.h>

Network *create_network(int max_layer_count) {
    Network *network = malloc(sizeof(Network));
    network->layer_count = 0;
    network->max_layer_count = max_layer_count;
    network->layers = malloc(sizeof(Layer) * max_layer_count);
    return network;
}

void add_layer(Network *network, Layer *layer) {
    if (network->layer_count == network->max_layer_count) {
        printf("layer count is equal to max layer count\n");
        return;
    }

    network->layers[network->layer_count] = layer;
    network->layer_count++;
}

void destroy_network(Network *network) {
    if (network == NULL) {
        printf("network is null\n");
        return;
    }
    for (int i = 0; i < network->layer_count; i++) {
        Layer *layer = network->layers[i];
        switch (layer->type) {
            case DENSE:
                destroy_dense_layer(layer->layer.dense);
                break;
            case CONV2D:
                destroy_conv2d_layer(layer->layer.conv2d);
                break;
            case MAXPOOL:
                destroy_maxpool_layer(layer->layer.maxpool);
                break;
            case FLATTEN:
                destroy_flatten_layer(layer->layer.flatten);
                break;
        }
    }
    free(network->layers);
    free(network);
    network = NULL;
}

Matrix *forward(Network *network, Matrix *input) {
    // yeah some sunzu shit right here
    Matrix **result = malloc(sizeof(Matrix));
    result[0] = input;
    for (int i = 0; i < network->layer_count; i++) {
        Layer *layer = network->layers[i];
        switch (layer->type) {
            case DENSE:
                result[0] = forward_dense(layer->layer.dense, result[0]);
                break;
            case CONV2D:
                result = forward_conv2d(layer->layer.conv2d, result);
                break;
            case MAXPOOL:
                result = forward_maxpool(layer->layer.maxpool, result);
                break;
            case FLATTEN:
                result[0] = forward_flatten(layer->layer.flatten, result);
                break;
        }
    }

    return result[0];
}

// 2f
Matrix *calc_loss_gradient(Matrix *output, Matrix *expected) {
    Matrix *loss_gradient = subtract(output, expected);
    Matrix *result = multiply_s(loss_gradient, 2.0f);

    destroy_matrix(loss_gradient);

    return result;
}

float calc_loss(Matrix *output, Matrix *expected) {
    Matrix *loss = subtract(output, expected);
    Matrix *squared_loss = power(loss, 2);
    float result = sum(squared_loss);
    destroy_matrix(loss);
    destroy_matrix(squared_loss);
    return result;
}

void backward(Network *network, Matrix *expected) {
    int last_layer_index = network->layer_count - 1;
    Matrix **output = malloc(sizeof(Matrix));
    Matrix **errors = malloc(sizeof(Matrix));
    for (int i = last_layer_index; i >= 0; i--) {
        Layer *layer = network->layers[i];
        switch (layer->type) {
            case DENSE:
                output[0] = layer->layer.dense->output;
                break;
            case CONV2D:
                output = layer->layer.conv2d->output;
                break;
            case MAXPOOL:
                output = layer->layer.maxpool->output;
                break;
            case FLATTEN:
                output[0] = layer->layer.flatten->output;
                break;
        }

        if (i == last_layer_index) {
            errors[0] = calc_loss_gradient(output[0], expected);
        }

        switch (layer->type) {
            case DENSE:
                errors[0] = backward_dense(layer->layer.dense, errors[0]);
                break;
            case CONV2D:
                errors = backward_conv2d(layer->layer.conv2d, errors);
                break;
            case MAXPOOL:
                errors = backward_maxpool(layer->layer.maxpool, errors);
                break;
            case FLATTEN:
                errors = backward_flatten(layer->layer.flatten, errors[0]);
                break;
        }
    }

    // free
    // TODO investigate why double free :(
    //destroy_matrix(errors);
}

void update(Network *network, float learning_rate) {
    for (int i = 0; i < network->layer_count; i++) {
        Layer *layer = network->layers[i];
        switch (layer->type) {
            case DENSE:
                update_dense(layer->layer.dense, learning_rate);
                break;
            case CONV2D:
                break;
            case MAXPOOL:
                // eh
                break;
            case FLATTEN:
                break;
        }
    }
}

float train_input(Network *network, Matrix *input, Matrix *expected, float learning_rate) {
    Matrix *output = forward(network, input);

    backward(network, expected);
    update(network, learning_rate);
    float loss = calc_loss(output, expected);

    destroy_matrix(output);

    return loss;
}

void train_dataset(Network *network, Dataset *dataset, int epochs, float learning_rate) {
    for (int i = 0; i < epochs; i++) {
        float total_loss = 0.0f;

        for (int j = 0; j < dataset->size; j++) {
            Matrix *input = dataset->inputs[j];
            Matrix *expected = dataset->expected[j];
            float loss = train_input(network, input, expected, learning_rate);
            total_loss += loss;
        }

        if (i % 10 == 0)
            printf("epoch %d loss %f\n", i, total_loss);
    }
}