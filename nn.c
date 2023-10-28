//
// Created by plusleft on 10/25/2023.
//

#include "nn.h"
#include <stdlib.h>
#include <stddef.h>

Layer *create_layer(int input_size, int output_size, Activation activation) {
    Layer *layer = malloc(sizeof(Layer));
    layer->input_size = input_size;
    layer->output_size = output_size;
    layer->activation = activation;
    layer->weights = create_matrix(input_size, output_size);
    layer->bias = create_matrix(1, output_size);
    layer->input = NULL;
    layer->output = NULL;
    layer->delta = NULL;
    return layer;
}

// using the xavier initialization weight = U [-(1/sqrt(n)), 1/sqrt(n)] where n is the number of inputs
void initialize_weights_xavier(Network *network) {
    for (int i = 0; i < network->layer_count; i++) {
        int input_size = network->layers[i]->input_size;
        Layer *layer = network->layers[i];

        float weight = 1.0f / sqrtf(input_size);
        for (int j = 0; j < layer->weights->rows; j++) {
            for (int k = 0; k < layer->weights->cols; k++) {
                layer->weights->data[j][k] = weight;
            }
        }
    }
}

void randomize_network(Network *network) {
    for (int i = 0; i < network->layer_count; i++) {
        Layer *layer = network->layers[i];
        for (int j = 0; j < layer->weights->rows; j++) {
            for (int k = 0; k < layer->weights->cols; k++) {
                layer->weights->data[j][k] = (float) rand() / (float) (RAND_MAX / 2) - 1;
            }
        }
        for (int j = 0; j < layer->bias->rows; j++) {
            for (int k = 0; k < layer->bias->cols; k++) {
                layer->bias->data[j][k] = (float) rand() / (float) (RAND_MAX / 2) - 1;
            }
        }
    }
}

Network *create_network(int layer_count, int *layer_sizes, Activation *activations) {
    Network *network = malloc(sizeof(Network));
    network->layer_count = layer_count;
    network->layers = malloc(sizeof(Layer *) * layer_count);
    for (int i = 0; i < layer_count; i++) {
        network->layers[i] = create_layer(layer_sizes[i], layer_sizes[i + 1], activations[i]);
    }
    return network;
}

void destroy_layer(Layer *layer) {
    if (layer == NULL) {
        printf("layer is null\n");
        return;
    }
    if (layer->weights == NULL) {
        printf("weights is null\n");
        return;
    }
    if (layer->bias == NULL) {
        printf("bias is null\n");
        return;
    }
    destroy_matrix(layer->weights);
    destroy_matrix(layer->bias);
    // maybe free input, output and delta?
    free(layer);
}

void destroy_network(Network *network) {
    for (int i = 0; i < network->layer_count; i++) {
        destroy_layer(network->layers[i]);
    }
    free(network->layers);
    free(network);
}

Matrix *forward(Network *network, Matrix *input) {
    Matrix *result = input;
    for (int i = 0; i < network->layer_count; i++) {
        Layer *layer = network->layers[i];
        result = dot(result, layer->weights);
        result = add(result, layer->bias);
        result = apply(result, layer->activation);

        // duplicate matrix
        layer->input = input;
        layer->output = result;
    }
    return result;
}

// 2f
Matrix *calc_loss_gradient(Matrix *output, Matrix *expected) {
    Matrix *loss_gradient = subtract(output, expected);
    Matrix *result = multiply_s(loss_gradient, 2.0f);

    destroy_matrix(loss_gradient);

    return result;
}

Matrix *backward(Network *network, Matrix *expected) {
    // yep https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
    int last_layer_index = network->layer_count - 1;
    for (int i = last_layer_index; i >= 0; i--) {
        Layer *layer = network->layers[i];

        Matrix *errors = NULL;

        if (i != last_layer_index) {
            errors = create_matrix(1, layer->output_size);
            for (int j = 0; j < layer->output_size; j++) {
                float error = 0.0f;
                for (int k = 0; k < network->layers[i + 1]->output_size; k++) {
                    float weight = network->layers[i + 1]->weights->data[j][k];
                    float delta = network->layers[i + 1]->delta->data[0][k];
                    error += weight * delta;
                }
                errors->data[0][j] = error;
            }
        } else {
            errors = calc_loss_gradient(layer->output, expected);
        }

        Matrix *transfer_derivative = derivative_m(layer->output, layer->activation);
        Matrix *delta = multiply(errors, transfer_derivative);

        destroy_matrix(errors);
        destroy_matrix(transfer_derivative);

        layer->delta = delta;
    }
}

void update_weights(Network *network, float learning_rate) {
    for (int i = 0; i < network->layer_count - 1; i++) {
        Layer *layer = network->layers[i];

        if (layer->delta == NULL) {
            printf("delta is null will not update weights\n");
            continue;
        }

        for (int j = 0; j < layer->output_size; j++) {
            for (int k = 0; k < layer->input_size; k++) {
                float delta = layer->delta->data[0][j];
                float input = layer->input->data[0][k];
                float weight = layer->weights->data[k][j];
                float new_weight = weight - learning_rate * delta * input;
                layer->weights->data[k][j] = new_weight;
            }
        }

        destroy_matrix(layer->delta);
        layer->delta = NULL;
    }
}

float calc_loss(Matrix *output, Matrix *expected) {
    Matrix *loss = subtract(output, expected);
    Matrix *squared_loss = power(loss, 2);
    float result = sum(squared_loss);
    destroy_matrix(loss);
    destroy_matrix(squared_loss);
    return result;
}

void train(Network *network, Matrix *input, Matrix *expected, int epochs, float learning_rate) {
    for (int i = 0; i < epochs; i++) {
        Matrix *output = forward(network, input);

        backward(network, expected);
        update_weights(network, learning_rate);

        float loss = calc_loss(output, expected);
        // if (i % 100 == 0)
        //     printf("loss: %f %d\n", loss, i);

        destroy_matrix(output);
    }
}