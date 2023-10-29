//
// Created by plusleft on 10/29/2023.
//

#include "data.h"
#include <stdlib.h>

Dataset *create_dataset(int max_size) {
    Dataset *dataset = malloc(sizeof(Dataset));
    dataset->size = 0;
    dataset->max_size = max_size;
    dataset->inputs = malloc(sizeof(Matrix *) * max_size);
    dataset->expected = malloc(sizeof(Matrix *) * max_size);
    return dataset;
}

void destroy_dataset(Dataset *dataset) {
    for (int i = 0; i < dataset->size; i++) {
        destroy_matrix(dataset->inputs[i]);
        destroy_matrix(dataset->expected[i]);
    }
    free(dataset->inputs);
    free(dataset->expected);
    free(dataset);
}

void add_data(Dataset *dataset, Matrix *input, Matrix *expected) {
    dataset->inputs[dataset->size] = input;
    dataset->expected[dataset->size] = expected;
    dataset->size++;
}

void print_dataset(Dataset *dataset) {
    for (int i = 0; i < dataset->size; i++) {
        printf("Input:\n");
        print_matrix(dataset->inputs[i]);
        printf("Expected:\n");
        print_matrix(dataset->expected[i]);
    }
}

Dataset *load_csv(Dataset *dataset) {
    return NULL;
}