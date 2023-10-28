//
// Created by plusleft on 10/28/2023.
//

#include "data.h"
#include <stdlib.h>

Dataset *create_dataset(int max_size) {
    Dataset *dataset = malloc(sizeof(Dataset));
    dataset->max_size = max_size;
    dataset->size = 0;
    dataset->inputs = malloc(sizeof(Matrix *) * max_size);
    dataset->outputs = malloc(sizeof(Matrix *) * max_size);
    return dataset;
}

void add_data(Dataset *dataset, Matrix *input, Matrix *output) {
    dataset->inputs[dataset->size] = input;
    dataset->outputs[dataset->size] = output;
    dataset->size++;
}

void destroy_dataset(Dataset *dataset) {
    for (int i = 0; i < dataset->size; i++) {
        destroy_matrix(dataset->inputs[i]);
        destroy_matrix(dataset->outputs[i]);
    }
    free(dataset->inputs);
    free(dataset->outputs);
    free(dataset);
}

void print_dataset(Dataset *dataset) {
    for (int i = 0; i < dataset->size; i++) {
        printf("data %d:\n", i);
        printf("input:\n");
        print_matrix(dataset->inputs[i]);
        printf("output:\n");
        print_matrix(dataset->outputs[i]);
    }
}

// TODO
Dataset *load_csv(char *path) {
    return NULL;
}