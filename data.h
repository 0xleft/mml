#pragma once
//
// Created by plusleft on 10/28/2023.
//

#include "matrix.h"

struct Dataset {
    int max_size;
    int size;
    Matrix **inputs;
    Matrix **outputs;
};

typedef struct Dataset Dataset;

Dataset *create_dataset(int max_size);
Dataset *load_csv(char *path);
void destroy_dataset(Dataset *dataset);
void add_data(Dataset *dataset, Matrix *input, Matrix *output);
void print_dataset(Dataset *dataset);