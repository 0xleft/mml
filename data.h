#pragma once

//
// Created by plusleft on 10/29/2023.
//

#include "matrix.h"

struct Dataset {
    int size;
    int max_size;
    Matrix **inputs;
    Matrix **expected;
};

typedef struct Dataset Dataset;

Dataset *create_dataset(int max_size);
void destroy_dataset(Dataset *dataset);