//
// Created by plusleft on 10/31/2023.
//

#include "common.h"

void initialize_weights_xavier_norm(int input_size, int output_size, Matrix *weights) {
    float lower_bound = -(sqrt(6) / sqrt(input_size + output_size));
    float upper_bound = sqrt(6) / sqrt(input_size + output_size);
    for (int j = 0; j < weights->rows; j++) {
        for (int k = 0; k < weights->cols; k++) {
            weights->data[j][k] = (float) rand() / (float) (RAND_MAX / (upper_bound - lower_bound)) + lower_bound;
        }
    }
}