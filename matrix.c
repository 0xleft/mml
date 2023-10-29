//
// Created by plusleft on 10/26/2023.
//

#include <stdlib.h>
#include "matrix.h"
#include "nn.h"
#include <math.h>

Matrix *create_matrix(int rows, int cols) {
    Matrix *matrix = malloc(sizeof(Matrix));
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->data = malloc(sizeof(float *) * rows);
    for (int i = 0; i < rows; i++) {
        matrix->data[i] = malloc(sizeof(float) * cols);
    }
    return matrix;
}

Matrix *transpose(Matrix *matrix) {
    Matrix *result = create_matrix(matrix->cols, matrix->rows);
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            result->data[j][i] = matrix->data[i][j];
        }
    }
    return result;
}

void print_dim(Matrix *matrix) {
    printf("%d %d\n", matrix->rows, matrix->cols);
}

// dot product
Matrix *dot(Matrix *a, Matrix *b) {
    Matrix *result = create_matrix(a->rows, b->cols);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            float sum = 0;
            for (int k = 0; k < a->cols; k++) {
                sum += a->data[i][k] * b->data[k][j];
            }
            result->data[i][j] = sum;
        }
    }
    return result;
}

Matrix *multiply(Matrix *a, Matrix *b) {
    Matrix *result = create_matrix(a->rows, a->cols);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            result->data[i][j] = a->data[i][j] * b->data[i][j];
        }
    }
    return result;
}

Matrix *add(Matrix *a, Matrix *b) {
    Matrix *result = create_matrix(a->rows, a->cols);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            result->data[i][j] = a->data[i][j] + b->data[i][j];
        }
    }
    return result;
}

Matrix *multiply_s(Matrix *matrix, float scalar) {
    Matrix *result = create_matrix(matrix->rows, matrix->cols);
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            result->data[i][j] = matrix->data[i][j] * scalar;
        }
    }
    return result;
}

Matrix *power(Matrix *matrix, float scalar) {
    Matrix *result = create_matrix(matrix->rows, matrix->cols);
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            result->data[i][j] = pow(matrix->data[i][j], scalar);
        }
    }
    return result;
}

float sum(Matrix *matrix) {
    float sum = 0;
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            sum += matrix->data[i][j];
        }
    }
    return sum;
}

float activate(float x, Activation activation) {
    switch (activation) {
        case SIGMOID:
            return 1 / (1 + exp(-x));
        case RELU:
            return x > 0 ? x : 0;
        case TANH:
            return tanh(x);
        case SOFTMAX:
            return exp(x);
    }
}

Matrix *apply(Matrix *matrix, Activation activation) {
    Matrix *result = create_matrix(matrix->rows, matrix->cols);
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            result->data[i][j] = activate(matrix->data[i][j], activation);
        }
    }
    return result;
}

float derivative(float x, Activation activation) {
    switch (activation) {
        case SIGMOID:
            return x * (1 - x);
        case RELU:
            return x > 0 ? 1 : 0;
        case TANH:
            return 1 - x * x;
        case SOFTMAX:
            return x * (1 - x);
    }
}

Matrix *derivative_m(Matrix *matrix, Activation activation) {
    Matrix *result = create_matrix(matrix->rows, matrix->cols);
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            result->data[i][j] = derivative(matrix->data[i][j], activation);
        }
    }
    return result;
}

Matrix *subtract(Matrix *a, Matrix *b) {
    Matrix *result = create_matrix(a->rows, a->cols);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            result->data[i][j] = a->data[i][j] - b->data[i][j];
        }
    }
    return result;
}

Matrix *copy_matrix(Matrix *matrix) {
    Matrix *result = create_matrix(matrix->rows, matrix->cols);
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; ++j) {
            result->data[i][j] = matrix->data[i][j];
        }
    }
    return result;
}

void destroy_matrix(Matrix *matrix) {
    if (matrix == NULL) {
        printf("matrix is null\n");
        return;
    }
    if (matrix->data == NULL) {
        printf("data is null\n");
        return;
    }
    free(matrix->data);
    free(matrix);
    matrix = NULL;
}

void print_matrix(Matrix *matrix) {
    if (matrix == NULL) {
        printf("matrix is null\n");
        return;
    }

    printf("%d %d\n", matrix->rows, matrix->cols);
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            printf("%f ", matrix->data[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}