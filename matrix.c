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
    matrix->data = malloc(sizeof(double) * rows * cols);
    return matrix;
}

/// You must free the result yourself and the input too.

Matrix *transpose(Matrix *matrix) {
    Matrix *result = create_matrix(matrix->cols, matrix->rows);
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            result->data[j * matrix->rows + i] = matrix->data[i * matrix->cols + j];
        }
    }

    return result;
}

Matrix *dot(Matrix *a, Matrix *b) {
    if (a->cols != b->rows) {
        return NULL;
    }

    Matrix *result = create_matrix(a->rows, b->cols);

    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            double sum = 0;
            for (int k = 0; k < a->cols; k++) {
                sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }
            result->data[i * result->cols + j] = sum;
        }
    }

    return result;
}

Matrix *add(Matrix *a, Matrix *b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        return NULL;
    }

    Matrix *result = create_matrix(a->rows, a->cols);

    for (int i = 0; i < a->rows * a->cols; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }

    return result;
}

Matrix *multiply_s(Matrix *matrix, float scalar) {
    Matrix *result = create_matrix(matrix->rows, matrix->cols);

    for (int i = 0; i < matrix->rows * matrix->cols; i++) {
        result->data[i] = matrix->data[i] * scalar;
    }

    return result;
}

Matrix *multiply(Matrix *a, Matrix *b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        return NULL;
    }

    Matrix *result = create_matrix(a->rows, a->cols);

    for (int i = 0; i < a->rows * a->cols; i++) {
        result->data[i] = a->data[i] * b->data[i];
    }

    return result;
}

Matrix *power(Matrix *matrix, float scalar) {
    Matrix *result = create_matrix(matrix->rows, matrix->cols);

    for (int i = 0; i < matrix->rows * matrix->cols; i++) {
        result->data[i] = pow(matrix->data[i], scalar);
    }

    return result;
}

float sum(Matrix *matrix) {
    float sum = 0;
    for (int i = 0; i < matrix->rows * matrix->cols; i++) {
        sum += matrix->data[i];
    }
    return sum;
}

Matrix *apply(Matrix *matrix, Activation activation) {
    Matrix *result = create_matrix(matrix->rows, matrix->cols);

    for (int i = 0; i < matrix->rows * matrix->cols; i++) {
        switch (activation) {
            case SIGMOID:
                result->data[i] = 1 / (1 + exp(-matrix->data[i]));
                break;
            case RELU:
                result->data[i] = matrix->data[i] > 0 ? matrix->data[i] : 0;
                break;
            case TANH:
                result->data[i] = tanh(matrix->data[i]);
                break;
            case SOFTMAX:
                result->data[i] = exp(matrix->data[i]);
                break;
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

Matrix *subtract(Matrix *a, Matrix *b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        return NULL;
    }

    Matrix *result = create_matrix(a->rows, a->cols);

    for (int i = 0; i < a->rows * a->cols; i++) {
        result->data[i] = a->data[i] - b->data[i];
    }

    return result;
}

void destroy_matrix(Matrix *matrix) {
    free(matrix->data);
    free(matrix);
}

void print_matrix(Matrix *matrix) {
    for (int i = 0; i < matrix->rows * matrix->cols; i++) {
        printf("%f ", matrix->data[i]);
        if ((i + 1) % matrix->cols == 0) {
            printf("\n");
        }
    }
}