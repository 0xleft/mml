#pragma once

//
// Created by plusleft on 10/26/2023.
//

enum Activation {
    SIGMOID,
    RELU,
    TANH,
    SOFTMAX
};

typedef enum Activation Activation;

struct Matrix {
    int rows;
    int cols;
    float *data;
};

typedef struct Matrix Matrix;

Matrix *create_matrix(int rows, int cols);
Matrix *transpose(Matrix *matrix);
Matrix *dot(Matrix *a, Matrix *b);
void destroy_matrix(Matrix *matrix);
Matrix *add(Matrix *a, Matrix *b);
Matrix *apply(Matrix *matrix, Activation activation);
float derivative(float x, Activation activation);
Matrix *subtract(Matrix *a, Matrix *b);
Matrix *multiply_s(Matrix *matrix, float scalar);
Matrix *multiply(Matrix *a, Matrix *b);
Matrix *loss_gradient(Matrix *output, Matrix *expected);
Matrix *power(Matrix *matrix, float scalar);
float sum(Matrix *matrix);
void print_matrix(Matrix *matrix);