#pragma once

//
// Created by plusleft on 10/26/2023.
//

#include <stdlib.h>
#include <stdio.h>

enum Activation {
    SIGMOID,
    RELU,
    TANH,
    LEAKY_RELU,
};

typedef enum Activation Activation;

struct Matrix {
    int rows;
    int cols;
    float **data;
};

typedef struct Matrix Matrix;

struct Matrix3D {
    Matrix **data;
    int depth;
};

typedef struct Matrix3D Matrix3D;

Matrix3D *create_matrix_3d(int depth);
void destroy_matrix_3d(Matrix3D *matrix);
Matrix *create_matrix(int rows, int cols);
Matrix *transpose(Matrix *matrix);
void destroy_matrix(Matrix *matrix);
Matrix *add(Matrix *a, Matrix *b);
Matrix *apply(Matrix *matrix, Activation activation);
float derivative(float x, Activation activation);
Matrix *subtract(Matrix *a, Matrix *b);
Matrix *multiply_s(Matrix *matrix, float scalar);
Matrix *multiply(Matrix *a, Matrix *b);
Matrix *power(Matrix *matrix, float scalar);
float sum(Matrix *matrix);
void print_matrix(Matrix *matrix);
void print_dim(Matrix *matrix);
Matrix *derivative_m(Matrix *matrix, Activation activation);
Matrix *dot(Matrix *a, Matrix *b);
Matrix *copy_matrix(Matrix *matrix);
float sum(Matrix *matrix);
Matrix *create_matrix_from_array(int rows, int cols, float *data);
Matrix *pad(Matrix *matrix, int pad_size);
Matrix *get_slice(Matrix *matrix, int start_row, int start_col, int rows, int cols);
Matrix *from_image(char *filename);
Matrix *flatten(Matrix *matrix);
float activate(float x, Activation activation);
Matrix *flip(Matrix *matrix);
float max(Matrix *matrix);
Matrix *convolve(Matrix *input, Matrix *kernel, int stride, int kernel_size, int padding, int output_size);
Matrix *value_ind(Matrix *matrix, float value);