//
// Created by plusleft on 10/26/2023.
//

#include <stdlib.h>
#include "matrix.h"
#include "nn.h"
#include <math.h>
// include lib png
#include <png.h>
#include <stdio.h>

Matrix *create_matrix(int rows, int cols) {
    Matrix *matrix = malloc(sizeof(Matrix));
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->data = malloc(sizeof(float *) * rows);
    for (int i = 0; i < rows; i++) {
        matrix->data[i] = malloc(sizeof(float) * cols);
        for (int j = 0; j < cols; j++) {
            matrix->data[i][j] = 0;
        }
    }
    return matrix;
}

Matrix *create_matrix_from_array(int rows, int cols, float *data) {
    Matrix *matrix = malloc(sizeof(Matrix));
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->data = malloc(sizeof(float *) * rows);
    for (int i = 0; i < rows; i++) {
        matrix->data[i] = malloc(sizeof(float) * cols);
        for (int j = 0; j < cols; j++) {
            matrix->data[i][j] = data[i * cols + j];
        }
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
    matrix->data = NULL;
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

// pad border with 0s
Matrix *pad(Matrix *matrix, int padding) {
    Matrix *result = create_matrix(matrix->rows + padding * 2, matrix->cols + padding * 2);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            if (i < padding || i >= result->rows - padding || j < padding || j >= result->cols - padding) {
                result->data[i][j] = 0;
            } else {
                result->data[i][j] = matrix->data[i - padding][j - padding];
            }
        }
    }
    return result;
}

Matrix *get_slice(Matrix *matrix, int row_start, int col_start, int row_size, int col_size) {
    Matrix *result = create_matrix(row_size, col_size);
    for (int i = 0; i < row_size; i++) {
        for (int j = 0; j < col_size; j++) {
            result->data[i][j] = matrix->data[row_start + i][col_start + j];
        }
    }
    return result;
}

// load image from file
Matrix *from_image(char *filename) {
    printf("from_image\n");
    printf("filename: %s\n", filename);

    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("file not found\n");
        return NULL;
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        printf("png_create_read_struct failed\n");
        return NULL;
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        printf("png_create_info_struct failed\n");
        return NULL;
    }

    if (setjmp(png_jmpbuf(png))) {
        printf("setjmp failed\n");
        return NULL;
    }

    png_init_io(png, fp);
    png_read_info(png, info);

    int width = png_get_image_width(png, info);
    int height = png_get_image_height(png, info);

    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth = png_get_bit_depth(png, info);

    if (bit_depth == 16) {
        png_set_strip_16(png);
    }

    if (color_type == PNG_COLOR_TYPE_PALETTE) {
        png_set_palette_to_rgb(png);
    }

    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) {
        png_set_expand_gray_1_2_4_to_8(png);
    }

    if (png_get_valid(png, info, PNG_INFO_tRNS)) {
        png_set_tRNS_to_alpha(png);
    }

    if (color_type == PNG_COLOR_TYPE_RGB || color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_PALETTE) {
        png_set_filler(png, 0xFF, PNG_FILLER_AFTER);
    }

    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA) {
        png_set_gray_to_rgb(png);
    }

    png_read_update_info(png, info);

    png_bytep *row_pointers = malloc(sizeof(png_bytep) * height);
    for (int y = 0; y < height; y++) {
        row_pointers[y] = malloc(png_get_rowbytes(png, info));
    }

    png_read_image(png, row_pointers);

    Matrix *result = create_matrix(height, width);

    for (int y = 0; y < height; y++) {
        png_bytep row = row_pointers[y];
        for (int x = 0; x < width; x++) {
            png_bytep px = &(row[x * 4]);
            result->data[y][x] = px[0];
        }
    }

    for (int y = 0; y < height; y++) {
        free(row_pointers[y]);
    }

    free(row_pointers);
    png_destroy_read_struct(&png, &info, NULL);
    fclose(fp);

    return result;
}

Matrix *flatten(Matrix *matrix) {
    Matrix *result = create_matrix(1, matrix->rows * matrix->cols);
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; ++j) {
            result->data[0][i * matrix->cols + j] = matrix->data[i][j];
        }
    }
    return result;
}

Matrix *flip(Matrix *matrix) {
    Matrix *result = create_matrix(matrix->rows, matrix->cols);
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = matrix->cols - 1; j >= 0; j--) {
            result->data[i][matrix->cols - 1 - j] = matrix->data[i][j];
        }
    }
    return result;
}

Matrix *convolve(Matrix *input, Matrix *kernel, int stride, int kernel_size, int padding, int output_size) {
    // move output to internal cuz we can calcualte it here
    // todo

    Matrix *padded_input = pad(input, padding);
    Matrix *result = create_matrix(output_size, output_size);

    // todo flip kernel

    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < output_size; j++) {
            Matrix *input_slice = get_slice(padded_input, i * stride, j * stride, kernel_size, kernel_size);

            // dot product
            float sum = 0;
            for (int k = 0; k < kernel_size; k++) {
                for (int l = 0; l < kernel_size; l++) {
                    sum += input_slice->data[k][l] * kernel->data[k][l];
                }
            }

            result->data[i][j] = sum;

            destroy_matrix(input_slice);
        }
    }

    destroy_matrix(padded_input);

    return result;
}