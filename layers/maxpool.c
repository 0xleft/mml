//
// Created by plusleft on 11/2/2023.
//

#include "maxpool.h"

MaxPoolLayer *create_maxpool_layer(int input_size, int input_count, int stride, int kernel_size) {
    MaxPoolLayer *layer = malloc(sizeof(MaxPoolLayer));
    layer->input_size = input_size;
    layer->input_count = input_count;
    layer->output_size = (input_size - kernel_size + 2 * 0) / stride + 1;
    layer->stride = stride;
    layer->kernel_size = kernel_size;
    layer->input = create_matrix_3d(input_count);
    layer->output = create_matrix_3d(input_count);
    layer->mask = create_matrix_3d(input_count);
    return layer;
}

void destroy_maxpool_layer(MaxPoolLayer *layer) {
    if (layer == NULL) {
        return;
    }
    destroy_matrix_3d(layer->input);
    destroy_matrix_3d(layer->output);
    destroy_matrix_3d(layer->mask);
    free(layer);
    layer = NULL;
}

Matrix *forward_maxpool_single(MaxPoolLayer *layer, Matrix *input, int index) {
    Matrix *result = create_matrix(layer->output_size, layer->output_size);

    Matrix *mask = create_matrix(layer->input_size, layer->input_size);

    for (int i = 0; i < layer->output_size; i++) {
        for (int j = 0; j < layer->output_size; j++) {
            Matrix *input_slice = get_slice(input, i * layer->stride, j * layer->stride, layer->kernel_size, layer->kernel_size);

            float max_value = max(input_slice);
            result->data[i][j] = max_value;

            // create mask
            int o = 0;
            for (int k = i * layer->stride; k < i * layer->stride + layer->kernel_size; k++) {
                int s = 0;
                for (int p = j * layer->stride; p < j * layer->stride + layer->kernel_size; p++) {
                    if (max_value == input_slice->data[o][s]) {
                        mask->data[k][p] = 1;
                    }
                    s++;
                }
                o++;
            }

            destroy_matrix(input_slice);
        }
    }


    layer->mask->data[index] = mask;
    return result;
}

Matrix3D *forward_maxpool(MaxPoolLayer *layer, Matrix3D *input) {
    Matrix3D *output = create_matrix_3d(layer->input_count);

    for (int i = 0; i < layer->input_count; i++) {
        output->data[i] = forward_maxpool_single(layer, input->data[i], i);
        layer->input->data[i] = copy_matrix(input->data[i]);
        layer->output->data[i] = copy_matrix(output->data[i]);
    }

    return output;
}

Matrix *backward_maxpool_single(MaxPoolLayer *layer, Matrix *loss_gradient, int index) {
    Matrix *dout = create_matrix(layer->input_size, layer->input_size);

    for (int i = 0; i < loss_gradient->rows; i++) {
        for (int j = 0; j < loss_gradient->cols; j++) {

            float d_X = loss_gradient->data[i][j];

            for (int k = i * layer->stride; k < i * layer->stride + layer->kernel_size; k++) {
                for (int p = j * layer->stride; p < j * layer->stride + layer->kernel_size; p++) {
                    dout->data[k][p] = layer->mask->data[index]->data[k][p] * d_X;
                }
            }
        }
    }

    return dout;
}

Matrix3D *backward_maxpool(MaxPoolLayer *layer, Matrix3D *loss_gradient) {
    Matrix3D *dout = create_matrix_3d(layer->input_count);

    for (int i = 0; i < layer->input_count; i++) {
        dout->data[i] = backward_maxpool_single(layer, loss_gradient->data[i], i);
    }

    return dout;
}