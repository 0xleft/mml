//
// Created by plusleft on 10/30/2023.
//

#include "nn.h"

int main() {
    Network *network;
    network = create_network(1);
    add_layer(network, create_conv2d_layer_l(1, 0, 3, 7, RELU, 0.0f, 0.0f));

    Matrix *input = from_image("tests/small.png");
    Matrix *expected = from_image("tests/small_expected.png");
    printf("input %dx%d\n", input->rows, input->cols);
    print_matrix(input);
    printf("expected %dx%d\n", expected->rows, expected->cols);
    print_matrix(expected);

    Matrix *output = forward(network, input);
    print_matrix(output);

    for (int i = 0; i < 10000; i++) {
        float loss = train_input(network, input, expected, 0.1f);

        if (i % 100 == 0) {
            printf("loss: %f\n", loss);
        }
    }

    destroy_matrix(input);
    destroy_matrix(output);
    destroy_network(network);
    return 0;
}