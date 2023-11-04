//
// Created by plusleft on 10/30/2023.
//

#include "nn.h"

int main() {
    Network *network;
    srand(time(NULL));
    network = create_network(5);
    // we are going for mnist
    float learning_rate = 0.001f;
    float epsilon = 0.00000001f;
    float decay = 0.01f;

    add_layer(network, create_flatten_layer_l(28, 1));
    add_layer(network, create_dense_layer_l(784, 1000, SIGMOID, epsilon, decay));
    add_layer(network, create_dense_layer_l(1000, 1000, RELU, epsilon, decay));
    add_layer(network, create_dense_layer_l(1000, 100, RELU, epsilon, decay));
    add_layer(network, create_dense_layer_l(100, 10, SIGMOID, epsilon, decay));

    Matrix *one = from_image("tests/mnist_my/one.png");
    Matrix *one_expected = create_matrix_from_array(1, 10, (float []) {1,0,0,0,0,0,0,0,0,0});
    Matrix *two = from_image("tests/mnist_my/two.png");
    Matrix *two_expected = create_matrix_from_array(1, 10, (float []) {0,1,0,0,0,0,0,0,0,0});
    Matrix *three = from_image("tests/mnist_my/three.png");
    Matrix *three_expected = create_matrix_from_array(1, 10, (float []) {0,0,1,0,0,0,0,0,0,0});
    Matrix *four = from_image("tests/mnist_my/four.png");
    Matrix *four_expected = create_matrix_from_array(1, 10, (float []) {0,0,0,1,0,0,0,0,0,0});
    Matrix *five = from_image("tests/mnist_my/five.png");
    Matrix *five_expected = create_matrix_from_array(1, 10, (float []) {0,0,0,0,1,0,0,0,0,0});
    Matrix *six = from_image("tests/mnist_my/six.png");
    Matrix *six_expected = create_matrix_from_array(1, 10, (float []) {0,0,0,0,0,1,0,0,0,0});
    Matrix *seven = from_image("tests/mnist_my/seven.png");
    Matrix *seven_expected = create_matrix_from_array(1, 10, (float []) {0,0,0,0,0,0,1,0,0,0});
    Matrix *eight = from_image("tests/mnist_my/eight.png");
    Matrix *eight_expected = create_matrix_from_array(1, 10, (float []) {0,0,0,0,0,0,0,1,0,0});
    Matrix *nine = from_image("tests/mnist_my/nine.png");
    Matrix *nine_expected = create_matrix_from_array(1, 10, (float []) {0,0,0,0,0,0,0,0,1,0});
    Matrix *zero = from_image("tests/mnist_my/zero.png");
    Matrix *zero_expected = create_matrix_from_array(1, 10, (float []) {0,0,0,0,0,0,0,0,0,1});

    Dataset *mnist_my_dataset = create_dataset(10);

    add_data(mnist_my_dataset, one, one_expected);
    add_data(mnist_my_dataset, two, two_expected);
    add_data(mnist_my_dataset, three, three_expected);
    add_data(mnist_my_dataset, four, four_expected);
    add_data(mnist_my_dataset, five, five_expected);
    add_data(mnist_my_dataset, six, six_expected);
    add_data(mnist_my_dataset, seven, seven_expected);
    add_data(mnist_my_dataset, eight, eight_expected);
    add_data(mnist_my_dataset, nine, nine_expected);
    add_data(mnist_my_dataset, zero, zero_expected);

    Matrix *output = forward(network, one);
    printf("output:\n");
    print_matrix(output);

    printf("loss: %f\n", calc_loss(output, one_expected));

    train_dataset(network, mnist_my_dataset, 1000, learning_rate);

    destroy_dataset(mnist_my_dataset);
    destroy_network(network);
    return 0;
}