//
// Created by plusleft on 10/25/2023.
//

#include "test.h"
#include "nn.h"

void test_matrix_functions() {
    Matrix *a = create_matrix(2, 2);
    Matrix *b = create_matrix(2, 2);

    a->data[0][0] = 1;
    a->data[0][1] = 2;
    a->data[1][0] = 3;
    a->data[1][1] = 4;

    b->data[0][0] = 5;
    b->data[0][1] = 6;
    b->data[1][0] = 7;
    b->data[1][1] = 8;

    Matrix *c = dot(a, b);
    if (c->data[0][0] != 19 || c->data[0][1] != 22 || c->data[1][0] != 43 || c->data[1][1] != 50) {
        printf("dot(a, b) failed\n");
    }

    Matrix *d = add(a, b);
    if (d->data[0][0] != 6 || d->data[0][1] != 8 || d->data[1][0] != 10 || d->data[1][1] != 12) {
        printf("add(a, b) failed\n");
    }

    Matrix *e = multiply_s(a, 2);
    if (e->data[0][0] != 2 || e->data[0][1] != 4 || e->data[1][0] != 6 || e->data[1][1] != 8) {
        printf("multiply_s(a, 2) failed\n");
    }

    Matrix *f = power(a, 2);
    if (f->data[0][0] != 1 || f->data[0][1] != 4 || f->data[1][0] != 9 || f->data[1][1] != 16) {
        printf("power(a, 2) failed\n");
    }

    float g = sum(a);
    if (g != 10) {
        printf("sum(a) failed\n");
    }

    Matrix *h = transpose(a);
    if (h->data[0][0] != 1 || h->data[0][1] != 3 || h->data[1][0] != 2 || h->data[1][1] != 4) {
        printf("transpose(a) failed\n");
    }

    Matrix *i = subtract(a, b);
    if (i->data[0][0] != -4 || i->data[0][1] != -4 || i->data[1][0] != -4 || i->data[1][1] != -4) {
        printf("subtract(a, b) failed\n");
    }

    Matrix *j = multiply(a, b);
    if (j->data[0][0] != 5 || j->data[0][1] != 12 || j->data[1][0] != 21 || j->data[1][1] != 32) {
        printf("multiply(a, b) failed\n");
    }

    destroy_matrix(a);
    destroy_matrix(b);
    destroy_matrix(c);
    destroy_matrix(d);
    destroy_matrix(e);
    destroy_matrix(f);
    destroy_matrix(h);
    destroy_matrix(i);
    destroy_matrix(j);
}

void test_nn_functions() {
    Network *network;
    int layer_sizes[] = {2, 3, 4, 1};
    Activation activations[] = {SIGMOID, SIGMOID, RELU};
    network = create_network(3, layer_sizes, activations);

    // set seed to make results reproducible
    // srand(23);
    // randomize_network(network);
    initialize_weights_xavier(network);

    Matrix *input = create_matrix(1, 2);
    input->data[0][0] = 1;
    input->data[0][1] = 2;

    Matrix *expected = create_matrix(1, 1);
    expected->data[0][0] = 0.5f;

    Matrix *output = forward(network, input);
    print_matrix(output);
    destroy_matrix(output);

    train(network, input, expected, 10000, 0.1f);

    output = forward(network, input);
    print_matrix(output);

    if (output->data[0][0] - expected->data[0][0] > 0.1) {
        printf("Training failed\n");
    }

    destroy_matrix(input);
    destroy_matrix(expected);
    destroy_matrix(output);
    destroy_network(network);
}

int main() {
    test_matrix_functions();
    test_nn_functions();

    return 0;
}