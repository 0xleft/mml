//
// Created by plusleft on 10/25/2023.
//

#include "test.h"
#include "nn.h"

#define RED "\033[0;31m"
#define RESET "\033[0m"

void test_dataset_functions() {
    printf("test_dataset_functions\n");
    Dataset *dataset = create_dataset(2);

    Matrix *input = create_matrix(1, 2);
    input->data[0][0] = 1;
    input->data[0][1] = 2;

    Matrix *expected = create_matrix(1, 2);
    expected->data[0][0] = 0.3f;
    expected->data[0][1] = 0.5f;

    add_data(dataset, input, expected);

    printf("dataset->size: %d\n", dataset->size);
    print_dataset(dataset);

    destroy_dataset(dataset);
}

void test_matrix_functions() {
    printf("test_matrix_functions\n");
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
        printf(RED"dot(a, b) failed\n"RESET);
    }

    Matrix *d = add(a, b);
    if (d->data[0][0] != 6 || d->data[0][1] != 8 || d->data[1][0] != 10 || d->data[1][1] != 12) {
        printf(RED"add(a, b) failed\n"RESET);
    }

    Matrix *e = multiply_s(a, 2);
    if (e->data[0][0] != 2 || e->data[0][1] != 4 || e->data[1][0] != 6 || e->data[1][1] != 8) {
        printf(RED"multiply_s(a, 2) failed\n"RESET);
    }

    Matrix *f = power(a, 2);
    if (f->data[0][0] != 1 || f->data[0][1] != 4 || f->data[1][0] != 9 || f->data[1][1] != 16) {
        printf(RED"power(a, 2) failed\n"RESET);
    }

    float g = sum(a);
    if (g != 10) {
        printf(RED"sum(a) failed\n"RESET);
    }

    Matrix *h = transpose(a);
    if (h->data[0][0] != 1 || h->data[0][1] != 3 || h->data[1][0] != 2 || h->data[1][1] != 4) {
        printf(RED"transpose(a) failed\n"RESET);
    }

    Matrix *i = subtract(a, b);
    if (i->data[0][0] != -4 || i->data[0][1] != -4 || i->data[1][0] != -4 || i->data[1][1] != -4) {
        printf(RED"subtract(a, b) failed\n"RESET);
    }

    Matrix *j = multiply(a, b);
    if (j->data[0][0] != 5 || j->data[0][1] != 12 || j->data[1][0] != 21 || j->data[1][1] != 32) {
        printf(RED"multiply(a, b) failed\n"RESET);
    }

    Matrix *k = create_matrix_from_array(2, 2, (float[]) {1, 2, 3, 4});
    print_matrix(k);
    if (k->data[0][0] != 1 || k->data[0][1] != 2 || k->data[1][0] != 3 || k->data[1][1] != 4) {
        printf(RED"create_matrix_from_array failed\n"RESET);
    }

    Matrix *l = create_matrix_from_array(2, 2, (float[]) {1, 2, 3, 4});
    Matrix *m = pad(l, 1);
    print_matrix(m);

    if (m->data[0][0] != 0 || m->data[0][1] != 0 || m->data[0][2] != 0 || m->data[0][3] != 0 ||
        m->data[1][0] != 0 || m->data[1][1] != 1 || m->data[1][2] != 2 || m->data[1][3] != 0 ||
        m->data[2][0] != 0 || m->data[2][1] != 3 || m->data[2][2] != 4 || m->data[2][3] != 0 ||
        m->data[3][0] != 0 || m->data[3][1] != 0 || m->data[3][2] != 0 || m->data[3][3] != 0) {
        printf(RED"pad failed\n"RESET);
    }

    // test loading images
    Matrix *image = from_image("tests/cross.png");
    if (image->rows != 32 || image->cols != 32) {
        printf(RED"from_image failed\n"RESET);
    }

    // test get_slice
    Matrix *slice = get_slice(image, 1, 1, 2, 2);
    if (slice->rows != 2 || slice->cols != 2) {
        printf(RED"get_slice failed\n"RESET);
    }

    // test flatten
    Matrix *flat = flatten(slice);
    if (flat->rows != 1 || flat->cols != 4) {
        printf(RED"flatten failed\n"RESET);
    }

    Matrix *ac = flip(a);
    if (ac->data[0][0] != 4 || ac->data[0][1] != 3 || ac->data[1][0] != 2 || ac->data[1][1] != 1) {
        printf(RED"flip failed\n"RESET);
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
    destroy_matrix(k);
    destroy_matrix(l);
    destroy_matrix(m);
    destroy_matrix(image);
    destroy_matrix(slice);
    destroy_matrix(flat);
    destroy_matrix(ac);
}

void test_nn_functions() {
    printf("test_nn_functions\n");
    Network *network;
    network = create_network(1);
    Layer *layer1 = create_dense_layer_l(2, 1, RELU, 0.1f, 0.1f);

    srand(0);
    add_layer(network, layer1);

    Matrix *input = create_matrix_from_array(1, 2, (float[]) {1, 2});
    Matrix *expected = create_matrix_from_array(1, 1, (float[]) {0.1f});

    print_matrix(network->layers[0]->layer.dense->weights);

    Matrix *output = forward(network, input);
    print_matrix(output);

    for (int i = 0; i < 10000; i++)
        train_input(network, input, expected, 0.1f);

    output = forward(network, input);
    print_matrix(output);

    if (output->data[0][0] - expected->data[0][0] > 0.1f) {
        printf(RED"train_input failed\n"RESET);
    }

    destroy_matrix(input);
    destroy_matrix(expected);
    destroy_matrix(output);
    destroy_network(network);
}

int main() {
    test_matrix_functions();
    test_dataset_functions();
    test_nn_functions();
    printf("done\n");

    return 0;
}