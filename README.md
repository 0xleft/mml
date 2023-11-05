# mml

Simple machine learning project written in pure c.

## Features

- [x] Basic neural network implementation
- [x] Back propagation
- [x] RMSProp optimizer
- [x] Datasets
- [x] Max pooling
- [x] Flatten
- [x] Convolution?

## Dependencies

```bash
# for loading images into matrices
sudo apt install zlib1g-dev
sudo apt install libpng-dev
```

## Usage

### Logic gates

```bash
./run.sh logic_gates

# output
Select a test:
1. and
2. or
3. xor
1
epoch 0 loss 1.322414
...
epoch 990 loss 0.000118
0 0: 0.000014
0 1: 0.000245
1 0: 0.007272
1 1: 0.992032
```

### SMALL MNIST

Currently only a flatten and then dense network.
We are basically looking if the number is lower than 5 or not.

```bash
./run.sh conv

# output
training...
epoch 0 loss 3.279951
...
epoch 2990 loss 0.043303
trained
output1:
1 1
0.964929

output9:
1 1
0.047376
```

### TESTS

```bash
./run.sh test
```

If you see any red text, it means one of the tests failed.