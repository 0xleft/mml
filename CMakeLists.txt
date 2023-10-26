cmake_minimum_required(VERSION 3.16.3)
project(mml C)

set(CMAKE_C_STANDARD 11)

add_executable(mml main.c
        nn.c
        nn.h
        visual.c
        visual.h)

add_executable(test test.c
        nn.c
        nn.h
        visual.c
        visual.h)