cmake_minimum_required(VERSION 3.16.3)
project(mml C)

set(CMAKE_C_STANDARD 11)

add_executable(mml main.c
        nn.c
        nn.h
        visual.c
        visual.h
        matrix.c
        matrix.h)

add_executable(test test.c
        nn.c
        nn.h
        visual.c
        visual.h
        matrix.c
        matrix.h)

# include math
target_link_libraries(mml m)
target_link_libraries(test m)