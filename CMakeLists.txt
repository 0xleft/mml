cmake_minimum_required(VERSION 3.16.3)
project(mml C)

set(CMAKE_C_STANDARD 11)

add_executable(mml main.c
        nn.c
        nn.h
        matrix.c
        matrix.h
        data.c
        data.h
        layers.c
        layers.h
        layers/dense.c
        layers/dense.h
        layers/conv2d.c
        layers/conv2d.h
        layers/common.c
        layers/common.h
)

add_executable(test test.c
        nn.c
        nn.h
        matrix.c
        matrix.h
        data.c
        data.h
        layers.c
        layers.h
        layers/dense.c
        layers/dense.h
        layers/conv2d.c
        layers/conv2d.h
        layers/common.c
        layers/common.h
)

add_executable(logic_gates logic.c
        nn.c
        nn.h
        matrix.c
        matrix.h
        data.c
        data.h
        layers.c
        layers.h
        layers/dense.c
        layers/dense.h
        layers/conv2d.c
        layers/conv2d.h
        layers/common.c
        layers/common.h
)

add_executable(conv conv.c
        nn.c
        nn.h
        matrix.c
        matrix.h
        data.c
        data.h
        layers.c
        layers.h
        layers/dense.c
        layers/dense.h
        layers/conv2d.c
        layers/conv2d.h
        layers/common.c
        layers/common.h
)

# include math
target_link_libraries(mml m)
target_link_libraries(test m)
target_link_libraries(logic_gates m)
target_link_libraries(conv m)

# include libpng
find_package(PNG REQUIRED)
target_link_libraries(mml ${PNG_LIBRARY})
target_link_libraries(test ${PNG_LIBRARY})
target_link_libraries(logic_gates ${PNG_LIBRARY})
target_link_libraries(conv ${PNG_LIBRARY})

