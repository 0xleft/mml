set -e

echo "Running..."

if [ "$1" = "test" ]; then
    cmake --build . --target test -- -j 4
    ./test
    exit 0
fi

if [ "$1" = "logic" ]; then
    cmake --build . --target logic_gates -- -j 4
    ./logic_gates
    exit 0
fi

cmake --build . --target mml -- -j 4

./mml $@