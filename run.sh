set -e

echo "Running..."

cmake .

if [ "$1" = "" ]; then
  echo "No target selected"
  exit 0
fi

cmake --build . --target $1 -- -j 4
./$1
exit 0