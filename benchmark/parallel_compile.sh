#!/bin/bash

echo "Compiling parallel.cpp with OpenMP..."

# Use variables for clarity
SRC=parallel.cpp
OUT=parallel

# Compile with OpenMP and optimizations
g++ "$SRC" -o "$OUT" \
    -DNDEBUG \
    -O3 \
    -march=native \
    -fopenmp \
    -I/usr/include/eigen3 \
    -I/usr/include \
    -L/usr/local/lib \
    -lpinocchio \
    -ldl

# Check success
if [ $? -eq 0 ]; then
    echo "Compilation complete! Binary: $OUT"
else
    echo "❌ Compilation failed."
    exit 1
fi
