#!/bin/bash



# compile clang code- kinematic trees

echo "compiling tree using CLANG"
$(clang++-10 -DNDEBUG -I /usr/include/eigen3 -I /usr/include -O3 -L /usr/local/lib -l pinocchio -std=c++11 -march=native tree_runner.cpp -o tree_runner_c) # compiling tree clang here
echo "compilation using CLANG complete!"

# compile gcc code - kinematic trees

echo "compiling tree using GCC"
$(g++ tree_runner.cpp -DNDEBUG -I /usr/include/eigen3 -I /usr/include -O3 -L /usr/local/lib -l pinocchio -march=native -ldl -o tree_runner_g)
echo "compilation using GCC complete!"

# trying -ffast-math here- didn't help