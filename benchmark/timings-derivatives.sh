#!/bin/bash

# compile GCC Code
$(g++ timings-derivatives.cpp -DNDEBUG -I /usr/include/eigen3 -O3 -I /usr/include -L /usr/local/lib -l pinocchio -march=native -o timings-derivatives) # compiling using gcc here

