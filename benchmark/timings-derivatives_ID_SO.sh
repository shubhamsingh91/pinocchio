#!/bin/bash

# compile GCC Code
$(g++ timings-derivatives_ID_SO.cpp -DNDEBUG -I /usr/include/eigen3 -O0 -I /usr/include -L /usr/local/lib -l pinocchio -o timings-derivatives_ID_SO) # compiling using gcc here

