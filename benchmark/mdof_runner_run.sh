#!/bin/bash

# compile_clang outside pinocchio

# run clang code -mdof

echo "Running mdof using clang"
./mdof_runner_c "c"

# run gcc code - mdof

echo "Running mdof using gcc"
./mdof_runner_g "g"