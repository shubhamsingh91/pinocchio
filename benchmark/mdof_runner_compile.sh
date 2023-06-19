#!/bin/bash



# compile clang code- mdof

 echo "compiling mdof_runner using clang"
 $(clang++-10 -DNDEBUG -I /usr/include/eigen3 -I /usr/include -O3 -L /usr/local/lib -l pinocchio -std=c++11 -march=native mdof_runner.cpp -o mdof_runner_c) # compiling  clang here
 echo "compilation complete!"

# compile gcc code - mdof
# trying -ffast-math here- didn't help

echo "compiling mdof_v5_avx using gcc"
$(g++ mdof_runner.cpp -DNDEBUG -I /usr/include/eigen3 -I /usr/include -O3 -L /usr/local/lib -l pinocchio -march=native -ldl -o mdof_runner_g)
echo "compilation complete!"