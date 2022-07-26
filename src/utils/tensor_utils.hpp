
// Author- Shubham Singh, singh281@utexas.edu
// Utilities for Eigen::Tensor slicing and accessing matrices, vectors

#ifndef __pinocchio_utils_tensor_utils_hpp__
#define __pinocchio_utils_tensor_utils_hpp__

#include <iostream>
using namespace std;

namespace pinocchio {

template <typename T>
void hess_assign_fd(Eigen::Tensor<double, 3> &hess,
                    const Eigen::MatrixBase<T> &mat, int n, int k) {
  for (int ii = 0; ii < n; ii++) {
    for (int jj = 0; jj < n; jj++) {
      hess(jj, ii, k) = mat(jj, ii);
    }
  }
}

} // namespace pinocchio

#endif //
