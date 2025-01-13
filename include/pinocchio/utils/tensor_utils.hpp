
// Author- Shubham Singh, singh281@utexas.edu
// Utilities for Eigen::Tensor slicing and accessing matrices, vectors

#ifndef __pinocchio_utils_tensor_utils_hpp__
#define __pinocchio_utils_tensor_utils_hpp__

#include <iostream>
#include <math.h>

using namespace std;

namespace pinocchio {

template <typename T>
using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template <typename Scalar, int rank, typename sizeType>
Eigen::MatrixXd Tensor_to_Matrix(const Eigen::Tensor<Scalar, rank>& tensor, const sizeType rows, const sizeType cols)
{
    return Eigen::Map<const MatrixType<Scalar>>(tensor.data(), rows, cols);
}

// to get a matrix of a tensor with 3rd dim
void tensor_to_mat(Eigen::Tensor<double, 3>& hess, Eigen::MatrixXd& mat, int dim, int r)
{
    for (int ii = 0; ii < dim; ii++) {
        for (int jj = 0; jj < dim; jj++) {
            mat(ii, jj) = hess(ii, jj, r);
        }
    }
}


void error_fun_SO(const Eigen::Tensor<double,3> &ten_in, const Eigen::Tensor<double,3> &ten_ref, Eigen::VectorXd & error_vec,const int n1,const int n2,const int n3)
{
  Eigen::Tensor<double,3> temp_diff(n1,n2,n3);
  temp_diff = ten_in-ten_ref;
  // Max Abs error
  Eigen::Tensor<double, 0> b =(temp_diff.abs()).maximum();
  error_vec[0]= b(0);
  double sum1 = 0.0;

  for(int k=0;k<n3;k++){
    for (int j=0;j<n2;j++){
        for(int i=0;i<n1;i++){
            sum1+=temp_diff(i,j,k)*temp_diff(i,j,k);
        }
    }
  }
  // RMS abs error
  error_vec[1]=sqrt(sum1/(n1*n2*n3));
 
  // Relative error
 temp_diff.setZero();
 double sum2=0.0;

 for(int k=0;k<n3;k++){
    for (int j=0;j<n2;j++){
        for(int i=0;i<n1;i++){
        if(ten_in(i,j,k)==0.0){
            temp_diff(i,j,k)=0.0;
        }else
        { 
            temp_diff(i,j,k) = (ten_in(i,j,k)-ten_ref(i,j,k))/max(abs(ten_in(i,j,k)),1.0);
        }
       sum2+=temp_diff(i,j,k)*temp_diff(i,j,k);  
        }
    }
  }
  Eigen::Tensor<double, 0> c =(temp_diff.abs()).maximum();
  error_vec[2] = c(0); //max relative error
 error_vec[3] = sqrt(sum2/(n1*n2*n3));     // rms relative error

}

template <typename T>
T tensorNorm(const Eigen::Tensor<T, 3>& tensor) {
    T sum_of_squares = 0;
    for (int i = 0; i < tensor.dimension(0); ++i) {
        for (int j = 0; j < tensor.dimension(1); ++j) {
            for (int k = 0; k < tensor.dimension(2); ++k) {
                sum_of_squares += tensor(i, j, k) * tensor(i, j, k);
            }
        }
    }
    return std::sqrt(sum_of_squares);
}

template <typename T>
T tensorMax(const Eigen::Tensor<T, 3>& tensor) {
    T max_value = tensor(0, 0, 0); // Initialize with the first element
    for (int i = 0; i < tensor.dimension(0); ++i) {
        for (int j = 0; j < tensor.dimension(1); ++j) {
            for (int k = 0; k < tensor.dimension(2); ++k) {
                if (tensor(i, j, k) > max_value) {
                    max_value = tensor(i, j, k);
                }
            }
        }
    }
    return max_value;
}

template <typename T>
Eigen::Tensor<T, 3> rotR(const Eigen::Tensor<T, 3>& tens_in) {
    // Define the permutation order
    Eigen::array<int, 3> permute_dims = {0, 2, 1};

    // Perform the permutation
    Eigen::Tensor<T, 3> tens_out = tens_in.shuffle(permute_dims);

    return tens_out;
}

template <typename T>
Eigen::Tensor<T, 3> rotR_manual(const Eigen::Tensor<T, 3>& tens_in)
{
    // Get the dimensions of the input tensor
    const auto& dims = tens_in.dimensions();
    // dims[0], dims[1], dims[2]
    // The shuffle order {0, 2, 1} means the output shape is [D0, D2, D1].
    
    Eigen::Tensor<T, 3> tens_out(dims[0], dims[2], dims[1]);

    // Manually copy data into permuted positions
    // (i, j, k) -> (i, k, j)
    for (int i = 0; i < dims[0]; ++i)
    {
        for (int j = 0; j < dims[1]; ++j)
        {
            for (int k = 0; k < dims[2]; ++k)
            {
                tens_out(i, k, j) = tens_in(i, j, k);
            }
        }
    }

    return tens_out;
}


double get_tens_diff_norm(Eigen::Tensor<double, 3>& ten1, Eigen::Tensor<double, 3>& ten2, int n)
{
    double tmp1 = 0.0;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                tmp1 = tmp1 + std::pow(ten1(k, j, i) - ten2(k, j, i), 2);
            }
        }
    }
    return tmp1;
}

double get_tens_diff_norm_v1(Eigen::Tensor<double, 3>& ten1, Eigen::Tensor<double, 3>& ten2, int n1,int n2,int n3)
{
    double tmp1 = 0.0;

    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            for (int k = 0; k < n3; k++) {
                tmp1 = tmp1 + std::pow(ten1(k, j, i) - ten2(k, j, i), 2);
            }
        }
    }
    return tmp1;
}
// original row major- EXPENSIVE
// Inserting a matrix in (along 1st and 2nd dim)
// Assigning code here- along 1st and second dim, 3rd dim constant (input)
// Templated
template <typename T>
void hess_assign_fd(Eigen::Tensor<double, 3>& hess, const Eigen::MatrixBase<T>& mat, int n, int k)
{
    for (int ii = 0; ii < n; ii++) {
        for (int jj = 0; jj < n; jj++) {
            hess(ii, jj, k) = mat(ii, jj);
        }
    }
}

// CHEAP
// chaning to col major- massive saving from original
// Inserting a matrix in (along 1st and 2nd dim)
// Assigning code here- along 1st and second dim, 3rd dim constant (input)
// Templated
template <typename T>
void hess_assign_fd_v1(Eigen::Tensor<double, 3>& hess, const Eigen::MatrixBase<T>& mat, int n, int k)
{
    for (int ii = 0; ii < n; ii++) {
        for (int jj = 0; jj < n; jj++) {
            hess(jj, ii, k) = mat(jj, ii);
        }
    }
}
// CHEAP
// chaning to col major- massive saving from original
// Inserting a matrix in (along 1st and 2nd dim)
// Assigning code here- along 1st and second dim, 3rd dim constant (input)
// Templated
template <typename T>
void hess_assign_fd_v1_gen(Eigen::Tensor<double, 3>& hess, const Eigen::MatrixBase<T>& mat, int r, int c, int k)
{
    for (int ii = 0; ii < c; ii++) {
        for (int jj = 0; jj < r; jj++) {
            hess(jj, ii, k) = mat(jj, ii);
        }
    }
}

// CHEAP
// chaning to col major- massive saving from original
// Inserting a matrix in (along 1st and 2nd dim)
// Assigning code here- along 1st and second dim, 3rd dim constant (input)
// Templated
template <typename T>
void hess_assign_fd_v2(Eigen::Tensor<double, 3>& hess, const Eigen::MatrixBase<T>& mat, int n, int n1, int n2, int n3)
{
    for (int ii = 0; ii < n; ii++) {
        for (int jj = 0; jj < n; jj++) {
            hess(jj+n1, ii+n2, n3) = mat(jj, ii);
        }
    }
}


// Assigning a tensor into another bigger tensor
template <typename T>
void tens_assign_tens(Eigen::Tensor<T, 3>& tens_in, Eigen::Tensor<T, 3>& tens_out,const int n, const int n1,const int n2, const int n3)
{
    Eigen::MatrixXd mat_in(n,n);
    for (int ii = 0; ii < n; ii++) {
     tensor_to_mat(tens_in, mat_in, n,ii);
     hess_assign_fd_v2(tens_out,mat_in,n,n1,n2,n3);

    }
}




// Original one- VERY EXPENSIVE
// Inserting a matrix in (along 2nd and 3rd dim)
// Assigning code here- for Finite-Diff hessian- along 2nd and 3rd dim, 1st dim constant (input)
// Templated
template <typename T>
void hess_assign_fd1(Eigen::Tensor<double, 3>& hess, const Eigen::MatrixBase<T>& mat, int n, int k)
{
    for (int ii = 0; ii < n; ii++) {
        for (int jj = 0; jj < n; jj++) {
            hess(k, ii, jj) = mat(ii, jj);
        }
    }
}

// CHEAP- ~200 mus for 500 link- CAN BE USED
// Inserting a matrix in (along 1st and 3rd dim)
// Assigning code here- for Finite-Diff hessian- along 1st and 3rd dim, 2nd dim constant (input)
// Templated
template <typename T>
void hess_assign_fd2(Eigen::Tensor<double, 3>& hess, const Eigen::MatrixBase<T>& mat, int n, int k)
{
    for (int ii = 0; ii < n; ii++) {
        for (int jj = 0; jj < n; jj++) {
            hess(jj, k, ii) = mat(jj, ii);
        }
    }
}

// Inserting a matrix in (along 1st and 3rd dim)
// Assigning code here- for Finite-Diff hessian- along 1st and 3rd dim, 2nd dim constant (input)
// same as hess_assign_fd2, but starting from n0 instead
// Templated
template <typename T>
void hess_assign_fd2_v2(Eigen::Tensor<double, 3>& hess, const Eigen::MatrixBase<T>& mat, int n0, int n, int k)
{
    for (int ii = n0; ii < n; ii++) {
        for (int jj = n0; jj < n; jj++) {
            hess(jj, k, ii) = mat(jj, ii);
        }
    }
}

// Inserting a matrix in (along 1st and 3rd dim)
// Assigning code here- for Finite-Diff hessian- along 1st and 3rd dim, 2nd dim constant (input)
// same as hess_assign_fd2, but starting from i and j for different indices, and ending at different indices
// Templated
template <typename T>
void hess_assign_fd2_v3(
    Eigen::Tensor<double, 3>& hess, const Eigen::MatrixBase<T>& mat, int i1, int i2, int j1, int j2, int k)
{
    for (int ii = i1; ii < i2; ii++) {
        for (int jj = j1; jj < j2; jj++) {
            hess(jj, k, ii) = mat(jj, ii);
        }
    }
}

// Inserting a matrix in (along 1st and 3rd dim)- but in a flatted tensor
// Assigning code here-  along 1st and 3rd dim, 2nd dim constant (input)
// same as hess_assign_fd2, but starting from i and j for different indices, and ending at different indices
// Templated
template <typename MatrixType1, typename MatrixType2>
void flattens_assign_fd2(MatrixType1& mat2, const MatrixType2& mat1, int n, int k)
{
    for (int ii = 0; ii < n; ii++) {
        for (int jj = 0; jj < n; jj++) {
            mat2(jj, k + ii * n) = mat1(jj, ii);
        }
    }
}

// changing order here- cuts down the cost in half
// STILL EXPENSIVE
// Inserting a matrix in (along 2nd and 3rd dim)
// Assigning code here- for Finite-Diff hessian- along 2nd and 3rd dim, 1st dim constant (input)
// Templated
template <typename T>
void hess_assign_fd1_v1(Eigen::Tensor<double, 3>& hess, const Eigen::MatrixBase<T>& mat, int n, int k)
{
    for (int ii = 0; ii < n; ii++) {
        for (int jj = 0; jj < n; jj++) {
            hess(k, jj, ii) = mat(jj, ii);
        }
    }
}

// super expensive- original
// get matrix from a tensor along dimension-1 (or keeping first dim constant)
// dim of tens : nxnxn
// dim of mat : nxn
template <typename T>
void get_mat_from_tens1(const Eigen::Tensor<double, 3>& tens, Eigen::MatrixBase<T>& mat, int n, int k)
{
    for (int ii = 0; ii < n; ii++) {
        for (int jj = 0; jj < n; jj++) {
            mat(ii, jj) = tens(k, jj, ii);
        }
    }
}

// changing order here- a little cheaper/expensive than tens1- changed mat access to col major
// STILL EXPENSIVE
// get matrix from a tensor along dimension-1 (or keeping first dim constant)
// dim of tens : nxnxn
// dim of mat : nxn
template <typename T>
void get_mat_from_tens1_v1(const Eigen::Tensor<double, 3>& tens, Eigen::MatrixBase<T>& mat, int n, int k)
{
    for (int ii = 0; ii < n; ii++) {
        for (int jj = 0; jj < n; jj++) {
            mat(jj, ii) = tens(k, ii, jj);
        }
    }
}

// NOT WORKING- AND ALSO DOESN'T HELP in RUNTIMES

// // using the slice function here
// // get matrix from a tensor along dimension-1 (or keeping first dim constant)
// // dim of tens : nxnxn
// // dim of mat : nxn
// void get_mat_from_tens1_v2(const Eigen::Tensor<double,3> &tens, Eigen::Tensor<double,2> &mat_tens, int n, int k)
// {
//   Eigen::array<int,3> off = {0,0,0};
//   Eigen::array<int,3> ext = {k,n,n};
//   Eigen::array<int,2> shape = {n,n};
//   mat_tens = tens.slice(off,ext).reshape(shape) ;

//   cout << "mat_tens = \n" << mat_tens << endl;
// }

// get matrix from a tensor along dimension-1 (or keeping first dim constant), and second dim 1:k
// dim of tens : nxnxn
// dim of mat : nxn
template <typename T>
void get_mat_from_tens1_sub(const Eigen::Tensor<double, 3>& tens, Eigen::MatrixBase<T>& mat, int n, int k)
{
    for (int ii = 0; ii < n; ii++) {
        for (int jj = 0; jj < k; jj++) // changed n to k here
        {
            mat(ii, jj) = tens(k, jj, ii);
        }
    }
}

// already in col major- CHEAP
// get matrix from a tensor along dimension-2 (or keeping second dim constant)
// dim of tens : nxnxn
// dim of mat : nxn
template <typename T>
void get_mat_from_tens2(const Eigen::Tensor<double, 3>& tens, Eigen::MatrixBase<T>& mat, int n, int k)
{
    for (int ii = 0; ii < n; ii++) {
        for (int jj = 0; jj < n; jj++) {
            mat(jj, ii) = tens(jj, k, ii);
        }
    }
}

// usual one- EXPENSIVE (not used anymore)
// get matrix from a tensor along dimension-3 (or keeping third dim constant)
template <typename T>
void get_mat_from_tens3(const Eigen::Tensor<double, 3>& tens, Eigen::MatrixBase<T>& mat, int n, int k)
{
    for (int ii = 0; ii < n; ii++) {
        for (int jj = 0; jj < n; jj++) {
            mat(ii, jj) = tens(ii, jj, k);
        }
    }
}

// This is 30 x faster than row-major one
// changing the tens access and matrix access to col-major
// get matrix from a tensor along dimension-3 (or keeping third dim constant)
template <typename T>
void get_mat_from_tens3_v1(const Eigen::Tensor<double, 3>& tens, Eigen::MatrixBase<T>& mat, int n, int k)
{
    for (int ii = 0; ii < n; ii++) {
        for (int jj = 0; jj < n; jj++) {
            mat(jj, ii) = tens(jj, ii, k);
        }
    }
}

// This is 30 x faster than row-major one
// changing the tens access and matrix access to col-major
// get matrix from a tensor along dimension-3 (or keeping third dim constant)
template <typename T>
void get_mat_from_tens3_v1_gen(const Eigen::Tensor<double, 3>& tens, Eigen::MatrixBase<T>& mat, int r, int c, int k)
{
    for (int ii = 0; ii < c; ii++) {
        for (int jj = 0; jj < r; jj++) {
            mat(jj, ii) = tens(jj, ii, k);
        }
    }
}

template <typename T>
void get_vec_from_tens3_v1_gen(const Eigen::Tensor<double, 3>& tens, Eigen::MatrixBase<T>& vec, int r, int j, int k)
{
    for (int ii = 0; ii < r; ii++) {
        vec(ii) = tens(ii, j, k);
    }
}


// This is 30 x faster than row-major one
// changing the tens access and matrix access to col-major
// get matrix from a tensor along dimension-3 (or keeping third dim constant)
// more templated
template <typename Scalar, typename Matrixtype>
void get_mat_from_tens3_v2(const Eigen::Tensor<Scalar, 3>& tens, Matrixtype& mat, int n, int k)
{
    for (int ii = 0; ii < n; ii++) {
        for (int jj = 0; jj < n; jj++) {
            mat(jj, ii) = tens(jj, ii, k);
        }
    }
}

// This is 30 x faster than row-major one
// more templated
// taking one nxn matrix out of flat tens (nxnxn)- flatted as nxn^2
template <typename Matrixtype1, typename Matrixtype2>
void get_mat_from_flattens3_v2(const Matrixtype1& flat_tens, Matrixtype2& mat, int n, int k)
{
    for (int ii = 0; ii < n; ii++) {
        for (int jj = 0; jj < n; jj++) {
            mat(jj, ii) = flat_tens(jj, ii + n * k);
        }
    }
}

template <typename MatrixType1, typename MatrixType2>
void get_col(const MatrixType1& mat, MatrixType2& vec, int n, int u)
{
    for (int i = 0; i < n; i++) {
        vec(i, 0) = mat(i, u);
    }
}

// already in col major- CHEAP
// get matrix from a tensor along dimension-2 (or keeping second dim constant)
// dim of tens : nxnxn
// dim of mat : nxn
template <typename MatrixType1, typename MatrixType2>
void get_mat_from_flat_tens2(const MatrixType1& mat1, MatrixType2& mat2, int n, int k)
{
    for (int ii = 0; ii < n; ii++) {
        for (int jj = 0; jj < n; jj++) {
            mat2(jj, ii) = mat1(jj, k + n * ii);
        }
    }
}

// Original
// VERY EXPENSIVE
// This method rotates a tensor along the 3rd dim, so that it's transposed along the 3rd dim
// dim of tens, rot_tens : nxnxn
// dim of mat : nxn
void tens_rot(const Eigen::Tensor<double, 3>& tens, Eigen::Tensor<double, 3>& rot_tens, int n)
{
    Eigen::MatrixXd temp_mat(n, n);
    for (int ii = 0; ii < n; ii++) {
        get_mat_from_tens1(tens, temp_mat, n, ii);
        hess_assign_fd(rot_tens, temp_mat, n, ii);
    }
}

// With col major- cuts down cost  in half
// VERY EXPENSIVE
// This method rotates a tensor along the 3rd dim, so that it's transposed along the 3rd dim
// dim of tens, rot_tens : nxnxn
// dim of mat : nxn
void tens_rot_v1(const Eigen::Tensor<double, 3>& tens, Eigen::Tensor<double, 3>& rot_tens, int n)
{
    Eigen::MatrixXd temp_mat(n, n);
    for (int ii = 0; ii < n; ii++) {
        get_mat_from_tens1_v1(tens, temp_mat, n, ii);
        hess_assign_fd_v1(rot_tens, temp_mat, n, ii);
    }
}


template <typename T>
void slice_in_v6(Eigen::Tensor<double, 3>& tens, const Eigen::MatrixBase<T>& vec, const int q, const int r)
{
   for (int ii = 0; ii < 6; ii++) {
        tens(ii , q, r) = vec(ii);
    }
}

// Assigning code here- for assigning a row or a column of the tensor- for Eigen::VectorXd
template <typename T>
void hess_assign(Eigen::Tensor<double, 3>& hess, const Eigen::MatrixBase<T>& vec, const int p, const int q, const int r,
    const int index, const int stride)
{
    if (index == 1) {
        for (int ii = 0; ii < stride; ii++) {
            hess(ii + p, q, r) = vec(ii);
        }

    } else if (index == 2) {
        for (int ii = 0; ii < stride; ii++) {
            hess(p, ii + q, r) = vec(ii);
        }

    } else if (index == 3) {
        for (int ii = 0; ii < stride; ii++) {
            hess(p, q, ii + r) = vec(ii);
        }
    } else {
        cout << "No index for hessian_assign provided!" << endl;
    }
}

template <typename T>
void tens_assign6_col(Eigen::Tensor<double,3> & hess, const Eigen::MatrixBase<T>& vec, const int p, const int q)
{

    // Directly assign each element (assuming vec.size() == 6)
    hess(0, p, q) = vec(0);
    hess(1, p, q) = vec(1);
    hess(2, p, q) = vec(2);
    hess(3, p, q) = vec(3);
    hess(4, p, q) = vec(4);
    hess(5, p, q) = vec(5);
}

// Assigning code here- for assigning a row or a column of the tensor- for Eigen::VectorXd
// nxnxn tensor flatted into nxn^2 matrix
// only accepts stride along the 1st index
template <typename MatrixType1, typename MatrixType2>
void hess_assign_flat(
    MatrixType1& mat, const MatrixType2& vec, const int p, const int q, const int r, const int index, const int stride)
{
    if (index == 1) {
        for (int ii = 0; ii < stride; ii++) {
            mat(ii + p, q + stride * r) = vec(ii, 0);
        }

    } else {
        cout << "No index for hessian_assign provided!" << endl;
    }
}

// Assigning code here- for assigning an entire matrix into another matrix- emulating the middleCols method
template <typename MatrixType1, typename MatrixType2>
void mat_middle_cols(MatrixType1& mat1, const MatrixType2& mat2, const int nrow, const int ncol, const int ncol1)
{
    for (int ii = 0; ii < ncol; ii++) {
        for (int jj = 0; jj < nrow; jj++) {
            mat1(jj, ii + ncol1) = mat2(jj, ii);
        }
    }
}
// Assigning code here- for assigning a row or a column of the tensor- for Eigen::VectorXd
template <typename T>
void hess_assign(Eigen::Tensor<double, 3>& hess, const std::vector<double>& vec, const int p, const int q, const int r,
    const int index, const int stride)
{
    if (index == 1) {
        for (int ii = 0; ii < stride; ii++) {
            hess(ii + p, q, r) = vec[ii];
        }

    } else if (index == 2) {
        for (int ii = 0; ii < stride; ii++) {
            hess(p, ii + q, r) = vec[ii];
        }

    } else if (index == 3) {
        for (int ii = 0; ii < stride; ii++) {
            hess(p, q, ii + r) = vec[ii];
        }
    } else {
        cout << "No index for hessian_assign provided!" << endl;
    }
}

// To get a row or column vector of a tensor
template <typename T>
void hess_get(const Eigen::Tensor<double, 3>& hess, Eigen::MatrixBase<T>& vec, int p, const int q, const int r,
    const int index, const int stride)
{
    if (index == 1) {
        for (int ii = 0; ii < stride; ii++) {
            vec(ii) = hess(ii + p, q, r);
        }

    } else if (index == 2) {
        for (int ii = 0; ii < stride; ii++) {
            vec(ii) = hess(p, ii + q, r);
        }

    } else if (index == 3) {
        for (int ii = 0; ii < stride; ii++) {
            vec(ii) = hess(p, q, ii + r);
        }
    } else {
        cout << "No index for hessian_get provided!" << endl;
    }
}

// To assign a matrix to a tensor, (k,1:k,:) and (1:k,k,:)
template <typename T>
static void hess_assign_mat1(Eigen::Tensor<double, 3>& hess, const Eigen::MatrixBase<T>& mat, int n, int k)
{
    for (int ii = 0; ii < k; ii++) {
        for (int jj = 0; jj < n; jj++) {
            hess(k, ii, jj) = mat(ii, jj);
            hess(ii, k, jj) = mat(ii, jj);
        }
    }
}

// function used to take the product of a matrix M, tensor T (m*T*m), along first and second dims of T
void mat_ten_mat(Eigen::Tensor<double, 3>& ten_in, Eigen::Tensor<double, 3>& ten_out, Eigen::MatrixXd& mat, int n)
{

    Eigen::MatrixXd temp1(n, n);
    Eigen::MatrixXd temp2(n, n);

    for (int ii = 0; ii < n; ii++) {
        get_mat_from_tens3(ten_in, temp1, n, ii);
        temp2 = mat * temp1 * mat;
        hess_assign_fd(ten_out, temp2, n, ii);
    }
}

// Funtion for assigning a matrix/vector/scalar in a 3D tensor
// Integer index gives the index which is constant
template <typename T>
void tens_assign(Eigen::Tensor<double, 3>& tens, const Eigen::MatrixBase<T>& mat, const int index, const int x0,
    const int y0, const int z0, const int x_stride, const int y_stride, const int z_stride)
{
    if (index == 1) // no stride in 1st dimension
    {
        for (int ii = z0; ii < z_stride; ii++) {
            for (int jj = y0; jj < y_stride; jj++) {
                tens(x0, jj, ii) = mat(jj, ii);
            }
        }
    } else if (index == 2) // no stride in 2nd dimension
    {
        for (int ii = z0; ii < z_stride; ii++) {
            for (int jj = x0; jj < x_stride; jj++) {
                tens(jj, y0, ii) = mat(jj, ii);
            }
        }
    } else if (index == 3) // no stride in 3rd dimension
    {
        for (int ii = y0; ii < y_stride; ii++) {
            for (int jj = x0; jj < x_stride; jj++) {
                tens(jj, ii, z0) = mat(jj, ii);
            }
        }
    } else {
        cout << "No index for hessian_assign provided!" << endl;
    }
}
}
#endif //
