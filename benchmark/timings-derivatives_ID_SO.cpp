//
// Copyright (c) 2018-2020 CNRS INRIA
// Contents- Only for Accuracy test of RNEA_SO_v2 with ID SO finite-diff
// created on 4/8/22 - adding the ID SO triple loop algo
// 1. FO RNEA derivatives faster
// 2. SO partial derivatives of ID using RNEA finite-diff
// 3. SO partial derivatives of ID using RNEA_v2
// 4. SO partials of ID using IDSVA triple loop algo RNEA_v7
// 5. SO partials of ID using IDSVA triple-loop, DoF-i, full j,k joints at once
// Modified on 5/16/22
// Added-  RNEA_SO_v8- with only DoF of i, but full j,k collectively

#include "pinocchio/algorithm/cholesky.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/rnea-derivatives-faster.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/rnea_SO_derivatives.hpp"
#include "pinocchio/container/aligned-vector.hpp"
#include "pinocchio/parsers/sample-models.hpp"
#include "pinocchio/parsers/urdf.hpp"

#include <iostream>

#include "pinocchio/utils/tensor_utils.hpp"
#include "pinocchio/utils/timer.hpp"

#include <typeinfo>

using std::cout;
using std::endl;

template <typename Matrix1, typename Matrix2, typename Matrix3>
void rnea_fd(const pinocchio::Model &model, pinocchio::Data &data_fd,
             const Eigen::VectorXd &q, const Eigen::VectorXd &v,
             const Eigen::VectorXd &a,
             const Eigen::MatrixBase<Matrix1> &_drnea_dq,
             const Eigen::MatrixBase<Matrix2> &_drnea_dv,
             const Eigen::MatrixBase<Matrix3> &_drnea_da) {
  Matrix1 &drnea_dq = PINOCCHIO_EIGEN_CONST_CAST(Matrix1, _drnea_dq);
  Matrix2 &drnea_dv = PINOCCHIO_EIGEN_CONST_CAST(Matrix2, _drnea_dv);
  Matrix3 &drnea_da = PINOCCHIO_EIGEN_CONST_CAST(Matrix3, _drnea_da);

  using namespace Eigen;
  VectorXd v_eps(VectorXd::Zero(model.nv));
  VectorXd q_plus(model.nq);
  VectorXd tau_plus(model.nv);
  const double alpha = 1e-8;

  VectorXd tau0 = rnea(model, data_fd, q, v, a);

  // dRNEA/dq

  for (int k = 0; k < model.nv; ++k) {
    v_eps[k] += alpha;
    q_plus = integrate(model, q, v_eps);
    tau_plus = rnea(model, data_fd, q_plus, v, a);
    drnea_dq.col(k) = (tau_plus - tau0) / alpha;
    v_eps[k] -= alpha;
  }

  // dRNEA/dv
  VectorXd v_plus(v);
  for (int k = 0; k < model.nv; ++k) {
    v_plus[k] += alpha;
    tau_plus = rnea(model, data_fd, q, v_plus, a);

    drnea_dv.col(k) = (tau_plus - tau0) / alpha;
    v_plus[k] -= alpha;
  }

  // dRNEA/da
  drnea_da = crba(model, data_fd, q);
  drnea_da.template triangularView<Eigen::StrictlyLower>() =
      drnea_da.transpose().template triangularView<Eigen::StrictlyLower>();
}

int main(int argc, const char **argv) {
  using namespace Eigen;
  using namespace pinocchio;

  PinocchioTicToc timer(PinocchioTicToc::US);
#ifdef NDEBUG
  const int NBT = 1;
#else
  const int NBT = 1;
  std::cout << "(the time score in debug mode is not relevant) " << std::endl;
#endif

  Model model;
  std::string model_name;
  std::cout << "Enter the model name " << endl;
  std::cin >> model_name;
  std::string filename =
      "/home/shubham/Desktop/Pinocchio_SO_v2/pinocchio-master/models" +
      std::string("/") + model_name + std::string(".urdf");

  if (argc > 1)
    filename = argv[1];
  bool with_ff;
  std::cout << "Enter if with floating base or not" << std::endl;
  std::cin >> with_ff;

  if (argc > 2) {
    const std::string ff_option = argv[2];
    if (ff_option == "-no-ff")
      with_ff = false;
  }

  if (with_ff)
    pinocchio::urdf::buildModel(filename, JointModelFreeFlyer(), model);
  //      pinocchio::urdf::buildModel(filename,JointModelRX(),model);
  else
    pinocchio::urdf::buildModel(filename, model);
  std::cout << "nq = " << model.nq << std::endl;
  std::cout << "nv = " << model.nv << std::endl;

  Data data(model);
  VectorXd qmax = Eigen::VectorXd::Ones(model.nq);

  PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) qs(NBT);
  PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) qdots(NBT);
  PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) qddots(NBT);
  PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) taus(NBT);
  PINOCCHIO_ALIGNED_STD_VECTOR(MatrixXd) tau_mat(NBT); // added a new variable

  for (size_t i = 0; i < NBT; ++i) {
    qs[i] = randomConfiguration(model, -qmax, qmax);
    // qdots[i]  = Eigen::VectorXd::Random(model.nv);
    qdots[i] = Eigen::VectorXd::Random(model.nv);
    //  qddots[i] = Eigen::VectorXd::Random(model.nv);
    qddots[i] = Eigen::VectorXd::Random(model.nv);
    taus[i] = Eigen::VectorXd::Random(model.nv);
    tau_mat[i] = Eigen::MatrixXd::Zero(model.nv, 2 * model.nv); // new variable
  }

  PINOCCHIO_EIGEN_PLAIN_ROW_MAJOR_TYPE(MatrixXd)
  drnea_dq(MatrixXd::Zero(model.nv, model.nv));
  PINOCCHIO_EIGEN_PLAIN_ROW_MAJOR_TYPE(MatrixXd)
  drnea_dv(MatrixXd::Zero(model.nv, model.nv));
  MatrixXd drnea_da(MatrixXd::Zero(model.nv, model.nv));

  cout << "RNEA----" << endl;
  SMOOTH(NBT) {
    rnea(model, data, qs[_smooth], qdots[_smooth], qddots[_smooth]);
  }

  taus[0] = data.tau;
  // cout << "tau = " << taus[0] << endl;

  cout << "RNEA derivatives Faster----" << endl;
  SMOOTH(NBT) {
    computeRNEADerivativesFaster(model, data, qs[_smooth], qdots[_smooth],
                                 qddots[_smooth], drnea_dq, drnea_dv, drnea_da);
  }

  //  std::cout << "drnea_dq = " << drnea_dq << endl;
  //  std::cout << "drnea_dv = " << drnea_dv << endl;

  // SO partials of tau using algo
  // SO partials variables here for the analytical algo

  //---------------------------------------------------//
  // SO Finite difference -----------------------------//
  //---------------------------------------------------//

  //  // perturbed variables here

  PINOCCHIO_EIGEN_PLAIN_ROW_MAJOR_TYPE(MatrixXd)
  drnea_dq_plus(MatrixXd::Zero(model.nv, model.nv));
  PINOCCHIO_EIGEN_PLAIN_ROW_MAJOR_TYPE(MatrixXd)
  drnea_dv_plus(MatrixXd::Zero(model.nv, model.nv));
  MatrixXd drnea_da_plus(MatrixXd::Zero(model.nv, model.nv));

  // SO partials variables here

  Eigen::Tensor<double, 3> dtau2_dq(model.nv, model.nv, model.nv);
  Eigen::Tensor<double, 3> dtau2_dqd(model.nv, model.nv, model.nv);
  Eigen::Tensor<double, 3> dtau2_MSO(model.nv, model.nv, model.nv);
  dtau2_dq.setZero();
  dtau2_dqd.setZero();
  dtau2_MSO.setZero();

  VectorXd v_eps(VectorXd::Zero(model.nv));
  VectorXd q_plus(model.nq);
  VectorXd qd_plus(model.nv);

  MatrixXd temp_mat1(MatrixXd::Zero(model.nv, model.nv));
  MatrixXd temp_mat2(MatrixXd::Zero(model.nv, model.nv));
  // difference variables

  Eigen::Tensor<double, 3> temp1_fd(model.nv, model.nv, model.nv);
  Eigen::Tensor<double, 3> temp2_fd(model.nv, model.nv, model.nv);
  Eigen::Tensor<double, 3> temp3_fd(model.nv, model.nv, model.nv);

  double alpha = 1e-7; // performs well

  // Partial wrt q
  SMOOTH(NBT)
  for (int k = 0; k < model.nv; ++k) {
    v_eps[k] += alpha;
    q_plus = integrate(
        model, qs[_smooth],
        v_eps); // This is used to add the v_eps to q in the k^th direction
    computeRNEADerivativesFaster(model, data, q_plus, qdots[_smooth],
                                 qddots[_smooth], drnea_dq_plus, drnea_dv_plus,
                                 drnea_da_plus);
    temp_mat1 = (drnea_dq_plus - drnea_dq) / alpha;
    hess_assign_fd(dtau2_dq, temp_mat1, model.nv, k);
    v_eps[k] -= alpha;
  }

  // Partial wrt qd
  SMOOTH(NBT)
  for (int k = 0; k < model.nv; ++k) {
    v_eps[k] += alpha;
    qd_plus = qdots[_smooth] +
              v_eps; // This is used to add the v_eps to q in the k^th direction
    computeRNEADerivativesFaster(model, data, qs[_smooth], qd_plus,
                                 qddots[_smooth], drnea_dq_plus, drnea_dv_plus,
                                 drnea_da_plus);
    temp_mat1 = (drnea_dv_plus - drnea_dv) / alpha; // SO partial wrt qdot
    temp_mat2 =
        (drnea_dq_plus - drnea_dq) / alpha; // MSO partial of dtau_dq wrt qdot
    hess_assign_fd(dtau2_dqd, temp_mat1, model.nv, k);
    hess_assign_fd(dtau2_MSO, temp_mat2, model.nv, k);
    v_eps[k] -= alpha;
  }
  // std::cout << "dtau2_dq = "<< dtau2_dq << endl;

  // ----------------------------------------------------
  // ID SO v7 Analytical Partials ------------------------
  // ----------------------------------------------------

  Eigen::Tensor<double, 3> dtau2_dq_ana_v2(model.nv, model.nv, model.nv);
  Eigen::Tensor<double, 3> dtau2_dv_ana_v2(model.nv, model.nv, model.nv);
  Eigen::Tensor<double, 3> dtau2_dqv_ana_v2(model.nv, model.nv, model.nv);
  Eigen::Tensor<double, 3> M_FO_v2(model.nv, model.nv, model.nv);

  M_FO_v2.setZero();

  // difference variables

  Eigen::Tensor<double, 3> temp1(model.nv, model.nv, model.nv);
  Eigen::Tensor<double, 3> temp2(model.nv, model.nv, model.nv);
  Eigen::Tensor<double, 3> temp3(model.nv, model.nv, model.nv);
  Eigen::Tensor<double, 3> temp4(model.nv, model.nv, model.nv);

  SMOOTH(NBT) {
    computeRNEA_SO_derivs(model, data, qs[_smooth], qdots[_smooth],
                          qddots[_smooth], dtau2_dq_ana_v2, dtau2_dv_ana_v2,
                          dtau2_dqv_ana_v2, M_FO_v2);
  }

  // Difference

  temp1 = dtau2_dq - dtau2_dq_ana_v2;
  temp2 = dtau2_dqd - dtau2_dv_ana_v2;
  temp3 = dtau2_MSO - dtau2_dqv_ana_v2;
  // temp4 = M_FO_v1 - M_FO_v2;

  std::cout << "---------------------------------------------------"
            << std::endl;
  std::cout << "Difference in the SO partial v7 w.r.t q is"
            << (temp1.abs()).maximum() << std::endl;
  std::cout << "Difference in the SO partial v7 w.r.t qd is"
            << (temp2.abs()).maximum() << std::endl;
  std::cout << "Difference in the MSO partial v7 w.r.t q and qd is"
            << (temp3.abs()).maximum() << std::endl;
  std::cout << "Difference in the M FO v7 w.r.t q is" << (temp4.abs()).maximum()
            << std::endl;

  return 0;
}
