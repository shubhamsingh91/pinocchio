//
// Copyright (c) 2018-2020 CNRS INRIA
//

#include "pinocchio/algorithm/aba-derivatives.hpp"
#include "pinocchio/algorithm/aba.hpp"
#include "pinocchio/algorithm/aba_v2.hpp"
#include "pinocchio/algorithm/cholesky.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics-derivatives.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/rnea-derivatives-faster.hpp"
#include "pinocchio/algorithm/rnea-derivatives.hpp"
<<<<<<< HEAD
#include "pinocchio/algorithm/rnea-derivatives-SO.hpp"
#include "pinocchio/algorithm/aba-derivatives.hpp"
#include "pinocchio/algorithm/aba.hpp"
=======
>>>>>>> master
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/rnea_SO_derivatives.hpp"
#include "pinocchio/container/aligned-vector.hpp"
#include "pinocchio/parsers/sample-models.hpp"
#include "pinocchio/parsers/urdf.hpp"

#include <iostream>

#include "pinocchio/utils/timer.hpp"

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
<<<<<<< HEAD
    integrate(model,q,v_eps,q_plus);
    tau_plus = rnea(model,data_fd,q_plus,v,a);
    
    drnea_dq.col(k) = (tau_plus - tau0)/alpha;
=======
    q_plus = integrate(model, q, v_eps);
    tau_plus = rnea(model, data_fd, q_plus, v, a);

    drnea_dq.col(k) = (tau_plus - tau0) / alpha;
>>>>>>> master
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

void aba_fd(const pinocchio::Model &model, pinocchio::Data &data_fd,
            const Eigen::VectorXd &q, const Eigen::VectorXd &v,
            const Eigen::VectorXd &tau, Eigen::MatrixXd &daba_dq,
            Eigen::MatrixXd &daba_dv, pinocchio::Data::RowMatrixXs &daba_dtau) {
  using namespace Eigen;
  VectorXd v_eps(VectorXd::Zero(model.nv));
  VectorXd q_plus(model.nq);
  VectorXd a_plus(model.nv);
  const double alpha = 1e-8;

  VectorXd a0 = aba(model, data_fd, q, v, tau);

  // dABA/dq
  for (int k = 0; k < model.nv; ++k) {
    v_eps[k] += alpha;
<<<<<<< HEAD
    integrate(model,q,v_eps,q_plus);
    a_plus = aba(model,data_fd,q_plus,v,tau);
    
    daba_dq.col(k) = (a_plus - a0)/alpha;
=======
    q_plus = integrate(model, q, v_eps);
    a_plus = aba(model, data_fd, q_plus, v, tau);

    daba_dq.col(k) = (a_plus - a0) / alpha;
>>>>>>> master
    v_eps[k] -= alpha;
  }

  // dABA/dv
  VectorXd v_plus(v);
  for (int k = 0; k < model.nv; ++k) {
    v_plus[k] += alpha;
    a_plus = aba(model, data_fd, q, v_plus, tau);

    daba_dv.col(k) = (a_plus - a0) / alpha;
    v_plus[k] -= alpha;
  }

  // dABA/dtau
  daba_dtau = computeMinverse(model, data_fd, q);
}

int main(int argc, const char **argv) {
  using namespace Eigen;
  using namespace pinocchio;

  PinocchioTicToc timer(PinocchioTicToc::US);
<<<<<<< HEAD
  #ifdef NDEBUG
  const int NBT = 1000*10;
  #else
    const int NBT = 1;
    std::cout << "(the time score in debug mode is not relevant) " << std::endl;
  #endif
    
  Model model;

  //std::string filename = PINOCCHIO_MODEL_DIR + std::string("/simple_humanoid.urdf");
  std::string filename =  std::string("../models/simple_humanoid.urdf");

  if(argc>1) filename = argv[1];
=======
#ifdef NDEBUG
  const int NBT = 1000 * 10;

#else
  const int NBT = 1;
  std::cout << "(the time score in debug mode is not relevant) " << std::endl;
#endif

  Model model;

  std::string filename = "../models" + std::string("/simple_humanoid.urdf");
  if (argc > 1)
    filename = argv[1];
>>>>>>> master
  bool with_ff = true;

  if (argc > 2) {
    const std::string ff_option = argv[2];
    if (ff_option == "-no-ff")
      with_ff = false;
  }

  if (filename == "HS")
    buildModels::humanoidRandom(model, true);
  else if (with_ff)
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
    qdots[i] = Eigen::VectorXd::Random(model.nv);
    qddots[i] = Eigen::VectorXd::Random(model.nv);
    taus[i] = Eigen::VectorXd::Random(model.nv);
    tau_mat[i] = Eigen::MatrixXd::Zero(model.nv, 2 * model.nv); // new variable
  }
<<<<<<< HEAD
  
  PINOCCHIO_EIGEN_PLAIN_ROW_MAJOR_TYPE(MatrixXd) drnea_dq(MatrixXd::Zero(model.nv,model.nv));
  PINOCCHIO_EIGEN_PLAIN_ROW_MAJOR_TYPE(MatrixXd) drnea_dv(MatrixXd::Zero(model.nv,model.nv));
  MatrixXd drnea_da(MatrixXd::Zero(model.nv,model.nv));
 
  MatrixXd daba_dq(MatrixXd::Zero(model.nv,model.nv));
  MatrixXd daba_dv(MatrixXd::Zero(model.nv,model.nv));
  Data::RowMatrixXs daba_dtau(Data::RowMatrixXs::Zero(model.nv,model.nv));

  Data::Tensor3x dtau2_dq(model.nv, model.nv, model.nv);
  Data::Tensor3x dtau2_dv(model.nv, model.nv, model.nv);
  Data::Tensor3x dtau2_dqv(model.nv, model.nv, model.nv);
  Data::Tensor3x dtau_dadq(model.nv, model.nv, model.nv);
  dtau2_dq.setZero();
  dtau2_dv.setZero();
  dtau2_dqv.setZero();
  dtau_dadq.setZero();
  
=======

  PINOCCHIO_EIGEN_PLAIN_ROW_MAJOR_TYPE(MatrixXd)
  drnea_dq(MatrixXd::Zero(model.nv, model.nv));
  PINOCCHIO_EIGEN_PLAIN_ROW_MAJOR_TYPE(MatrixXd)
  drnea_dv(MatrixXd::Zero(model.nv, model.nv));
  MatrixXd drnea_da(MatrixXd::Zero(model.nv, model.nv));

  MatrixXd daba_dq(MatrixXd::Zero(model.nv, model.nv));
  MatrixXd daba_dv(MatrixXd::Zero(model.nv, model.nv));
  Data::RowMatrixXs daba_dtau(Data::RowMatrixXs::Zero(model.nv, model.nv));

  // RNEA SO variables

  Eigen::Tensor<double, 3> dtau2_dq(model.nv, model.nv, model.nv);
  Eigen::Tensor<double, 3> dtau2_dv(model.nv, model.nv, model.nv);
  Eigen::Tensor<double, 3> dtau2_dqv(model.nv, model.nv, model.nv);
  Eigen::Tensor<double, 3> M_FO(model.nv, model.nv, model.nv);

>>>>>>> master
  timer.tic();
  SMOOTH(NBT) {
    forwardKinematics(model, data, qs[_smooth], qdots[_smooth],
                      qddots[_smooth]);
  }
  std::cout << "FK= \t\t\t\t";
  timer.toc(std::cout, NBT);

  timer.tic();
  SMOOTH(NBT) {
    computeForwardKinematicsDerivatives(model, data, qs[_smooth],
                                        qdots[_smooth], qddots[_smooth]);
  }
  std::cout << "FK derivatives= \t\t";
  timer.toc(std::cout, NBT);

  timer.tic();
  SMOOTH(NBT) {
    rnea(model, data, qs[_smooth], qdots[_smooth], qddots[_smooth]);
  }
  std::cout << "RNEA= \t\t\t\t";
  timer.toc(std::cout, NBT);

  timer.tic();
  SMOOTH(NBT) {
    computeRNEADerivatives(model, data, qs[_smooth], qdots[_smooth],
                           qddots[_smooth], drnea_dq, drnea_dv, drnea_da);
  }
  std::cout << "RNEA derivatives= \t\t";
  timer.toc(std::cout, NBT);

  timer.tic();
  SMOOTH(NBT) {
<<<<<<< HEAD
    computeRNEADerivativesSO(model, data, qs[_smooth], qdots[_smooth],
                             qddots[_smooth], dtau2_dq, dtau2_dv, dtau2_dqv,
                             dtau_dadq);
  }
  std::cout << "RNEA derivatives SO= \t\t";
  timer.toc(std::cout, NBT);

  timer.tic();
  SMOOTH(NBT/100)
  {
    rnea_fd(model,data,qs[_smooth],qdots[_smooth],qddots[_smooth],
            drnea_dq,drnea_dv,drnea_da);
=======
    computeRNEADerivativesFaster(model, data, qs[_smooth], qdots[_smooth],
                                 qddots[_smooth], drnea_dq, drnea_dv, drnea_da);
>>>>>>> master
  }
  std::cout << "RNEA derivatives v2= \t\t";
  timer.toc(std::cout, NBT);

  taus[0] = data.tau; // input tau taken from RNEA_derivs- SS

  timer.tic();
  SMOOTH(NBT) {
    computeABADerivatives(model, data, qs[_smooth], qdots[_smooth],
                          taus[_smooth], daba_dq, daba_dv, daba_dtau);
  }
  std::cout << "ABA derivatives= \t\t";
  timer.toc(std::cout, NBT);

  tau_mat[0] << -drnea_dq, -drnea_dv; // concatenating partial wrt q and qdot

  timer.tic();
  SMOOTH(NBT) { computeMinv_AZA(model, data, qs[_smooth], tau_mat[_smooth]); }
  std::cout << "Minv + AZA = \t\t\t";
  timer.toc(std::cout, NBT);

  timer.tic();
  SMOOTH(NBT) {
    computeRNEA_SO_derivs(model, data, qs[_smooth], qdots[_smooth],
                          qddots[_smooth], dtau2_dq, dtau2_dv, dtau2_dqv, M_FO);
  }
  std::cout << "RNEA SO derivatives= \t\t";
  timer.toc(std::cout, NBT);

#ifndef NO_FINITE_DIFFS
  timer.tic();
  SMOOTH(NBT / 100) {
    rnea_fd(model, data, qs[_smooth], qdots[_smooth], qddots[_smooth], drnea_dq,
            drnea_dv, drnea_da);
  }
  std::cout << "RNEA finite differences= \t";
  timer.toc(std::cout, NBT / 100);
#endif

  timer.tic();
  SMOOTH(NBT) { aba(model, data, qs[_smooth], qdots[_smooth], taus[_smooth]); }
  std::cout << "ABA= \t\t\t\t";
  timer.toc(std::cout, NBT);

#ifndef NO_FINITE_DIFFS
  timer.tic();
  SMOOTH(NBT / 100) {
    aba_fd(model, data, qs[_smooth], qdots[_smooth], taus[_smooth], daba_dq,
           daba_dv, daba_dtau);
  }
  std::cout << "ABA finite differences= \t";
  timer.toc(std::cout, NBT / 100);
#endif

  timer.tic();
  SMOOTH(NBT) { computeMinverse(model, data, qs[_smooth]); }
  std::cout << "M.inverse() from ABA = \t\t";
  timer.toc(std::cout, NBT);

  //--------

  MatrixXd Minv(model.nv, model.nv);
  Minv.setZero();
  timer.tic();
  SMOOTH(NBT) {
    crba(model, data, qs[_smooth]);
    cholesky::decompose(model, data);
    cholesky::computeMinv(model, data, Minv);
  }
  std::cout << "Minv from Cholesky = \t\t";
  timer.toc(std::cout, NBT);

  std::cout << "--" << std::endl;

  return 0;
}
