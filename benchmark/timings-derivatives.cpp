//
// Copyright (c) 2018-2020 CNRS INRIA
//

#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/kinematics-derivatives.hpp"
#include "pinocchio/algorithm/rnea-derivatives.hpp"
#include "pinocchio/algorithm/rnea-derivatives-faster.hpp"
#include "pinocchio/algorithm/aba-derivatives.hpp"
#include "pinocchio/algorithm/aba.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/cholesky.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/parsers/sample-models.hpp"
#include "pinocchio/container/aligned-vector.hpp"
#include "pinocchio/algorithm/aba_v2.hpp"

#include <iostream>

#include "pinocchio/utils/timer.hpp"

template<typename Matrix1, typename Matrix2, typename Matrix3>
void rnea_fd(const pinocchio::Model & model, pinocchio::Data & data_fd,
             const Eigen::VectorXd & q,
             const Eigen::VectorXd & v,
             const Eigen::VectorXd & a,
             const Eigen::MatrixBase<Matrix1> & _drnea_dq,
             const Eigen::MatrixBase<Matrix2> & _drnea_dv,
             const Eigen::MatrixBase<Matrix3> & _drnea_da)
{
  Matrix1 & drnea_dq = PINOCCHIO_EIGEN_CONST_CAST(Matrix1,_drnea_dq);
  Matrix2 & drnea_dv = PINOCCHIO_EIGEN_CONST_CAST(Matrix2,_drnea_dv);
  Matrix3 & drnea_da = PINOCCHIO_EIGEN_CONST_CAST(Matrix3,_drnea_da);
  
  using namespace Eigen;
  VectorXd v_eps(VectorXd::Zero(model.nv));
  VectorXd q_plus(model.nq);
  VectorXd tau_plus(model.nv);
  const double alpha = 1e-8;
  
  VectorXd tau0 = rnea(model,data_fd,q,v,a);
  
  // dRNEA/dq
  for(int k = 0; k < model.nv; ++k)
  {
    v_eps[k] += alpha;
    q_plus = integrate(model,q,v_eps);
    tau_plus = rnea(model,data_fd,q_plus,v,a);
    
    drnea_dq.col(k) = (tau_plus - tau0)/alpha;
    v_eps[k] -= alpha;
  }
  
  // dRNEA/dv
  VectorXd v_plus(v);
  for(int k = 0; k < model.nv; ++k)
  {
    v_plus[k] += alpha;
    tau_plus = rnea(model,data_fd,q,v_plus,a);
    
    drnea_dv.col(k) = (tau_plus - tau0)/alpha;
    v_plus[k] -= alpha;
  }
  
  // dRNEA/da
  drnea_da = crba(model,data_fd,q);
  drnea_da.template triangularView<Eigen::StrictlyLower>()
  = drnea_da.transpose().template triangularView<Eigen::StrictlyLower>();
  
}

void aba_fd(const pinocchio::Model & model, pinocchio::Data & data_fd,
            const Eigen::VectorXd & q,
            const Eigen::VectorXd & v,
            const Eigen::VectorXd & tau,
            Eigen::MatrixXd & daba_dq,
            Eigen::MatrixXd & daba_dv,
            pinocchio::Data::RowMatrixXs & daba_dtau)
{
  using namespace Eigen;
  VectorXd v_eps(VectorXd::Zero(model.nv));
  VectorXd q_plus(model.nq);
  VectorXd a_plus(model.nv);
  const double alpha = 1e-8;
  
  VectorXd a0 = aba(model,data_fd,q,v,tau);
  
  // dABA/dq
  for(int k = 0; k < model.nv; ++k)
  {
    v_eps[k] += alpha;
    q_plus = integrate(model,q,v_eps);
    a_plus = aba(model,data_fd,q_plus,v,tau);
    
    daba_dq.col(k) = (a_plus - a0)/alpha;
    v_eps[k] -= alpha;
  }
  
  // dABA/dv
  VectorXd v_plus(v);
  for(int k = 0; k < model.nv; ++k)
  {
    v_plus[k] += alpha;
    a_plus = aba(model,data_fd,q,v_plus,tau);
    
    daba_dv.col(k) = (a_plus - a0)/alpha;
    v_plus[k] -= alpha;
  }
  
  // dABA/dtau
  daba_dtau = computeMinverse(model,data_fd,q);
}

int main(int argc, const char ** argv)
{
  using namespace Eigen;
  using namespace pinocchio;

  PinocchioTicToc timer(PinocchioTicToc::US);
  #ifdef NDEBUG
  const int NBT = 1000*100;
  #else
    const int NBT = 1;
    std::cout << "(the time score in debug mode is not relevant) " << std::endl;
  #endif
    
  Model model;

  std::string filename = "/home/ss86299/Desktop/pinocchio/models" + std::string("/simple_humanoid.urdf");
  if(argc>1) filename = argv[1];
  bool with_ff = true;
  
  if(argc>2)
  {
    const std::string ff_option = argv[2];
    if(ff_option == "-no-ff")
      with_ff = false;
  }
    
  if( filename == "HS") 
    buildModels::humanoidRandom(model,true);
  else
    if(with_ff)
      pinocchio::urdf::buildModel(filename,JointModelFreeFlyer(),model);
//      pinocchio::urdf::buildModel(filename,JointModelRX(),model);
    else
      pinocchio::urdf::buildModel(filename,model);
  std::cout << "nq = " << model.nq << std::endl;
  std::cout << "nv = " << model.nv << std::endl;

  Data data(model);
  VectorXd qmax = Eigen::VectorXd::Ones(model.nq);

  PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) qs     (NBT);
  PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) qdots  (NBT);
  PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) qddots (NBT);
  PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) taus (NBT);
  PINOCCHIO_ALIGNED_STD_VECTOR(MatrixXd) tau_mat (NBT);  // added a new variable
 
  for(size_t i=0;i<NBT;++i)
  {
    qs[i]     = randomConfiguration(model,-qmax,qmax);
    qdots[i]  = Eigen::VectorXd::Random(model.nv);
    qddots[i] = Eigen::VectorXd::Random(model.nv);
    taus[i] = Eigen::VectorXd::Random(model.nv);
    tau_mat[i] = Eigen::MatrixXd::Zero(model.nv,2*model.nv); // new variable
  }
  
  PINOCCHIO_EIGEN_PLAIN_ROW_MAJOR_TYPE(MatrixXd) drnea_dq(MatrixXd::Zero(model.nv,model.nv));
  PINOCCHIO_EIGEN_PLAIN_ROW_MAJOR_TYPE(MatrixXd) drnea_dv(MatrixXd::Zero(model.nv,model.nv));
  MatrixXd drnea_da(MatrixXd::Zero(model.nv,model.nv));
 
  MatrixXd daba_dq(MatrixXd::Zero(model.nv,model.nv));
  MatrixXd daba_dv(MatrixXd::Zero(model.nv,model.nv));
  Data::RowMatrixXs daba_dtau(Data::RowMatrixXs::Zero(model.nv,model.nv));
  
  timer.tic();
  SMOOTH(NBT)
  {
    forwardKinematics(model,data,qs[_smooth],qdots[_smooth],qddots[_smooth]);
  }
  std::cout << "FK= \t\t"; timer.toc(std::cout,NBT);

  timer.tic();
  SMOOTH(NBT)
  {
    computeForwardKinematicsDerivatives(model,data,qs[_smooth],qdots[_smooth],qddots[_smooth]);
  }
  std::cout << "FK derivatives= \t\t"; timer.toc(std::cout,NBT);

  timer.tic();
  SMOOTH(NBT)
  {
    rnea(model,data,qs[_smooth],qdots[_smooth],qddots[_smooth]);
  }
  std::cout << "RNEA= \t\t"; timer.toc(std::cout,NBT);
  
  timer.tic();
  SMOOTH(NBT)
  {
    computeRNEADerivatives(model,data,qs[_smooth],qdots[_smooth],qddots[_smooth],
                           drnea_dq,drnea_dv,drnea_da);
  }
  std::cout << "RNEA derivatives= \t\t"; timer.toc(std::cout,NBT);


  timer.tic();
  SMOOTH(NBT)
  {
    computeRNEADerivativesFaster(model,data,qs[_smooth],qdots[_smooth],qddots[_smooth],
                           drnea_dq,drnea_dv,drnea_da);
  }
  std::cout << "RNEA derivatives v2= \t\t"; timer.toc(std::cout,NBT);


    taus[0] = data.tau;  // input tau taken from RNEA_derivs- SS

  timer.tic();
  SMOOTH(NBT)
  {
    computeABADerivatives(model,data,qs[_smooth],qdots[_smooth],taus[_smooth],
                          daba_dq,daba_dv,daba_dtau);
  }
  std::cout << "ABA derivatives= \t\t"; timer.toc(std::cout,NBT);

  //----------------------------------------------------------------------------//
  //--------------------- New Minv_v2 = Minv + AZA multicol---------------------//
  //---------------------- SS 6/26/21 ------------------------------------------//
  //----------------------------------------------------------------------------//

    tau_mat[0] << -drnea_dq,-drnea_dv; // concatenating partial wrt q and qdot

    timer.tic();
    SMOOTH(NBT)
    {
        computeMinv_AZA(model,data,qs[_smooth],tau_mat[_smooth]);
    }
  std::cout << "Minv + AZA = \t\t"; timer.toc(std::cout,NBT);

//-------------------------------------------------------------------------------------------
// UNCOMMENT and use NBT=1 for testing this
// Recommended to use for systems with N>60
//-------------------------------------------------------------------------------------------

    // MatrixXd diff_daba_dq2(MatrixXd::Zero(model.nv,model.nv));
    // MatrixXd diff_daba_dqd2(MatrixXd::Zero(model.nv,model.nv));

    // diff_daba_dq2 = daba_dq-data.Minv_mat_prod.middleCols(0,model.nv);
    // diff_daba_dqd2 = daba_dv-data.Minv_mat_prod.middleCols(model.nv,model.nv);

    // std::cout << "------------------------------------------------------" << std::endl;
    // std::cout << "difference matrix for AZA from orig FD partial wrt q is " << diff_daba_dq2.squaredNorm() << std::endl;
    // std::cout << "difference matrix for AZA from orig FD partial wrt qd is " << diff_daba_dqd2.squaredNorm() << std::endl;
    // std::cout << "--------------------------------------------------------" << std::endl;

//-------------------------------------------------------------------------------------------

#ifndef NO_FINITE_DIFFS
  timer.tic();
  SMOOTH(NBT/100)
  {
    rnea_fd(model,data,qs[_smooth],qdots[_smooth],qddots[_smooth],
            drnea_dq,drnea_dv,drnea_da);
  }
  std::cout << "RNEA finite differences= \t\t"; timer.toc(std::cout,NBT/100);
#endif

  timer.tic();
  SMOOTH(NBT)
  {
    aba(model,data,qs[_smooth],qdots[_smooth],taus[_smooth]);
  }
  std::cout << "ABA= \t\t"; timer.toc(std::cout,NBT);
  



#ifndef NO_FINITE_DIFFS
  timer.tic();
  SMOOTH(NBT)
  {
    aba_fd(model,data,qs[_smooth],qdots[_smooth],taus[_smooth],
           daba_dq,daba_dv,daba_dtau);
  }
  std::cout << "ABA finite differences= \t\t"; timer.toc(std::cout,NBT);
#endif 

  timer.tic();
  SMOOTH(NBT)
  {
    computeMinverse(model,data,qs[_smooth]);
  }
  std::cout << "M.inverse() from ABA = \t\t"; timer.toc(std::cout,NBT);
  
//--------

  MatrixXd Minv(model.nv,model.nv); Minv.setZero();
  timer.tic();
  SMOOTH(NBT)
  {
    crba(model,data,qs[_smooth]);
    cholesky::decompose(model,data);
    cholesky::computeMinv(model,data,Minv);
  }
  std::cout << "Minv from Cholesky = \t\t"; timer.toc(std::cout,NBT);

  std::cout << "--" << std::endl;



  return 0;
}
