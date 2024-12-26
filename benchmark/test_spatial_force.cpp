//
// Copyright (c) 2018-2020 CNRS INRIA
//

#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/kinematics-derivatives.hpp"
#include "pinocchio/algorithm/rnea-derivatives.hpp"
#include "pinocchio/algorithm/spatial-force-derivatives.hpp"
#include "pinocchio/algorithm/rnea-derivatives-faster.hpp"
#include "pinocchio/algorithm/rnea-second-order-derivatives.hpp"
#include "pinocchio/algorithm/aba-derivatives.hpp"
#include "pinocchio/algorithm/aba.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/cholesky.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/parsers/sample-models.hpp"
#include "pinocchio/container/aligned-vector.hpp"
#include <iostream>

#include "pinocchio/utils/timer.hpp"
#include "pinocchio/utils/tensor_utils.hpp"


// computes the partial of f_i^{C} w.r.t q_j
/*
*  [[dfc1_dq1] [dfc1_dq2]  [dfc1_dqn]]- page 1 of tensor
*  [[dfc2_dq1] [dfc2_dq2]  [dfc2_dqn]] - page 2 of tensor
* .
*  [[dfcn_dq1] [dfcn_dq2]  [dfcn_dqn]] - page n of tensor
*/


void calc_df(const pinocchio::Model & model, pinocchio::Data & data_fd,
             const Eigen::VectorXd & q,
             const Eigen::VectorXd & v,
             const Eigen::VectorXd & a,
             Eigen::Tensor<double,3>  & df_dq,
             Eigen::Tensor<double,3>  & df_dv,
             Eigen::Tensor<double,3>  & df_da)

{
    using namespace Eigen;
    VectorXd v_eps(VectorXd::Zero(model.nv));
    VectorXd q_plus(model.nq);
    const double alpha = 1e-8;

    computeRNEADerivatives(model,data_fd,q,v,a);
    auto f_vec_orig = data_fd.of; // std::vector
    
    // df/dq

    for(int i = 0; i < model.nv; ++i)
    {
        for (int j = 0; j < model.nv; ++j)
        {
            v_eps[j] += alpha;
            integrate(model,q,v_eps,q_plus);
            computeRNEADerivatives(model,data_fd,q_plus,v,a); 
            
            auto f_vec_plus = data_fd.of;

            auto vec6 = (f_vec_plus.at(i+1).toVector()-f_vec_orig.at(i+1).toVector())/alpha; // dfci_dqj
            pinocchio::tens_assign6_col(df_dq, vec6, j, i); // assigning derivative w.r.t qj along the columns

            v_eps[j] -= alpha;

        }
    }

    // df/dv
    VectorXd v_plus(v);
    for(int i = 0; i < model.nv; ++i)
    {
        for (int j = 0; j < model.nv; ++j)
        {
            v_plus[j] += alpha;
            computeRNEADerivatives(model,data_fd,q,v_plus,a); 
            
            auto f_vec_plus = data_fd.of;

            auto vec6 = (f_vec_plus.at(i+1).toVector()-f_vec_orig.at(i+1).toVector())/alpha; // dfci_dqj
            pinocchio::tens_assign6_col(df_dv, vec6, j, i); // assigning derivative w.r.t qj along the columns

            v_plus[j] -= alpha;

        }
    }

    // df/dv
    VectorXd a_plus(a);
    for(int i = 0; i < model.nv; ++i)
    {
        for (int j = 0; j < model.nv; ++j)
        {
            a_plus[j] += alpha;
            computeRNEADerivatives(model,data_fd,q,v,a_plus); 
            
            auto f_vec_plus = data_fd.of;

            auto vec6 = (f_vec_plus.at(i+1).toVector()-f_vec_orig.at(i+1).toVector())/alpha; // dfci_dqj
            pinocchio::tens_assign6_col(df_da, vec6, j, i); // assigning derivative w.r.t qj along the columns

            a_plus[j] -= alpha;

        }
    }
}


int main(int argc, const char ** argv)
{
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

  //std::string filename = PINOCCHIO_MODEL_DIR + std::string("/simple_humanoid.urdf");
  // std::string filename =  std::string("../models/double_pendulum.urdf");
  std::string filename =  std::string("../models/simple_humanoid.urdf");

  if(argc>1) filename = argv[1];
  bool with_ff = false;
  
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
  VectorXd qmax = Eigen::VectorXd::Random(model.nq);

  PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) qs     (NBT);
  PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) qdots  (NBT);
  PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) qddots (NBT);
  PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) taus (NBT);
  
  for(size_t i=0;i<NBT;++i)
  {
    // qs[i]     = randomConfiguration(model,-qmax,qmax);
    qs[i]     = Eigen::VectorXd::Random(model.nv);
    qdots[i]  = Eigen::VectorXd::Random(model.nv);
    qddots[i] = Eigen::VectorXd::Random(model.nv);
    taus[i] = Eigen::VectorXd::Random(model.nv);
  }
  
  PINOCCHIO_EIGEN_PLAIN_ROW_MAJOR_TYPE(MatrixXd) drnea_dq(MatrixXd::Zero(model.nv,model.nv));
  PINOCCHIO_EIGEN_PLAIN_ROW_MAJOR_TYPE(MatrixXd) drnea_dv(MatrixXd::Zero(model.nv,model.nv));
  MatrixXd drnea_da(MatrixXd::Zero(model.nv,model.nv));

  SMOOTH(NBT)
  {
    computeRNEADerivatives(model,data,qs[_smooth],qdots[_smooth],qddots[_smooth],
                           drnea_dq,drnea_dv,drnea_da);

    std::cout << "model.njoints = " << model.njoints << std::endl;
  
    auto f_vec = data.of;


    // Finite-diff of df_dq
    pinocchio::DataTpl<double>::Matrix6x df_dq_fd, df_dv_fd, df_da_fd;
    Eigen::Tensor<double, 3> df_dq_fd_tensor (6,model.nv, model.nv);
    Eigen::Tensor<double, 3> df_dv_fd_tensor (6,model.nv, model.nv);
    Eigen::Tensor<double, 3> df_da_fd_tensor (6,model.nv, model.nv);

    Eigen::Tensor<double, 3> df_dq_ana_tensor (6,model.nv, model.nv);
    Eigen::Tensor<double, 3> df_dv_ana_tensor (6,model.nv, model.nv);
    Eigen::Tensor<double, 3> df_da_ana_tensor (6,model.nv, model.nv);

    df_dq_fd.resize(6,model.nv);
    df_dv_fd.resize(6,model.nv);
    df_da_fd.resize(6,model.nv);

    std::cout << "---------- Running finite-diff --------------- " << std::endl;
    calc_df(model, data, qs[_smooth], qdots[_smooth], qddots[_smooth], 
              df_dq_fd_tensor, df_dv_fd_tensor, df_da_fd_tensor);

    std::cout << "---------- Running Analytical --------------- " << std::endl;
    computeSpatialForceDerivs(model,data,qs[_smooth],qdots[_smooth],qddots[_smooth],
          df_dq_ana_tensor, df_dv_ana_tensor, df_da_ana_tensor);
    

    std::cout << "------------ Comparing df_dq----------------------" << std::endl;

    for (int i = 0; i< model.nv; i++)
    {   
        for (int j = 0; j < model.nv; j++)
        {
          // std::cout << " j = " << j << std::endl;

          pinocchio::DataTpl<double>::Vector6c temp, temp_ana;
          pinocchio::hess_get(df_dq_fd_tensor,temp,     0,j,i,1,6);
          pinocchio::hess_get(df_dq_ana_tensor,temp_ana,0,j,i,1,6); // dfic_dqi

          auto diff = (temp - temp_ana).norm();

          if (i == j)
          {
            df_dq_fd.col(i) = temp; // note that this is what Pinocchio's rnea-derivatives.hxx return
          }
          
          if (diff > 1e-3)
          {
          std::cout << " i = " << i << std::endl;
          std::cout << " j = " << j << std::endl;
          std::cout << "diff between fd and analytical = " << diff << std::endl;
          }
        }
    }

    std::cout << "------------ Comparing df_dv----------------------" << std::endl;
    for (int i = 0; i< model.nv; i++)
    {   
        // std::cout << " i = " << i << std::endl;
        for (int j = 0; j < model.nv; j++)
        {

          pinocchio::DataTpl<double>::Vector6c temp, temp_ana;
          pinocchio::hess_get(df_dv_fd_tensor,temp,     0,j,i,1,6);
          pinocchio::hess_get(df_dv_ana_tensor,temp_ana,0,j,i,1,6); // dfic_dqi

          if (i == j)
          {
            df_dv_fd.col(i) = temp; // note that this is what Pinocchio's rnea-derivatives.hxx return
          }

          auto diff = (temp - temp_ana).norm();
          
          if (diff > 1e-3)
          {
          std::cout << " i = " << i << std::endl;
          std::cout << " j = " << j << std::endl;
          std::cout << "diff between fd and analytical = " << diff << std::endl;
          }
        }
    }

    std::cout << "------------ Comparing df_da----------------------" << std::endl;
    for (int i = 0; i< model.nv; i++)
    {   
        // std::cout << " i = " << i << std::endl;
        for (int j = 0; j < model.nv; j++)
        {

          pinocchio::DataTpl<double>::Vector6c temp, temp_ana;
          pinocchio::hess_get(df_da_fd_tensor,temp,     0,j,i,1,6);
          pinocchio::hess_get(df_da_ana_tensor,temp_ana,0,j,i,1,6); // dfic_dqi
          
          if (i == j)
          {
            df_da_fd.col(i) = temp; // note that this is what Pinocchio's rnea-derivatives.hxx return
          }

          auto diff = (temp - temp_ana).norm();
          
          if (diff > 1e-3)
          {
          std::cout << " i = " << i << std::endl;
          std::cout << " j = " << j << std::endl;
          std::cout << "diff between fd and analytical = " << diff << std::endl;
          }
        }
    }

    std::cout << "diff bw Pinocchio's dFdq and finite-diff = " << (df_dq_fd - data.dFdq).norm() << std::endl;
    std::cout << "diff bw Pinocchio's dFdv and finite-diff = " << (df_dv_fd - data.dFdv).norm() << std::endl;
    std::cout << "diff bw Pinocchio's dFda and finite-diff = " << (df_da_fd - data.dFda).norm() << std::endl;


  }


  return 0;
}