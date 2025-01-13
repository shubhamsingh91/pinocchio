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
#include "pinocchio/algorithm/spatial-force-second-order-derivatives.hpp"
#include "pinocchio/algorithm/aba-derivatives.hpp"
#include "pinocchio/algorithm/aba.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/cholesky.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/parsers/sample-models.hpp"
#include "pinocchio/container/aligned-vector.hpp"
#include <iostream>
#include <fstream>

#include "pinocchio/utils/timer.hpp"
#include "pinocchio/utils/tensor_utils.hpp"


// computes the partial of f_i^{C} w.r.t q_j
/*
*  [[dfc1_dq1] [dfc1_dq2]  [dfc1_dqn]]- page 1 of tensor
*  [[dfc2_dq1] [dfc2_dq2]  [dfc2_dqn]] - page 2 of tensor
* .
*  [[dfcn_dq1] [dfcn_dq2]  [dfcn_dqn]] - page n of tensor
*/

#ifdef FO_DERIVS
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

#endif

int main(int argc, const char ** argv)
{
  using namespace Eigen;
  using namespace pinocchio;

  typedef Eigen::Tensor<double, 3> ten3d;

  PinocchioTicToc timer(PinocchioTicToc::US);
  #ifdef NDEBUG
  const int NBT = 1;
  #else
    const int NBT = 1;
    std::cout << "(the time score in debug mode is not relevant) " << std::endl;
  #endif
    
  Model model;

  std::ifstream infile("model.txt");
    std::string filename;

    if (infile.is_open()) {
        std::getline(infile, filename);
        infile.close();
    } else {
        std::cerr << "Unable to open file";
        return 1;
    }
    std::cout << "filename: " << filename << std::endl;

  //std::string filename = PINOCCHIO_MODEL_DIR + std::string("/simple_humanoid.urdf");
  // std::string filename =  std::string("../models/double_pendulum.urdf");
  // std::string filename =  std::string("../models/simple_humanoid.urdf");
  // std::string filename =  std::string("../models/ur3_robot.urdf");
  // std::string filename =  std::string("../models/3link.urdf");
  // std::string filename =  std::string("../models/3link_bf_2.urdf");

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
  VectorXd qmax = Eigen::VectorXd::Ones(model.nq)*0.1;

  PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) qs     (NBT);
  PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) qdots  (NBT);
  PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) qddots (NBT);
  PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) taus (NBT);
  
  for(size_t i=0;i<NBT;++i)
  {
    // qs[i]     = randomConfiguration(model,-qmax,qmax);
    qs[i]     = Eigen::VectorXd::Ones(model.nv)*0.1;
    qdots[i]  = Eigen::VectorXd::Ones(model.nv)*0.1;
    qddots[i] = Eigen::VectorXd::Ones(model.nv)*0.1;
    taus[i] = Eigen::VectorXd::Ones(model.nv)*0.1;
  }

  std::cout << "model.njoints = " << model.njoints << std::endl;

  SMOOTH(NBT)
  {
  #ifdef FO_DERIVS

  PINOCCHIO_EIGEN_PLAIN_ROW_MAJOR_TYPE(MatrixXd) drnea_dq(MatrixXd::Zero(model.nv,model.nv));
  PINOCCHIO_EIGEN_PLAIN_ROW_MAJOR_TYPE(MatrixXd) drnea_dv(MatrixXd::Zero(model.nv,model.nv));
  MatrixXd drnea_da(MatrixXd::Zero(model.nv,model.nv));

    computeRNEADerivatives(model,data,qs[_smooth],qdots[_smooth],qddots[_smooth],
                           drnea_dq,drnea_dv,drnea_da);

  
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

   #endif

    //--------------------------------------------------------------------------------
    //------------------------- SO partials of cumulative force-----------------------
    //--------------------------------------------------------------------------------
    std::vector<Eigen::Tensor<double,3>> d2f_dq2_fd, d2f_dv2_fd, d2f_da2_fd , d2f_daq_fd;
    std::vector<Eigen::Tensor<double,3>> d2f_dq2_ana, d2f_dv2_ana, d2f_da2_ana, d2f_daq_ana;

    for (int i = 0; i < model.njoints - 1; i++) {
        d2f_dq2_fd.emplace_back(6, model.nv, model.nv);  
        d2f_dv2_fd.emplace_back(6, model.nv, model.nv);  
        d2f_da2_fd.emplace_back(6, model.nv, model.nv);  
        d2f_daq_fd.emplace_back(6, model.nv, model.nv);

        d2f_dq2_ana.emplace_back(6, model.nv, model.nv);  
        d2f_dv2_ana.emplace_back(6, model.nv, model.nv);  
        d2f_da2_ana.emplace_back(6, model.nv, model.nv);  
        d2f_daq_ana.emplace_back(6, model.nv, model.nv);  
    }

    for (auto &t : d2f_da2_fd) t.setZero();

    double alpha = 1e-6; // performs well

    VectorXd v_eps(VectorXd::Zero(model.nv));
    VectorXd a_eps(VectorXd::Zero(model.nv));
    VectorXd q_plus(model.nq);
    VectorXd qd_plus(model.nv);
    VectorXd a_plus(model.nv);

    ten3d df_dq_ana (6,model.nv, model.nv);
    ten3d df_dv_ana (6,model.nv, model.nv);
    ten3d df_da_ana (6,model.nv, model.nv);

    computeSpatialForceDerivs(model,data,qs[_smooth],qdots[_smooth],qddots[_smooth],
      df_dq_ana, df_dv_ana, df_da_ana);

    ten3d df_dq_ana_tensor_plus (6,model.nv, model.nv);
    ten3d df_dv_ana_tensor_plus (6,model.nv, model.nv);
    ten3d df_da_ana_tensor_plus (6,model.nv, model.nv);

    ten3d t_d2fc_dq_dqk , t_d2fc_dv_dvk, t_d2fc_da_dqk;
    Eigen::MatrixXd m_d2fci_dq_dqk(6,model.nv);
    Eigen::MatrixXd m_d2fci_da_dqk(6,model.nv);
    Eigen::MatrixXd m_d2fci_dv_dvk(6,model.nv);

    // Partial wrt q
    for (int k = 0; k < model.nv; ++k) {
        v_eps[k] += alpha;
        q_plus = integrate(model, qs[_smooth], v_eps); // This is used to add the v_eps to q in the k^th direction
      
        computeSpatialForceDerivs(model,data,q_plus,qdots[_smooth],qddots[_smooth],
        df_dq_ana_tensor_plus, df_dv_ana_tensor_plus, df_da_ana_tensor_plus);

        t_d2fc_dq_dqk = (df_dq_ana_tensor_plus - df_dq_ana) / alpha; // 3d tensor d2fc/dq dqk
        t_d2fc_da_dqk = (df_da_ana_tensor_plus - df_da_ana) / alpha; // 3d tensor d2fc/da dqk
        
        for (int i = 0; i < model.nv; i++)
        {
          get_mat_from_tens3_v1_gen(t_d2fc_dq_dqk, m_d2fci_dq_dqk, 6, model.nv, i);
          hess_assign_fd_v1_gen(d2f_dq2_fd.at(i), m_d2fci_dq_dqk, 6, model.nv, k); // slicing in the matrix along the kth page for ith tensor             
     
          get_mat_from_tens3_v1_gen(t_d2fc_da_dqk, m_d2fci_da_dqk, 6, model.nv, i);
          hess_assign_fd_v1_gen(d2f_daq_fd.at(i), m_d2fci_da_dqk, 6, model.nv, k); // slicing in the matrix along the kth page for ith tensor
        }

        v_eps[k] -= alpha;
    }

    // Partial wrt v
    for (int k = 0; k < model.nv; ++k) {
    v_eps[k] += alpha;
    qd_plus = qdots[_smooth] + v_eps; // This is used to add the v_eps to q in the k^th direction

    computeSpatialForceDerivs(model,data,qs[_smooth], qd_plus ,qddots[_smooth],
    df_dq_ana_tensor_plus, df_dv_ana_tensor_plus, df_da_ana_tensor_plus);

    t_d2fc_dv_dvk = (df_dv_ana_tensor_plus - df_dv_ana) / alpha; // 3d tensor d2fc/dq dqk
    for (int i = 0; i < model.nv; i++)
    {
          get_mat_from_tens3_v1_gen(t_d2fc_dv_dvk, m_d2fci_dv_dvk, 6, model.nv, i);
    hess_assign_fd_v1_gen(d2f_dv2_fd.at(i), m_d2fci_dv_dvk, 6, model.nv, k); // slicing in the matrix along the kth page for ith tensor     
    }

    v_eps[k] -= alpha;
    }
   //------- Analytical algorithm

    ComputeSpatialForceSecondOrderDerivatives(model, data, qs[_smooth], qdots[_smooth], qddots[_smooth],
                                             d2f_dq2_ana, d2f_dv2_ana, d2f_daq_fd, d2f_daq_ana);


    // Comparing the results
    for (int i = 0; i < model.nv; ++i)
    {
     
     std::cout << "i = " << i << std::endl;
      Eigen::Tensor<double,3> concrete_tensor = (d2f_dq2_fd.at(i) - d2f_dq2_ana.at(i)).eval();
      auto diff_eq = tensorMax(concrete_tensor);

      Eigen::Tensor<double,3> concrete_tensor_SO_v = (d2f_dv2_fd.at(i) - d2f_dv2_ana.at(i)).eval();
      auto diff_dv = tensorMax(concrete_tensor_SO_v);

      Eigen::Tensor<double,3> concrete_tensor_SO_a = (d2f_da2_fd.at(i) - d2f_da2_ana.at(i)).eval();
      auto diff_da = tensorMax(concrete_tensor_SO_a);

      Eigen::Tensor<double,3> concrete_tensor_SO_aq = (d2f_daq_fd.at(i) - d2f_daq_ana.at(i)).eval();
      auto diff_daq = tensorMax(concrete_tensor_SO_aq);

      if (diff_eq > 1e-3)
      {
        std::cout << "diff SO-q \n"   << std::endl;
      }

      if (diff_dv > 1e-3)
      {
        std::cout << "diff SO-v \n"   << std::endl;
      }

      if (diff_da > 1e-3)
      {
        std::cout << "diff SO-a =  " << diff_da   << std::endl;

      }

      if (diff_daq > 1e-3)
      {
        std::cout << "diff SO-aq \n"   << std::endl;
     
      }
    }

  }


  return 0;
}