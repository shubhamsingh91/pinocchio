 //
// Copyright (c) 2018-2020 CNRS INRIA
// Author : Shubham Singh singh281@utexas.edu

#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/frames.hpp"
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
    auto f_vec_orig = data_fd.f; // std::vector
    
    // df/dq

    for(int i = 0; i < model.nv; ++i)
    {
        for (int j = 0; j < model.nv; ++j)
        {
            v_eps[j] += alpha;
            integrate(model,q,v_eps,q_plus);
            computeRNEADerivatives(model,data_fd,q_plus,v,a); 
            
            auto f_vec_plus = data_fd.f;

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
            
            auto f_vec_plus = data_fd.f;

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
            
            auto f_vec_plus = data_fd.f;

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
  #ifdef SPEED
  const int NBT = 1000;
  #else
    const int NBT = 1;
    std::cout << "(the time score in non-speed mode is not relevant) " << std::endl;
  #endif

  std::cout << "NBT = " << NBT << std::endl;
    
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
    else
      pinocchio::urdf::buildModel(filename,model);
  std::cout << "nq = " << model.nq << std::endl;
  std::cout << "nv = " << model.nv << std::endl;

  Data data(model);
  VectorXd qmax = Eigen::VectorXd::Ones(model.nq)*1;

  PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) qs     (NBT);
  PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) qdots  (NBT);
  PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) qddots (NBT);
  PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) taus (NBT);
  
  for(size_t i=0;i<NBT;++i)
  {
    std::cout << "i = " << i << std::endl;
    qs[i]     = randomConfiguration(model,-qmax,qmax);
    std::cout << "qs[" << i << "] = " << qs[i].transpose() << std::endl;
    // qs[i]     = Eigen::VectorXd::Ones(model.nv)*1;
    qdots[i]  = Eigen::VectorXd::Ones(model.nv)*1;
    qddots[i] = Eigen::VectorXd::Ones(model.nv)*1;
    taus[i] = Eigen::VectorXd::Ones(model.nv)*1;
  }

  std::cout << "model.njoints = " << model.njoints << std::endl;

  SMOOTH(NBT)
  {
    //-----------------------------------------//
    // Methods for Joint Jacobians
    //-----------------------------------------//
    //-----------------------------------------//
    // getting joint-IDs for each joints
    std::vector<JointIndex> joint_ids(model.njoints);
    for(JointIndex i=0; i<model.njoints; ++i)
    {
      joint_ids[i] = model.joints[i].id();
      std::cout << "joint_ids[" << i << "] = " << joint_ids[i] << std::endl;
    }
   // compute joint jacobians -  computes data.J in world frame- stack of all motion subspace matrices
   computeJointJacobians(model,data,qs[NBT-1]);


    // getjointjacobian
    pinocchio::Data::Matrix6x J(6,model.nv);
    J.setZero();
    getJointJacobian(model,data,3,WORLD,J);

    std::cout << "J = " << J << std::endl;

    //-----------------------------------------//
    // Methods for frames/frame jacobians
        //-----------------------------------------//
  // print the list of frames and IDs
  std::cout << "Frames: " << std::endl;
  for(FrameIndex i=0; i<model.nframes; ++i)
  {
    std::cout << "Frame[" << i << "] = " << model.frames[i].name
              << std::endl;
  }
  int ee_link_id = model.getFrameId("ee_link");
  std::cout << "ee_link_id = " << ee_link_id << std::endl;

  std::cout << data.oMf.size() << " frames placements in data.oMf" << std::endl;
  std::cout << "data.oMf[ee_link_id] = " << data.oMf[ee_link_id].translation().transpose() << std::endl;
  std::cout << "data.oMf[ee_link_id] = " << data.oMf[ee_link_id].rotation().transpose() << std::endl;
 
  // running FK
  forwardKinematics(model,data,qs[NBT-1]);
  updateFramePlacements(model,data);

  std::cout << "data.oMf[ee_link_id] after FK= " << data.oMf[ee_link_id].translation().transpose() << std::endl;
  std::cout << "data.oMf[ee_link_id] after FK= " << data.oMf[ee_link_id].rotation().transpose() << std::endl;




  #ifdef FO_DERIVS

  
  
    
  #endif

  #ifdef SO_DERIVS

   
  #endif   

  }


  //- Runtime test for SO derivatives
  #ifdef SPEED



  #endif

  return 0;
}