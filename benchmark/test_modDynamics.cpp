// Author : Shubham Singh singh281@utexas.edu

#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/aba-derivatives.hpp"
#include "pinocchio/algorithm/aba.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/modrnea.hpp"
#include "pinocchio/algorithm/modaba.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/cholesky.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/parsers/sample-models.hpp"
#include "pinocchio/container/aligned-vector.hpp"
#include <iostream>
#include <fstream>

#include "pinocchio/utils/timer.hpp"
#include "pinocchio/utils/tensor_utils.hpp"

int main(int argc, const char ** argv)
{
  using namespace Eigen;
  using namespace pinocchio;

  PinocchioTicToc timer(PinocchioTicToc::US);
  #ifdef SPEED
  const int NBT = 1000;
  #else
    const int NBT = 1000;
    std::cout << "(the time score in non-speed mode is not relevant) " << std::endl;
  #endif

  std::cout << "NBT = " << NBT << std::endl;
    
  int n_models = 6; // no of robots to be used
  string str_robotname[n_models];

  str_robotname[0] = "double_pendulum"; // double pendulum
  str_robotname[1] = "ur3_robot";       // UR3
  str_robotname[2] = "hyq";             // hyq
  str_robotname[3] = "baxter_simple";   // baxter_simple
  str_robotname[4] = "atlas";           // atlas
  str_robotname[5] = "talos_full_v2";   // Talos
  char tmp[256];
  getcwd(tmp, 256); 

  Model model;

for (int mm = 0; mm < n_models; mm++) {

    Model model;

    string str_file_ext;
    string robot_name = "";
    string str_urdf;

    robot_name = str_robotname[mm];
    std ::string filename = "../models/" + robot_name + std::string(".urdf");

    bool with_ff = false; // All for only fixed-base models
    if ((mm == 2) || (mm == 4) || (mm == 5)) {
        with_ff = true; // True for hyQ and atlas, talos_full_v2
    }
    if (with_ff)
        pinocchio::urdf::buildModel(filename, JointModelFreeFlyer(), model);
    else
        pinocchio::urdf::buildModel(filename, model);
    if (with_ff) {
        robot_name += std::string("_f");
    }


  cout << "Model is" << robot_name << endl;
  std::cout << "nq = " << model.nq << std::endl;
  std::cout << "nv = " << model.nv << std::endl;

  Data data(model);
  VectorXd qmax = Eigen::VectorXd::Random(model.nq);

  PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) qs     (NBT);
  PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) qdots  (NBT);
  PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) qddots (NBT);
  PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) taus (NBT);
  PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) lambdas (NBT);
  PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) mus (NBT);
  
  
  for(size_t i=0;i<NBT;++i)
  {
    qs[i]     = randomConfiguration(model,-qmax,qmax);
    // qs[i]     = Eigen::VectorXd::Random(model.nv);
    qdots[i]  = Eigen::VectorXd::Random(model.nv);
    qddots[i] = Eigen::VectorXd::Random(model.nv);
    taus[i] = Eigen::VectorXd::Random(model.nv);
    lambdas[i] = Eigen::VectorXd::Random(model.nv);
    mus[i] = Eigen::VectorXd::Random(model.nv);
  }


  SMOOTH(NBT)
  {
    // Compute modified ID
    rnea(model,data,qs[_smooth],qdots[_smooth],qddots[_smooth]);
    taus[_smooth] = data.tau;


    double modtau = modrnea(model,data,qs[_smooth],qdots[_smooth],qddots[_smooth], lambdas[_smooth]);

    double diff_mod = modtau - lambdas[_smooth].transpose()*taus[_smooth];

    if (abs(diff_mod)>1e-6) {
      throw std::runtime_error("modID is not correct");
    std::cout << "modtau = " << modtau << 
       "  , Accuracy check for modID = " << diff_mod << std::endl;
    }

    // compute Modified FD

    Eigen::VectorXd qddot = aba(model,data,qs[_smooth],qdots[_smooth],taus[_smooth]);
    double modqdd = modaba(model,data,qs[_smooth],qdots[_smooth],taus[_smooth], mus[_smooth]);

    double diff_modFD = modqdd - mus[_smooth].transpose()*qddot;

    if (abs(diff_modFD)>1e-6) {
      throw std::runtime_error("modFD is not correct");
    std::cout << "modqdd = " << modqdd << 
       "  , Accuracy check for modFD = " << diff_modFD << std::endl;

    }


  }
   




  

}

    return 0;
}