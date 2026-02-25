// Author : Shubham Singh singh281@utexas.edu

#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/rnea-derivatives-faster.hpp"
#include "pinocchio/algorithm/mod-rnea-second-order-derivatives.hpp"
#include "pinocchio/algorithm/mod-rnea-derivatives.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/parsers/sample-models.hpp"
#include "pinocchio/container/aligned-vector.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include "pinocchio/utils/timer.hpp"
#include "pinocchio/utils/tensor_utils.hpp"

void print_pretty(const std::string & str)
{
  std::cout << "\n##############################################" << std::endl;
  std::cout << str << std::endl;
  std::cout << "##############################################\n" << std::endl;
}

int main(int argc, const char ** argv)
{
  using namespace Eigen;
  using namespace pinocchio;

  // std::cout << std::fixed << std::setprecision(10); // Uncomment for more precision

  PinocchioTicToc timer(PinocchioTicToc::US);
  #ifdef SPEED
  const int NBT = 1000;
  #else
    const int NBT = 1;
    std::cout << "(the time score in non-speed mode is not relevant) " << std::endl;
  #endif

  std::cout << "NBT = " << NBT << std::endl;
    
  std::vector<std::string> robot_name_vec;

  robot_name_vec.push_back("double_pendulum"); // double pendulum
  robot_name_vec.push_back("ur3_robot");       // UR3
  robot_name_vec.push_back("hyq");             // hyq
  robot_name_vec.push_back("baxter_simple");   // baxter_simple
  robot_name_vec.push_back("atlas");           // atlas

  char tmp[256];
  getcwd(tmp, 256); 

  Model model;

for (int mm = 0; mm < robot_name_vec.size(); mm++) {

    Model model;

    string str_file_ext;
    string robot_name = "";
    string str_urdf;

    robot_name = robot_name_vec.at(mm);
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
    // Use fixed values matching MATLAB for debugging
    if (model.nv == 3) {
      qs[i].resize(3);
      qs[i] << 0.2551, 0.5060, 0.6991;
      qdots[i].resize(3);
      qdots[i] << 0.8909, 0.9593, 0.5472;
      qddots[i].resize(3);
      qddots[i] << 0.1386, 0.1493, 0.2575;
      lambdas[i].resize(3);
      lambdas[i] << 0.8407, 0.2543, 0.8143;
    } else {
      qs[i]     = randomConfiguration(model,-qmax,qmax);
      qdots[i]  = Eigen::VectorXd::Random(model.nv);
      qddots[i] = Eigen::VectorXd::Random(model.nv);
      lambdas[i] = Eigen::VectorXd::Random(model.nv);
    }
    taus[i] = Eigen::VectorXd::Random(model.nv);
    mus[i] = Eigen::VectorXd::Random(model.nv);
  }


  SMOOTH(NBT)
  {
    // randomizing fext
    typedef PINOCCHIO_ALIGNED_STD_VECTOR(Force) ForceVector;
    ForceVector fext((size_t)model.njoints);
    for(ForceVector::iterator it = fext.begin(); it != fext.end(); ++it)
      (*it).setRandom();

    VectorXd dtau_dq_mod_orig(VectorXd::Zero(model.nv));
    VectorXd dtau_dv_mod_orig(VectorXd::Zero(model.nv));
    VectorXd dtau_da_mod_orig(VectorXd::Zero(model.nv));

    VectorXd dtau_dq_mod_plus(VectorXd::Zero(model.nv));
    VectorXd dtau_dv_mod_plus(VectorXd::Zero(model.nv));
    VectorXd dtau_da_mod_plus(VectorXd::Zero(model.nv));

    // SO derivs of mod-ID using finite-diff
    MatrixXd dtau_dqq_mod_fd(MatrixXd::Zero(model.nv, model.nv));
    MatrixXd dtau_dvv_mod_fd(MatrixXd::Zero(model.nv, model.nv));
    MatrixXd dtau_dvq_mod_fd(MatrixXd::Zero(model.nv, model.nv));
    MatrixXd dtau_daq_mod_fd(MatrixXd::Zero(model.nv, model.nv));

    // SO derivs of mod-ID analytical
    MatrixXd dtau_dqq_mod(MatrixXd::Zero(model.nv, model.nv));
    MatrixXd dtau_dvv_mod(MatrixXd::Zero(model.nv, model.nv));
    MatrixXd dtau_dvq_mod(MatrixXd::Zero(model.nv, model.nv));
    MatrixXd dtau_daq_mod(MatrixXd::Zero(model.nv, model.nv));

    dtau_dqq_mod.setZero();  // Initialize to zero before the algorithm
    dtau_dvv_mod.setZero();
    dtau_dvq_mod.setZero();

    // Print q, qd, qdd, lambda so we can use same values in MATLAB
    std::cout << "\nq = " << qs[_smooth].transpose() << std::endl;
    std::cout << "qd = " << qdots[_smooth].transpose() << std::endl;
    std::cout << "qdd = " << qddots[_smooth].transpose() << std::endl;
    std::cout << "lambda = " << lambdas[_smooth].transpose() << std::endl;

    computeModRNEASecondOrderDerivatives(model, data, qs[_smooth], qdots[_smooth],
                                        qddots[_smooth], lambdas[_smooth],
                                        dtau_dqq_mod, dtau_dvv_mod, dtau_dvq_mod);
  
    // Running modID rnea derivatives
    computeModRNEADerivatives(model, data, qs[_smooth], qdots[_smooth], 
                             qddots[_smooth], lambdas[_smooth]);

    dtau_dq_mod_orig = data.dtau_dq_mod;
    dtau_dv_mod_orig = data.dtau_dv_mod;
    dtau_da_mod_orig = data.M_mod;

    // Debug: verify first-order derivatives using FD
    std::cout << "\n=== First-order derivative verification ===" << std::endl;
    std::cout << "dtau_dq_mod_orig = " << dtau_dq_mod_orig.transpose() << std::endl;
    std::cout << "dtau_dv_mod_orig = " << dtau_dv_mod_orig.transpose() << std::endl;
    std::cout << "M_mod = " << dtau_da_mod_orig.transpose() << std::endl;

    // Compute modID value at current point for FD verification
    VectorXd tau_orig = pinocchio::rnea(model, data, qs[_smooth], qdots[_smooth], qddots[_smooth]);
    double modID_orig = tau_orig.dot(lambdas[_smooth]);
    std::cout << "modID_orig (tau.dot(lambda)) = " << modID_orig << std::endl;

    // First-order FD verification
    VectorXd dtau_dq_fd(VectorXd::Zero(model.nv));
    VectorXd dtau_dv_fd(VectorXd::Zero(model.nv));
    VectorXd dtau_da_fd(VectorXd::Zero(model.nv));
    {
        VectorXd v_eps_fo(VectorXd::Zero(model.nv));
        VectorXd q_plus_fo(model.nq);
        VectorXd v_plus_fo(qdots[_smooth]);
        VectorXd a_plus_fo(qddots[_smooth]);
        const double alpha_fo = 1e-8;

        // d(modID)/dq via FD
        for(int k = 0; k < model.nv; ++k) {
            v_eps_fo[k] = alpha_fo;
            pinocchio::integrate(model, qs[_smooth], v_eps_fo, q_plus_fo);
            VectorXd tau_plus = pinocchio::rnea(model, data, q_plus_fo, qdots[_smooth], qddots[_smooth]);
            double modID_plus = tau_plus.dot(lambdas[_smooth]);
            dtau_dq_fd[k] = (modID_plus - modID_orig) / alpha_fo;
            v_eps_fo[k] = 0;
        }

        // d(modID)/dv via FD
        for(int k = 0; k < model.nv; ++k) {
            v_plus_fo[k] += alpha_fo;
            VectorXd tau_plus = pinocchio::rnea(model, data, qs[_smooth], v_plus_fo, qddots[_smooth]);
            double modID_plus = tau_plus.dot(lambdas[_smooth]);
            dtau_dv_fd[k] = (modID_plus - modID_orig) / alpha_fo;
            v_plus_fo[k] -= alpha_fo;
        }

        // d(modID)/da via FD
        for(int k = 0; k < model.nv; ++k) {
            a_plus_fo[k] += alpha_fo;
            VectorXd tau_plus = pinocchio::rnea(model, data, qs[_smooth], qdots[_smooth], a_plus_fo);
            double modID_plus = tau_plus.dot(lambdas[_smooth]);
            dtau_da_fd[k] = (modID_plus - modID_orig) / alpha_fo;
            a_plus_fo[k] -= alpha_fo;
        }

        std::cout << "dtau_dq_fd = " << dtau_dq_fd.transpose() << std::endl;
        std::cout << "dtau_dv_fd = " << dtau_dv_fd.transpose() << std::endl;
        std::cout << "dtau_da_fd = " << dtau_da_fd.transpose() << std::endl;
        std::cout << "FO dq diff = " << (dtau_dq_mod_orig - dtau_dq_fd).norm() << std::endl;
        std::cout << "FO dv diff = " << (dtau_dv_mod_orig - dtau_dv_fd).norm() << std::endl;
        std::cout << "FO da diff = " << (dtau_da_mod_orig - dtau_da_fd).norm() << std::endl;
    }

    VectorXd v_eps(VectorXd::Zero(model.nv));
    VectorXd q_plus(model.nq);
    const double alpha = 1e-8;

    // SO partial derivatives of modID using finite-differences
    // d²tau/dqq via FD
    for(int k = 0; k < model.nv; ++k)
    {
      v_eps[k] += alpha;
      pinocchio::integrate(model,qs[_smooth],v_eps,q_plus);
      computeModRNEADerivatives(model, data, q_plus, qdots[_smooth],
                                qddots[_smooth], lambdas[_smooth],
                                dtau_dq_mod_plus, dtau_dv_mod_plus, dtau_da_mod_plus);
      dtau_dqq_mod_fd.col(k) = (dtau_dq_mod_plus - dtau_dq_mod_orig)/alpha;
      dtau_dvq_mod_fd.col(k) = (dtau_dv_mod_plus - dtau_dv_mod_orig)/alpha;
      v_eps[k] -= alpha;
    }

    VectorXd v_plus(qdots[_smooth]);

    // d²tau/dvv via FD
    for(int k = 0; k < model.nv; ++k)
    {
      v_plus[k] += alpha;
      computeModRNEADerivatives(model, data, qs[_smooth], v_plus,
                                qddots[_smooth], lambdas[_smooth],
                                dtau_dq_mod_plus, dtau_dv_mod_plus, dtau_da_mod_plus);

      dtau_dvv_mod_fd.col(k) = (dtau_dv_mod_plus - dtau_dv_mod_orig)/alpha;
      v_plus[k] -= alpha;
    }

    // compare between analytical and finite-diff
    MatrixXd dtau_dqq_mod_diff = dtau_dqq_mod - dtau_dqq_mod_fd;
    MatrixXd dtau_dvv_mod_diff = dtau_dvv_mod - dtau_dvv_mod_fd;
    // Note: dtau_dvq_mod is d²τ/(dq dv), FD computes d²τ/(dv dq)
    // For mixed partials, these are equal (by Schwarz's theorem), but the matrix
    // representation differs by transpose: (d²τ/dqdv)_ij = ∂²τ/(∂q_i ∂v_j)
    // FD gives: (d²τ/dvdq)_ij = ∂²τ/(∂v_i ∂q_j) = ∂²τ/(∂q_j ∂v_i) = (d²τ/dqdv)_ji
    MatrixXd dtau_dvq_mod_diff = dtau_dvq_mod - dtau_dvq_mod_fd.transpose();

    std::cout << "dtau_dqq_mod_diff = " << dtau_dqq_mod_diff.norm() << std::endl;
    std::cout << "dtau_dvv_mod_diff = " << dtau_dvv_mod_diff.norm() << std::endl;
    std::cout << "dtau_dvq_mod_diff (vs FD') = " << dtau_dvq_mod_diff.norm() << std::endl;

    std::cout << "\ndtau_dqq_mod =\n" << dtau_dqq_mod << std::endl;
    std::cout << "\ndtau_dqq_mod_fd =\n" << dtau_dqq_mod_fd << std::endl;
    std::cout << "\ndqq diff =\n" << dtau_dqq_mod_diff << std::endl;

  }

   

}

    return 0;
}