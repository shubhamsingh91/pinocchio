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
    qs[i]     = randomConfiguration(model,-qmax,qmax);
    qdots[i]  = Eigen::VectorXd::Random(model.nv);
    qddots[i] = Eigen::VectorXd::Random(model.nv);
    taus[i] = Eigen::VectorXd::Random(model.nv);
    lambdas[i] = Eigen::VectorXd::Random(model.nv);
    mus[i] = Eigen::VectorXd::Random(model.nv);
  }


  SMOOTH(NBT)
  {
    VectorXd dtau_dq_mod_orig(VectorXd::Zero(model.nv));
    VectorXd dtau_dv_mod_orig(VectorXd::Zero(model.nv));
    VectorXd dtau_da_mod_orig(VectorXd::Zero(model.nv));

    VectorXd dtau_dq_mod_plus(VectorXd::Zero(model.nv));
    VectorXd dtau_dv_mod_plus(VectorXd::Zero(model.nv));
    VectorXd dtau_da_mod_plus(VectorXd::Zero(model.nv));

    // SO derivs of mod-ID analytical
    MatrixXd dtau_dqq_mod(MatrixXd::Zero(model.nv, model.nv));
    MatrixXd dtau_dvv_mod(MatrixXd::Zero(model.nv, model.nv));
    MatrixXd dtau_dvq_mod(MatrixXd::Zero(model.nv, model.nv));
    MatrixXd dtau_qa_mod(MatrixXd::Zero(model.nv, model.nv));

    // SO derivs of mod-ID using finite-diff
    MatrixXd dtau_dqq_mod_fd(MatrixXd::Zero(model.nv, model.nv));
    MatrixXd dtau_dvv_mod_fd(MatrixXd::Zero(model.nv, model.nv));
    MatrixXd dtau_dvq_mod_fd(MatrixXd::Zero(model.nv, model.nv));
    MatrixXd dtau_qa_mod_fd(MatrixXd::Zero(model.nv, model.nv));

    // Compute SO derivatives analytically (4-matrix version: dqq, dvv, dqv, dqa)
    print_pretty("mod ID SO derivatives");
    computeModRNEASecondOrderDerivatives(model, data, qs[_smooth], qdots[_smooth],
                                        qddots[_smooth], lambdas[_smooth],
                                        dtau_dqq_mod, dtau_dvv_mod, dtau_dvq_mod, dtau_qa_mod);

    // Compute FO derivatives at nominal point (needed for FD of SO)
    computeModRNEADerivatives(model, data, qs[_smooth], qdots[_smooth],
                             qddots[_smooth], lambdas[_smooth]);

    dtau_dq_mod_orig = data.dtau_dq_mod;
    dtau_dv_mod_orig = data.dtau_dv_mod;
    dtau_da_mod_orig = data.M_mod;

    VectorXd v_eps(VectorXd::Zero(model.nv));
    VectorXd q_plus(model.nq);
    const double alpha = 1e-7;

    // d²(modID)/dqq and d²(modID)/dvq via FD (perturb q, observe change in FO derivs)
    for(int k = 0; k < model.nv; ++k)
    {
      v_eps[k] += alpha;
      pinocchio::integrate(model,qs[_smooth],v_eps,q_plus);
      computeModRNEADerivatives(model, data, q_plus, qdots[_smooth],
                                qddots[_smooth], lambdas[_smooth],
                                dtau_dq_mod_plus, dtau_dv_mod_plus, dtau_da_mod_plus);
      dtau_dqq_mod_fd.col(k) = (dtau_dq_mod_plus - dtau_dq_mod_orig)/alpha;
      dtau_dvq_mod_fd.col(k) = (dtau_dv_mod_plus - dtau_dv_mod_orig)/alpha;
      dtau_qa_mod_fd.col(k) = (dtau_da_mod_plus - dtau_da_mod_orig)/alpha;
      v_eps[k] -= alpha;
    }

    // d²(modID)/dvv via FD (perturb v, observe change in FO derivs)
    VectorXd v_plus(qdots[_smooth]);
    for(int k = 0; k < model.nv; ++k)
    {
      v_plus[k] += alpha;
      computeModRNEADerivatives(model, data, qs[_smooth], v_plus,
                                qddots[_smooth], lambdas[_smooth],
                                dtau_dq_mod_plus, dtau_dv_mod_plus, dtau_da_mod_plus);
      dtau_dvv_mod_fd.col(k) = (dtau_dv_mod_plus - dtau_dv_mod_orig)/alpha;
      v_plus[k] -= alpha;
    }

    // Check: analytical vs finite-diff
    double dqq_err = (dtau_dqq_mod - dtau_dqq_mod_fd).norm();
    double dvv_err = (dtau_dvv_mod - dtau_dvv_mod_fd).norm();
    double dvq_err = (dtau_dvq_mod - dtau_dvq_mod_fd.transpose()).norm();

    std::cout << "dtau_dqq_mod_diff = " << dqq_err << std::endl;
    std::cout << "dtau_dvv_mod_diff = " << dvv_err << std::endl;
    std::cout << "dtau_dvq_mod_diff = " << dvq_err << std::endl;

    if (dqq_err > std::sqrt(alpha)) {
      std::cout << "dtau_dqq_mod_diff = " << dqq_err << std::endl;
      throw std::runtime_error("dtau_dqq_mod is not correct");
    }

    if (dvv_err > std::sqrt(alpha)) {
      std::cout << "dtau_dvv_mod_diff = " << dvv_err << std::endl;
      throw std::runtime_error("dtau_dvv_mod is not correct");
    }

    if (dvq_err > std::sqrt(alpha)) {
      std::cout << "dtau_dvq_mod_diff = " << dvq_err << std::endl;
      throw std::runtime_error("dtau_dvq_mod is not correct");
    }

    // ===================== dtau_vq = dtau_qv^T (Schwarz's theorem) =====================
    print_pretty("mod ID SO derivatives: dtau_vq");

    // dtau_dvq_mod stores ∂²(λτ)/(∂q_i ∂v_j), so dtau_vq = dtau_dvq_mod^T
    MatrixXd dtau_vq_mod = dtau_dvq_mod.transpose();

    // Verify with FD: perturb v_k, observe change in dtau_dq_mod
    MatrixXd dtau_vq_mod_fd(MatrixXd::Zero(model.nv, model.nv));
    VectorXd v_plus_vq(qdots[_smooth]);
    for(int k = 0; k < model.nv; ++k)
    {
      v_plus_vq[k] += alpha;
      computeModRNEADerivatives(model, data, qs[_smooth], v_plus_vq,
                                qddots[_smooth], lambdas[_smooth],
                                dtau_dq_mod_plus, dtau_dv_mod_plus, dtau_da_mod_plus);
      dtau_vq_mod_fd.col(k) = (dtau_dq_mod_plus - dtau_dq_mod_orig) / alpha;
      v_plus_vq[k] -= alpha;
    }

    // dtau_vq_mod_fd(j,k) = ∂²(λτ)/(∂q_j ∂v_k), so dtau_vq = dtau_vq_mod_fd^T
    double vq_err = (dtau_vq_mod - dtau_vq_mod_fd.transpose()).norm();

    std::cout << "dtau_vq_mod_diff = " << vq_err << std::endl;

    if (vq_err > std::sqrt(alpha)) {
      std::cout << "dtau_vq_mod_diff = " << vq_err << std::endl;
      throw std::runtime_error("dtau_vq_mod is not correct");
    }

    // ===================== dtau_qa = ∂²(λτ)/(∂q∂a) = ∂M_mod/∂q =====================
    print_pretty("mod ID SO derivatives: dtau_qa");

    // dtau_qa_mod(i,j) = ∂²(λτ)/(∂q_i ∂a_j) — computed analytically by SO algorithm
    // dtau_qa_mod_fd(j,k) = ∂M_mod_j/∂q_k = ∂²(λτ)/(∂a_j ∂q_k)
    // By Schwarz: dtau_qa_mod = dtau_qa_mod_fd^T
    double qa_err = (dtau_qa_mod - dtau_qa_mod_fd.transpose()).norm();

    std::cout << "dtau_qa_mod_diff = " << qa_err << std::endl;

    if (qa_err > std::sqrt(alpha)) {
      std::cout << "dtau_qa_mod_diff = " << qa_err << std::endl;
      throw std::runtime_error("dtau_qa_mod is not correct");
    }

  }

}

    return 0;
}
