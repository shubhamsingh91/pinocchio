// Author : Shubham Singh singh281@utexas.edu

#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/rnea-derivatives-faster.hpp"
#include "pinocchio/algorithm/mod-rnea-second-order-derivatives.hpp"
#include "pinocchio/algorithm/mod-rnea-derivatives.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/aba.hpp"
#include "pinocchio/algorithm/aba-derivatives.hpp"
#include "pinocchio/algorithm/mod-aba-derivatives.hpp"
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
  
  std::cout << "-------------------------------------------" << std::endl;
  std::cout << "-------------------------------------------" << std::endl;
  cout << "\nModel is" << robot_name << endl;
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
    MatrixXd dtau_aq_mod_fd(MatrixXd::Zero(model.nv, model.nv));

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
      dtau_aq_mod_fd.col(k) = (dtau_da_mod_plus - dtau_da_mod_orig)/alpha;
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

    // ===================== dtau_qv = dtau_dvq^T (Schwarz's theorem) =====================
    print_pretty("mod ID SO derivatives: dtau_qv");

    // dtau_dvq_mod stores ∂²(λτ)/(∂q_i ∂v_j), so dtau_qv = dtau_dvq_mod^T
    MatrixXd dtau_qv_mod = dtau_dvq_mod.transpose();

    // Verify with FD: perturb v_k, observe change in dtau_dq_mod
    MatrixXd dtau_qv_mod_fd(MatrixXd::Zero(model.nv, model.nv));
    VectorXd v_plus_qv(qdots[_smooth]);
    for(int k = 0; k < model.nv; ++k)
    {
      v_plus_qv[k] += alpha;
      computeModRNEADerivatives(model, data, qs[_smooth], v_plus_qv,
                                qddots[_smooth], lambdas[_smooth],
                                dtau_dq_mod_plus, dtau_dv_mod_plus, dtau_da_mod_plus);
      dtau_qv_mod_fd.col(k) = (dtau_dq_mod_plus - dtau_dq_mod_orig) / alpha;
      v_plus_qv[k] -= alpha;
    }

    // dtau_qv_mod_fd(j,k) = ∂²(λτ)/(∂q_j ∂v_k), so dtau_qv = dtau_qv_mod_fd^T
    double qv_err = (dtau_qv_mod - dtau_qv_mod_fd.transpose()).norm();

    std::cout << "dtau_qv_mod_diff = " << qv_err << std::endl;

    if (qv_err > std::sqrt(alpha)) {
      std::cout << "dtau_qv_mod_diff = " << qv_err << std::endl;
      throw std::runtime_error("dtau_qv_mod is not correct");
    }

    // ===================== dtau_qa = ∂²(λτ)/(∂q∂a) = ∂M_mod/∂q =====================
    print_pretty("mod ID SO derivatives: dtau_qa");

    // dtau_qa_mod(i,j) = ∂²(λτ)/(∂q_i ∂a_j) — computed analytically by SO algorithm
    // dtau_aq_mod_fd(j,k) = ∂M_mod_j/∂q_k = ∂²(λτ)/(∂a_j ∂q_k)
    // By Schwarz: dtau_qa_mod = dtau_aq_mod_fd^T
    double qa_err = (dtau_qa_mod - dtau_aq_mod_fd.transpose()).norm();

    std::cout << "dtau_qa_mod_diff = " << qa_err << std::endl;

    if (qa_err > std::sqrt(alpha)) {
      std::cout << "dtau_qa_mod_diff = " << qa_err << std::endl;
      throw std::runtime_error("dtau_qa_mod is not correct");
    }

    // ===================== Second-Order Forward Dynamics Derivatives =====================
    // Compute SO FD derivatives analytically from SO ID derivatives using chain rule,
    // and verify against finite differences of FO modABA derivatives.
    //
    // Key formulas (μ·ä = modaba, λ = M^{-1}μ, ä = FD(q,v,τ)):
    //   d²(μä)/dqdq = -d2tau_dqq - dqa · ∂ä/∂q - (∂ä/∂q)^T · dqa^T
    //   d²(μä)/dvdv = -d2tau_dvv
    //   d²(μä)/dqdv = -d2tau_dqv - dqa · ∂ä/∂v
    //   d²(μä)/dqτ  = -dqa · M^{-1}
    //   d²(μä)/dvτ  = 0
    //   d²(μä)/dττ  = 0

    // Relaxed threshold for SO FD: the analytical formula involves matrix products
    // that amplify FD truncation error, especially for larger models.
    const double fd_so_thr = 10 * std::sqrt(alpha);

    print_pretty("mod FD SO derivatives");

    // 1. Compute FD solution: a = aba(q, v, tau)
    VectorXd qddot_fd = aba(model, data, qs[_smooth], qdots[_smooth], taus[_smooth]);

    // 2. Compute ABA derivatives (gives ddq_dq, ddq_dv, Minv)
    computeABADerivatives(model, data, qs[_smooth], qdots[_smooth], taus[_smooth]);
    MatrixXd ddq_dq_mat = data.ddq_dq;
    MatrixXd ddq_dv_mat = data.ddq_dv;
    MatrixXd Minv_mat = data.Minv;
    Minv_mat.triangularView<Eigen::StrictlyLower>() =
      Minv_mat.transpose().triangularView<Eigen::StrictlyLower>();

    // 3. Compute lambda = M^{-1} * mu
    VectorXd lambda_fd = Minv_mat * mus[_smooth];

    // 4. Compute SO modID derivatives at (q, v, a_fd, lambda_fd)
    MatrixXd d2tau_dqq_fd(MatrixXd::Zero(model.nv, model.nv));
    MatrixXd d2tau_dvv_fd(MatrixXd::Zero(model.nv, model.nv));
    MatrixXd d2tau_dqv_fd(MatrixXd::Zero(model.nv, model.nv));
    MatrixXd d2tau_dqa_fd(MatrixXd::Zero(model.nv, model.nv));

    computeModRNEASecondOrderDerivatives(model, data, qs[_smooth], qdots[_smooth],
                                          qddot_fd, lambda_fd,
                                          d2tau_dqq_fd, d2tau_dvv_fd, d2tau_dqv_fd, d2tau_dqa_fd);

    // 5. Compute analytical SO FD derivatives
    MatrixXd d2qdd_dqq_ana = -d2tau_dqq_fd - d2tau_dqa_fd * ddq_dq_mat
                               - ddq_dq_mat.transpose() * d2tau_dqa_fd.transpose();
    MatrixXd d2qdd_dvv_ana = -d2tau_dvv_fd;
    MatrixXd d2qdd_dqv_ana = -d2tau_dqv_fd - d2tau_dqa_fd * ddq_dv_mat;
    MatrixXd d2qdd_dqtau_ana = -d2tau_dqa_fd * Minv_mat;

    // 6. Compute FD SO derivatives by perturbing FO modABA derivatives
    VectorXd dqdd_dq_fd_orig(VectorXd::Zero(model.nv));
    VectorXd dqdd_dv_fd_orig(VectorXd::Zero(model.nv));
    VectorXd dqdd_dtau_fd_orig(VectorXd::Zero(model.nv));

    computeModABADerivatives(model, data, qs[_smooth], qdots[_smooth], taus[_smooth], mus[_smooth],
                              dqdd_dq_fd_orig, dqdd_dv_fd_orig, dqdd_dtau_fd_orig);

    VectorXd dqdd_dq_fd_plus(VectorXd::Zero(model.nv));
    VectorXd dqdd_dv_fd_plus(VectorXd::Zero(model.nv));
    VectorXd dqdd_dtau_fd_plus(VectorXd::Zero(model.nv));

    // ---- Perturb q ----
    MatrixXd d2qdd_dqq_fdiff(MatrixXd::Zero(model.nv, model.nv));
    MatrixXd d2qdd_dvq_fdiff(MatrixXd::Zero(model.nv, model.nv));
    MatrixXd d2qdd_tauq_fdiff(MatrixXd::Zero(model.nv, model.nv));

    VectorXd v_eps_fd(VectorXd::Zero(model.nv));
    VectorXd q_plus_fd(model.nq);

    for(int k = 0; k < model.nv; ++k) {
      v_eps_fd[k] += alpha;
      pinocchio::integrate(model, qs[_smooth], v_eps_fd, q_plus_fd);
      computeModABADerivatives(model, data, q_plus_fd, qdots[_smooth], taus[_smooth], mus[_smooth],
                                dqdd_dq_fd_plus, dqdd_dv_fd_plus, dqdd_dtau_fd_plus);
      d2qdd_dqq_fdiff.col(k) = (dqdd_dq_fd_plus - dqdd_dq_fd_orig) / alpha;
      d2qdd_dvq_fdiff.col(k) = (dqdd_dv_fd_plus - dqdd_dv_fd_orig) / alpha;
      d2qdd_tauq_fdiff.col(k) = (dqdd_dtau_fd_plus - dqdd_dtau_fd_orig) / alpha;
      v_eps_fd[k] -= alpha;
    }

    double dqq_fd2_err = (d2qdd_dqq_ana - d2qdd_dqq_fdiff).norm();
    std::cout << "d2qdd_dqq_diff = " << dqq_fd2_err << std::endl;
    if (dqq_fd2_err > fd_so_thr) {
      throw std::runtime_error("d2qdd_dqq is not correct");
    }

    double dvq_fd2_err = (d2qdd_dqv_ana.transpose() - d2qdd_dvq_fdiff).norm();
    std::cout << "d2qdd_dvq_diff = " << dvq_fd2_err << std::endl;
    if (dvq_fd2_err > fd_so_thr) {
      throw std::runtime_error("d2qdd_dvq is not correct");
    }

    double tauq_fd2_err = (d2qdd_dqtau_ana.transpose() - d2qdd_tauq_fdiff).norm();
    std::cout << "d2qdd_tauq_diff = " << tauq_fd2_err << std::endl;
    if (tauq_fd2_err > fd_so_thr) {
      throw std::runtime_error("d2qdd_tauq is not correct");
    }

    // ---- Perturb v ----
    MatrixXd d2qdd_dqv_fdiff(MatrixXd::Zero(model.nv, model.nv));
    MatrixXd d2qdd_dvv_fdiff(MatrixXd::Zero(model.nv, model.nv));
    MatrixXd d2qdd_tauv_fdiff(MatrixXd::Zero(model.nv, model.nv));

    VectorXd v_plus_fd(qdots[_smooth]);

    for(int k = 0; k < model.nv; ++k) {
      v_plus_fd[k] += alpha;
      computeModABADerivatives(model, data, qs[_smooth], v_plus_fd, taus[_smooth], mus[_smooth],
                                dqdd_dq_fd_plus, dqdd_dv_fd_plus, dqdd_dtau_fd_plus);
      d2qdd_dqv_fdiff.col(k) = (dqdd_dq_fd_plus - dqdd_dq_fd_orig) / alpha;
      d2qdd_dvv_fdiff.col(k) = (dqdd_dv_fd_plus - dqdd_dv_fd_orig) / alpha;
      d2qdd_tauv_fdiff.col(k) = (dqdd_dtau_fd_plus - dqdd_dtau_fd_orig) / alpha;
      v_plus_fd[k] -= alpha;
    }

    double dqv_fd2_err = (d2qdd_dqv_ana - d2qdd_dqv_fdiff).norm();
    std::cout << "d2qdd_dqv_diff = " << dqv_fd2_err << std::endl;
    if (dqv_fd2_err > fd_so_thr) {
      throw std::runtime_error("d2qdd_dqv is not correct");
    }

    double dvv_fd2_err = (d2qdd_dvv_ana - d2qdd_dvv_fdiff).norm();
    std::cout << "d2qdd_dvv_diff = " << dvv_fd2_err << std::endl;
    if (dvv_fd2_err > fd_so_thr) {
      throw std::runtime_error("d2qdd_dvv is not correct");
    }

    double tauv_fd2_err = d2qdd_tauv_fdiff.norm();
    std::cout << "d2qdd_tauv_diff (expect ~0) = " << tauv_fd2_err << std::endl;
    if (tauv_fd2_err > fd_so_thr) {
      throw std::runtime_error("d2qdd_tauv is not zero");
    }

    // ---- Perturb tau ----
    MatrixXd d2qdd_dqtau_fdiff(MatrixXd::Zero(model.nv, model.nv));
    MatrixXd d2qdd_dvtau_fdiff(MatrixXd::Zero(model.nv, model.nv));
    MatrixXd d2qdd_tautau_fdiff(MatrixXd::Zero(model.nv, model.nv));

    VectorXd tau_plus_fd(taus[_smooth]);

    for(int k = 0; k < model.nv; ++k) {
      tau_plus_fd[k] += alpha;
      computeModABADerivatives(model, data, qs[_smooth], qdots[_smooth], tau_plus_fd, mus[_smooth],
                                dqdd_dq_fd_plus, dqdd_dv_fd_plus, dqdd_dtau_fd_plus);
      d2qdd_dqtau_fdiff.col(k) = (dqdd_dq_fd_plus - dqdd_dq_fd_orig) / alpha;
      d2qdd_dvtau_fdiff.col(k) = (dqdd_dv_fd_plus - dqdd_dv_fd_orig) / alpha;
      d2qdd_tautau_fdiff.col(k) = (dqdd_dtau_fd_plus - dqdd_dtau_fd_orig) / alpha;
      tau_plus_fd[k] -= alpha;
    }

    double dqtau_fd2_err = (d2qdd_dqtau_ana - d2qdd_dqtau_fdiff).norm();
    std::cout << "d2qdd_dqtau_diff = " << dqtau_fd2_err << std::endl;
    if (dqtau_fd2_err > fd_so_thr) {
      throw std::runtime_error("d2qdd_dqtau is not correct");
    }

    double dvtau_fd2_err = d2qdd_dvtau_fdiff.norm();
    std::cout << "d2qdd_dvtau_diff (expect ~0) = " << dvtau_fd2_err << std::endl;
    if (dvtau_fd2_err > fd_so_thr) {
      throw std::runtime_error("d2qdd_dvtau is not zero");
    }

    double tautau_fd2_err = d2qdd_tautau_fdiff.norm();
    std::cout << "d2qdd_tautau_diff (expect ~0) = " << tautau_fd2_err << std::endl;
    if (tautau_fd2_err > fd_so_thr) {
      throw std::runtime_error("d2qdd_tautau is not zero");
    }

  }

}

    return 0;
}
