/* bench_modFD_SO.cpp
 * Modified FD Second-Order Derivatives Benchmark
 *
 * Compares 4 approaches for computing the Hessian of the scalar mu*qddot(q,v,tau):
 *
 * Case 1:  Full SO AD — trace modaba() with CasADi, take Hessian (2 AD diffs)
 * Case 2a: FO AD over full FO — trace computeABADerivativesFaster(), contract with mu, Jacobian (1 AD diff)
 * Case 2b: FO AD over mod FO — trace computeModABADerivatives(), Jacobian (1 AD diff)
 * Case 3:  Full analytical — chain-rule via computeModRNEASecondOrderDerivatives() + computeABADerivatives()
 *
 * All 4 produce the same 4 nv x nv matrices: dqq, dvv, dqv, dqtau
 */

#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/aba.hpp"
#include "pinocchio/algorithm/modaba.hpp"
#include "pinocchio/algorithm/aba-derivatives-faster.hpp"
#include "pinocchio/algorithm/aba-derivatives.hpp"
#include "pinocchio/algorithm/mod-aba-derivatives.hpp"
#include "pinocchio/algorithm/mod-rnea-second-order-derivatives.hpp"
#include "pinocchio/codegen/code-generator-algo.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/container/aligned-vector.hpp"
#include "pinocchio/utils/timer.hpp"

#include <casadi/casadi.hpp>
#include "pinocchio/autodiff/casadi.hpp"

#include <iostream>
#include <fstream>

using namespace std;
using namespace Eigen;
using namespace pinocchio;
using namespace pinocchio::casadi;

int main(int /*argc*/, const char* /*argv*/[])
{
    PinocchioTicToc timer(PinocchioTicToc::US);

#ifdef NDEBUG
    int NBT_BASE = 100000;
#else
    int NBT_BASE = 1;
    std::cout << "(the time score in debug mode is not relevant)" << std::endl;
#endif

    int n_models = 5;
    string str_robotname[] = {
        "double_pendulum",
        "ur3_robot",
        "hyq",
        "atlas",
        "talos_full_v2"
    };

    for (int mm = 0; mm < n_models; mm++) {

        Model model;
        bool with_ff = false;

        if ((mm == 2) || (mm == 3) || (mm == 4)) {
            with_ff = true;
        }

        string robot_name = str_robotname[mm];
        string filename = "../models/" + robot_name + string(".urdf");

        if (with_ff)
            pinocchio::urdf::buildModel(filename, JointModelFreeFlyer(), model);
        else
            pinocchio::urdf::buildModel(filename, model);

        if (with_ff)
            robot_name += string("_f");

        std::cout << "\n============================================" << std::endl;
        std::cout << "Model: " << robot_name << "  nq=" << model.nq << "  nv=" << model.nv << std::endl;
        std::cout << "============================================" << std::endl;

        Data data(model);
        typedef Model::ConfigVectorType ConfigVector;
        typedef Model::TangentVectorType TangentVector;
        const int nv = model.nv;
        const int nq = model.nq;

        // FD CasADi traces are very expensive for large models;
        // use fewer iterations but still enough for stable averages
        int NBT = NBT_BASE;
        if (nv > 20) NBT = std::min(NBT, 10000);
        if (nv > 40) NBT = std::min(NBT, 5000);

        VectorXd qmax = VectorXd::Ones(nq);

        // Random test data
        PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) qs(NBT);
        PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) qdots(NBT);
        PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) taus(NBT);
        PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) mus(NBT);

        for (size_t i = 0; i < (size_t)NBT; ++i) {
            qs[i] = randomConfiguration(model, -qmax, qmax);
            qdots[i] = VectorXd::Random(nv);
            taus[i] = VectorXd::Random(nv);
            mus[i] = VectorXd::Random(nv);
        }

        int n_cases = 4;
        double time_cases[n_cases];

        // ================================================================
        // CasADi setup (shared by cases 1, 2a, 2b)
        // ================================================================

        typedef ::casadi::SX ADcScalar;
        typedef pinocchio::ModelTpl<ADcScalar> ADcModel;
        typedef ADcModel::Data ADcData;
        typedef ADcModel::ConfigVectorType ConfigVectorAD;
        typedef ADcModel::TangentVectorType TangentVectorAD;

        ADcModel adc_model = model.cast<ADcScalar>();
        ADcData adc_data(adc_model);

        // Symbolic variables
        ::casadi::SX cs_q = ::casadi::SX::sym("q", nq);
        ::casadi::SX cs_v_int = ::casadi::SX::sym("v_inc", nv);
        ::casadi::SX cs_v = ::casadi::SX::sym("v", nv);
        ::casadi::SX cs_tau = ::casadi::SX::sym("tau", nv);

        // q_int = integrate(q, v_int)
        ConfigVectorAD q_ad(nq), v_int_ad(nv), q_int_ad(nq);
        q_ad = Eigen::Map<ConfigVectorAD>(static_cast<std::vector<ADcScalar>>(cs_q).data(), nq, 1);
        v_int_ad = Eigen::Map<ConfigVectorAD>(static_cast<std::vector<ADcScalar>>(cs_v_int).data(), nv, 1);
        pinocchio::integrate(adc_model, q_ad, v_int_ad, q_int_ad);

        TangentVectorAD v_ad(nv);
        v_ad = Eigen::Map<TangentVectorAD>(static_cast<std::vector<ADcScalar>>(cs_v).data(), nv, 1);

        TangentVectorAD tau_ad(nv);
        tau_ad = Eigen::Map<TangentVectorAD>(static_cast<std::vector<ADcScalar>>(cs_tau).data(), nv, 1);

        // Mu as symbolic variable
        ::casadi::SX cs_mu = ::casadi::SX::sym("mu", nv);
        TangentVectorAD mu_ad(nv);
        mu_ad = Eigen::Map<TangentVectorAD>(static_cast<std::vector<ADcScalar>>(cs_mu).data(), nv, 1);

        // Evaluation vectors
        std::vector<double> q_vec((size_t)nq);
        std::vector<double> v_int_vec((size_t)nv, 0.0);
        std::vector<double> v_vec((size_t)nv);
        std::vector<double> tau_vec((size_t)nv);
        std::vector<double> mu_vec((size_t)nv);

        // ================================================================
        // Case 1: Full SO AD — trace modaba, take Hessian
        // ================================================================
        std::cout << "\n--- Case 1: Full SO AD (trace modaba) ---" << std::endl;

        ADcData adc_data1(adc_model);
        ADcScalar cs_modqdd = modaba(adc_model, adc_data1, q_int_ad, v_ad, tau_ad, mu_ad);

        // Hessian blocks via 2 jacobian calls each
        ::casadi::SX grad_q_1 = jacobian(cs_modqdd, cs_v_int);   // 1 x nv
        ::casadi::SX hess_qq_1 = jacobian(grad_q_1, cs_v_int);   // nv x nv: d2f/(dq_i dq_j)

        ::casadi::SX grad_v_1 = jacobian(cs_modqdd, cs_v);       // 1 x nv
        ::casadi::SX hess_vv_1 = jacobian(grad_v_1, cs_v);       // nv x nv: d2f/(dv_i dv_j)

        // dqv: d2f/(dq_i dv_j) — diff q-gradient w.r.t. v
        ::casadi::SX hess_qv_1 = jacobian(grad_q_1, cs_v);       // nv x nv
        // dqtau: d2f/(dq_i dtau_j) — diff q-gradient w.r.t. tau
        ::casadi::SX hess_qtau_1 = jacobian(grad_q_1, cs_tau);   // nv x nv

        ::casadi::Function eval_case1(
            "case1_modaba_SO",
            ::casadi::SXVector {cs_q, cs_v_int, cs_v, cs_tau, cs_mu},
            ::casadi::SXVector {hess_qq_1, hess_vv_1, hess_qv_1, hess_qtau_1});

        // ================================================================
        // Case 2a: FO AD over computeABADerivativesFaster + contract
        // ================================================================
        std::cout << "--- Case 2a: FO AD over full FO derivs ---" << std::endl;

        ADcData adc_data2a(adc_model);
        pinocchio::computeABADerivativesFaster(adc_model, adc_data2a, q_int_ad, v_ad, tau_ad);
        // Symmetrize Minv
        (adc_data2a.Minv).template triangularView<Eigen::StrictlyLower>()
            = (adc_data2a.Minv).transpose().template triangularView<Eigen::StrictlyLower>();

        // Contract with mu: g_q(i) = sum_j mu(j) * ddq_dq(j,i)
        ::casadi::SX g_q_2a(nv, 1), g_v_2a(nv, 1), g_tau_2a(nv, 1);
        for (int i = 0; i < nv; i++) {
            g_q_2a(i) = 0; g_v_2a(i) = 0; g_tau_2a(i) = 0;
            for (int j = 0; j < nv; j++) {
                g_q_2a(i) += cs_mu(j) * adc_data2a.ddq_dq(j, i);
                g_v_2a(i) += cs_mu(j) * adc_data2a.ddq_dv(j, i);
                g_tau_2a(i) += cs_mu(j) * adc_data2a.Minv(j, i);
            }
        }

        ::casadi::SX hess_qq_2a = jacobian(g_q_2a, cs_v_int);
        ::casadi::SX hess_vv_2a = jacobian(g_v_2a, cs_v);
        ::casadi::SX hess_qv_2a = jacobian(g_q_2a, cs_v);         // d2f/(dq_i dv_j)
        ::casadi::SX hess_qtau_2a = jacobian(g_q_2a, cs_tau);     // d2f/(dq_i dtau_j)

        ::casadi::Function eval_case2a(
            "case2a_ABAFO_contract",
            ::casadi::SXVector {cs_q, cs_v_int, cs_v, cs_tau, cs_mu},
            ::casadi::SXVector {hess_qq_2a, hess_vv_2a, hess_qv_2a, hess_qtau_2a});

        // ================================================================
        // Case 2b: FO AD over computeModABADerivatives
        // ================================================================
        std::cout << "--- Case 2b: FO AD over mod FO derivs ---" << std::endl;

        ADcData adc_data2b(adc_model);
        pinocchio::computeModABADerivatives(adc_model, adc_data2b, q_int_ad, v_ad, tau_ad, mu_ad);

        // Extract gradient vectors from data
        ::casadi::SX g_q_2b(nv, 1), g_v_2b(nv, 1), g_tau_2b(nv, 1);
        for (int i = 0; i < nv; i++) {
            g_q_2b(i) = adc_data2b.ddq_dq_mod[i];
            g_v_2b(i) = adc_data2b.ddq_dv_mod[i];
            g_tau_2b(i) = adc_data2b.ddq_dtau_mod[i];
        }

        ::casadi::SX hess_qq_2b = jacobian(g_q_2b, cs_v_int);
        ::casadi::SX hess_vv_2b = jacobian(g_v_2b, cs_v);
        ::casadi::SX hess_qv_2b = jacobian(g_q_2b, cs_v);         // d2f/(dq_i dv_j)
        ::casadi::SX hess_qtau_2b = jacobian(g_q_2b, cs_tau);     // d2f/(dq_i dtau_j)

        ::casadi::Function eval_case2b(
            "case2b_modABAFO",
            ::casadi::SXVector {cs_q, cs_v_int, cs_v, cs_tau, cs_mu},
            ::casadi::SXVector {hess_qq_2b, hess_vv_2b, hess_qv_2b, hess_qtau_2b});

        // ================================================================
        // ACCURACY CHECK (single evaluation)
        // ================================================================
        std::cout << "\n--- Accuracy check ---" << std::endl;

        // Set up evaluation point
        Eigen::Map<ConfigVector>(q_vec.data(), nq, 1) = qs[0];
        Eigen::Map<TangentVector>(v_vec.data(), nv, 1) = qdots[0];
        Eigen::Map<TangentVector>(tau_vec.data(), nv, 1) = taus[0];
        Eigen::Map<TangentVector>(mu_vec.data(), nv, 1) = mus[0];

        // Case 3: Analytical reference via chain rule
        // Step 1: FD solution a = aba(q,v,tau)
        VectorXd qddot_fd = aba(model, data, qs[0], qdots[0], taus[0]);

        // Step 2: ABA derivatives → ddq_dq, ddq_dv, Minv
        computeABADerivatives(model, data, qs[0], qdots[0], taus[0]);
        MatrixXd ddq_dq_mat = data.ddq_dq;
        MatrixXd ddq_dv_mat = data.ddq_dv;
        MatrixXd Minv_mat = data.Minv;
        Minv_mat.triangularView<Eigen::StrictlyLower>() =
            Minv_mat.transpose().triangularView<Eigen::StrictlyLower>();

        // Step 3: lambda = M^{-1} * mu
        VectorXd lambda_fd = Minv_mat * mus[0];

        // Step 4: SO modID derivatives at (q, v, a_fd, lambda_fd)
        MatrixXd d2tau_dqq(MatrixXd::Zero(nv, nv));
        MatrixXd d2tau_dvv(MatrixXd::Zero(nv, nv));
        MatrixXd d2tau_dqv(MatrixXd::Zero(nv, nv));
        MatrixXd d2tau_dqa(MatrixXd::Zero(nv, nv));
        computeModRNEASecondOrderDerivatives(model, data, qs[0], qdots[0],
            qddot_fd, lambda_fd, d2tau_dqq, d2tau_dvv, d2tau_dqv, d2tau_dqa);

        // Step 5: Chain-rule formulas
        MatrixXd dqq_ana = -d2tau_dqq - d2tau_dqa * ddq_dq_mat
                            - ddq_dq_mat.transpose() * d2tau_dqa.transpose();
        MatrixXd dvv_ana = -d2tau_dvv;
        MatrixXd dqv_ana = -d2tau_dqv - d2tau_dqa * ddq_dv_mat;
        MatrixXd dqtau_ana = -d2tau_dqa * Minv_mat;

        // Helper to extract nv x nv from CasADi result
        auto casadi_to_eigen = [&](const ::casadi::DM& dm, int rows, int cols) -> MatrixXd {
            std::vector<double> v = static_cast<std::vector<double>>(dm);
            return Eigen::Map<MatrixXd>(v.data(), rows, cols);
        };

        // Case 1 evaluation
        auto res1 = eval_case1(::casadi::DMVector {q_vec, v_int_vec, v_vec, tau_vec, mu_vec});
        MatrixXd dqq_1 = casadi_to_eigen(res1[0], nv, nv);
        MatrixXd dvv_1 = casadi_to_eigen(res1[1], nv, nv);
        MatrixXd dqv_1 = casadi_to_eigen(res1[2], nv, nv);
        MatrixXd dqtau_1 = casadi_to_eigen(res1[3], nv, nv);

        // Case 2a evaluation
        auto res2a = eval_case2a(::casadi::DMVector {q_vec, v_int_vec, v_vec, tau_vec, mu_vec});
        MatrixXd dqq_2a = casadi_to_eigen(res2a[0], nv, nv);
        MatrixXd dvv_2a = casadi_to_eigen(res2a[1], nv, nv);
        MatrixXd dqv_2a = casadi_to_eigen(res2a[2], nv, nv);
        MatrixXd dqtau_2a = casadi_to_eigen(res2a[3], nv, nv);

        // Case 2b evaluation
        auto res2b = eval_case2b(::casadi::DMVector {q_vec, v_int_vec, v_vec, tau_vec, mu_vec});
        MatrixXd dqq_2b = casadi_to_eigen(res2b[0], nv, nv);
        MatrixXd dvv_2b = casadi_to_eigen(res2b[1], nv, nv);
        MatrixXd dqv_2b = casadi_to_eigen(res2b[2], nv, nv);
        MatrixXd dqtau_2b = casadi_to_eigen(res2b[3], nv, nv);

        // Print accuracy
        std::cout << "Case 1 vs 3 (full SO AD vs analytical):" << std::endl;
        std::cout << "  dqq:   " << (dqq_1 - dqq_ana).norm() << std::endl;
        std::cout << "  dvv:   " << (dvv_1 - dvv_ana).norm() << std::endl;
        std::cout << "  dqv:   " << (dqv_1 - dqv_ana).norm() << std::endl;
        std::cout << "  dqtau: " << (dqtau_1 - dqtau_ana).norm() << std::endl;

        std::cout << "Case 2a vs 3 (AD over full FO vs analytical):" << std::endl;
        std::cout << "  dqq:   " << (dqq_2a - dqq_ana).norm() << std::endl;
        std::cout << "  dvv:   " << (dvv_2a - dvv_ana).norm() << std::endl;
        std::cout << "  dqv:   " << (dqv_2a - dqv_ana).norm() << std::endl;
        std::cout << "  dqtau: " << (dqtau_2a - dqtau_ana).norm() << std::endl;

        std::cout << "Case 2b vs 3 (AD over mod FO vs analytical):" << std::endl;
        std::cout << "  dqq:   " << (dqq_2b - dqq_ana).norm() << std::endl;
        std::cout << "  dvv:   " << (dvv_2b - dvv_ana).norm() << std::endl;
        std::cout << "  dqv:   " << (dqv_2b - dqv_ana).norm() << std::endl;
        std::cout << "  dqtau: " << (dqtau_2b - dqtau_ana).norm() << std::endl;

        // ================================================================
        // TIMING
        // ================================================================
        std::cout << "\n--- Timing (" << NBT << " iterations) ---" << std::endl;

        // Case 3: Analytical (chain rule)
        timer.tic();
        SMOOTH(NBT)
        {
            // aba for qddot
            VectorXd qddot_t = aba(model, data, qs[_smooth], qdots[_smooth], taus[_smooth]);
            // ABA derivatives for ddq_dq, ddq_dv, Minv
            computeABADerivatives(model, data, qs[_smooth], qdots[_smooth], taus[_smooth]);
            MatrixXd Minv_t = data.Minv;
            Minv_t.triangularView<Eigen::StrictlyLower>() =
                Minv_t.transpose().triangularView<Eigen::StrictlyLower>();
            VectorXd lambda_t = Minv_t * mus[_smooth];
            // SO modID derivatives
            d2tau_dqq.setZero(); d2tau_dvv.setZero(); d2tau_dqv.setZero(); d2tau_dqa.setZero();
            computeModRNEASecondOrderDerivatives(model, data, qs[_smooth], qdots[_smooth],
                qddot_t, lambda_t, d2tau_dqq, d2tau_dvv, d2tau_dqv, d2tau_dqa);
            // Chain-rule
            MatrixXd dqq_t = -d2tau_dqq - d2tau_dqa * data.ddq_dq
                              - data.ddq_dq.transpose() * d2tau_dqa.transpose();
            MatrixXd dvv_t = -d2tau_dvv;
            MatrixXd dqv_t = -d2tau_dqv - d2tau_dqa * data.ddq_dv;
            MatrixXd dqtau_t = -d2tau_dqa * Minv_t;
        }
        time_cases[0] = timer.toc() / NBT;
        std::cout << "Case 3  (analytical):          " << time_cases[0] << " us" << std::endl;

        // Case 1: Full SO AD
        timer.tic();
        SMOOTH(NBT)
        {
            Eigen::Map<ConfigVector>(q_vec.data(), nq, 1) = qs[_smooth];
            Eigen::Map<TangentVector>(v_vec.data(), nv, 1) = qdots[_smooth];
            Eigen::Map<TangentVector>(tau_vec.data(), nv, 1) = taus[_smooth];
            Eigen::Map<TangentVector>(mu_vec.data(), nv, 1) = mus[_smooth];

            auto res = eval_case1(::casadi::DMVector {q_vec, v_int_vec, v_vec, tau_vec, mu_vec});
        }
        time_cases[1] = timer.toc() / NBT;
        std::cout << "Case 1  (full SO AD):          " << time_cases[1] << " us" << std::endl;

        // Case 2a: FO AD over full FO derivs
        timer.tic();
        SMOOTH(NBT)
        {
            Eigen::Map<ConfigVector>(q_vec.data(), nq, 1) = qs[_smooth];
            Eigen::Map<TangentVector>(v_vec.data(), nv, 1) = qdots[_smooth];
            Eigen::Map<TangentVector>(tau_vec.data(), nv, 1) = taus[_smooth];
            Eigen::Map<TangentVector>(mu_vec.data(), nv, 1) = mus[_smooth];

            auto res = eval_case2a(::casadi::DMVector {q_vec, v_int_vec, v_vec, tau_vec, mu_vec});
        }
        time_cases[2] = timer.toc() / NBT;
        std::cout << "Case 2a (AD over full FO):     " << time_cases[2] << " us" << std::endl;

        // Case 2b: FO AD over mod FO derivs
        timer.tic();
        SMOOTH(NBT)
        {
            Eigen::Map<ConfigVector>(q_vec.data(), nq, 1) = qs[_smooth];
            Eigen::Map<TangentVector>(v_vec.data(), nv, 1) = qdots[_smooth];
            Eigen::Map<TangentVector>(tau_vec.data(), nv, 1) = taus[_smooth];
            Eigen::Map<TangentVector>(mu_vec.data(), nv, 1) = mus[_smooth];

            auto res = eval_case2b(::casadi::DMVector {q_vec, v_int_vec, v_vec, tau_vec, mu_vec});
        }
        time_cases[3] = timer.toc() / NBT;
        std::cout << "Case 2b (AD over mod FO):      " << time_cases[3] << " us" << std::endl;

        // Speedups
        std::cout << "\nSpeedups vs analytical:" << std::endl;
        std::cout << "  Case 1  / Case 3 = " << time_cases[1] / time_cases[0] << "x" << std::endl;
        std::cout << "  Case 2a / Case 3 = " << time_cases[2] / time_cases[0] << "x" << std::endl;
        std::cout << "  Case 2b / Case 3 = " << time_cases[3] / time_cases[0] << "x" << std::endl;

        // Write timing data to file
        // Format: case3(analytical), case1(full SO AD), case2a(AD full FO), case2b(AD mod FO)
        string outdir = "data/modFD_SO/";
        string outfile = outdir + robot_name + ".txt";
        std::ofstream ofs(outfile);
        if (ofs.is_open()) {
            ofs << time_cases[0] << std::endl;
            ofs << time_cases[1] << std::endl;
            ofs << time_cases[2] << std::endl;
            ofs << time_cases[3] << std::endl;
            ofs.close();
            std::cout << "Timing saved to " << outfile << std::endl;
        } else {
            std::cerr << "Warning: could not open " << outfile << " for writing" << std::endl;
        }

    } // end model loop

    return 0;
}
