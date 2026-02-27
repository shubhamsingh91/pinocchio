/* bench_modFD_SO_accuracy.cpp
 * Accuracy check for modified FD second-order derivatives.
 *
 * Compares 3 CasADi approaches against the analytical chain-rule reference:
 *   Case 1:  Full SO AD — trace modaba(), Hessian (2 AD diffs)
 *   Case 2a: FO AD over full FO — trace computeABADerivativesFaster(), contract, Jacobian (1 AD diff)
 *   Case 2b: FO AD over mod FO — trace computeModABADerivatives(), Jacobian (1 AD diff)
 *   Case 3:  Analytical chain-rule from computeModRNEASecondOrderDerivatives() + computeABADerivatives()
 *   Case 4:  Full SO tensors (ComputeRNEASecondOrderDerivatives) + lambda contraction + FD chain-rule vs Case 3
 */

#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/aba.hpp"
#include "pinocchio/algorithm/modaba.hpp"
#include "pinocchio/algorithm/aba-derivatives-faster.hpp"
#include "pinocchio/algorithm/aba-derivatives.hpp"
#include "pinocchio/algorithm/mod-aba-derivatives.hpp"
#include "pinocchio/algorithm/mod-rnea-second-order-derivatives.hpp"
#include "pinocchio/algorithm/rnea-second-order-derivatives.hpp"
#include "pinocchio/codegen/code-generator-algo.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/container/aligned-vector.hpp"

#include <casadi/casadi.hpp>
#include "pinocchio/autodiff/casadi.hpp"

#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>
#include <limits>

using namespace std;
using namespace Eigen;
using namespace pinocchio;
using namespace pinocchio::casadi;

int main()
{
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
        bool with_ff = (mm == 2) || (mm == 3) || (mm == 4);

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
        const int nv = model.nv;
        const int nq = model.nq;

        VectorXd qmax = VectorXd::Ones(nq);
        VectorXd q = randomConfiguration(model, -qmax, qmax);
        VectorXd v = VectorXd::Random(nv);
        VectorXd tau = VectorXd::Random(nv);
        VectorXd mu = VectorXd::Random(nv);

        // ================================================================
        // CasADi symbolic setup
        // ================================================================
        typedef ::casadi::SX ADcScalar;
        typedef pinocchio::ModelTpl<ADcScalar> ADcModel;
        typedef ADcModel::Data ADcData;
        typedef ADcModel::ConfigVectorType ConfigVectorAD;
        typedef ADcModel::TangentVectorType TangentVectorAD;

        ADcModel adc_model = model.cast<ADcScalar>();

        ::casadi::SX cs_q = ::casadi::SX::sym("q", nq);
        ::casadi::SX cs_v_int = ::casadi::SX::sym("v_inc", nv);
        ::casadi::SX cs_v = ::casadi::SX::sym("v", nv);
        ::casadi::SX cs_tau = ::casadi::SX::sym("tau", nv);
        ::casadi::SX cs_mu = ::casadi::SX::sym("mu", nv);

        ConfigVectorAD q_ad(nq), v_int_ad(nv), q_int_ad(nq);
        q_ad = Eigen::Map<ConfigVectorAD>(static_cast<std::vector<ADcScalar>>(cs_q).data(), nq, 1);
        v_int_ad = Eigen::Map<ConfigVectorAD>(static_cast<std::vector<ADcScalar>>(cs_v_int).data(), nv, 1);
        pinocchio::integrate(adc_model, q_ad, v_int_ad, q_int_ad);

        TangentVectorAD v_ad(nv), tau_ad(nv), mu_ad(nv);
        v_ad = Eigen::Map<TangentVectorAD>(static_cast<std::vector<ADcScalar>>(cs_v).data(), nv, 1);
        tau_ad = Eigen::Map<TangentVectorAD>(static_cast<std::vector<ADcScalar>>(cs_tau).data(), nv, 1);
        mu_ad = Eigen::Map<TangentVectorAD>(static_cast<std::vector<ADcScalar>>(cs_mu).data(), nv, 1);

        ::casadi::SXVector cs_inputs = {cs_q, cs_v_int, cs_v, cs_tau, cs_mu};

        // ================================================================
        // Case 1: Full SO AD — trace modaba, take Hessian
        // ================================================================
        ADcData adc_data1(adc_model);
        ADcScalar cs_modqdd = modaba(adc_model, adc_data1, q_int_ad, v_ad, tau_ad, mu_ad);

        ::casadi::SX grad_q_1 = jacobian(cs_modqdd, cs_v_int);
        ::casadi::SX grad_v_1 = jacobian(cs_modqdd, cs_v);

        ::casadi::Function eval_case1(
            "case1", cs_inputs,
            ::casadi::SXVector {
                jacobian(grad_q_1, cs_v_int),  // dqq
                jacobian(grad_v_1, cs_v),      // dvv
                jacobian(grad_q_1, cs_v),      // dqv
                jacobian(grad_q_1, cs_tau)     // dqtau
            });

        // ================================================================
        // Case 2a: FO AD over computeABADerivativesFaster + contract
        // ================================================================
        ADcData adc_data2a(adc_model);
        pinocchio::computeABADerivativesFaster(adc_model, adc_data2a, q_int_ad, v_ad, tau_ad);
        (adc_data2a.Minv).template triangularView<Eigen::StrictlyLower>()
            = (adc_data2a.Minv).transpose().template triangularView<Eigen::StrictlyLower>();

        ::casadi::SX g_q_2a(nv, 1), g_v_2a(nv, 1);
        for (int i = 0; i < nv; i++) {
            g_q_2a(i) = 0; g_v_2a(i) = 0;
            for (int j = 0; j < nv; j++) {
                g_q_2a(i) += cs_mu(j) * adc_data2a.ddq_dq(j, i);
                g_v_2a(i) += cs_mu(j) * adc_data2a.ddq_dv(j, i);
            }
        }

        ::casadi::Function eval_case2a(
            "case2a", cs_inputs,
            ::casadi::SXVector {
                jacobian(g_q_2a, cs_v_int),  // dqq
                jacobian(g_v_2a, cs_v),      // dvv
                jacobian(g_q_2a, cs_v),      // dqv
                jacobian(g_q_2a, cs_tau)     // dqtau
            });

        // ================================================================
        // Case 2b: FO AD over computeModABADerivatives
        // ================================================================
        ADcData adc_data2b(adc_model);
        pinocchio::computeModABADerivatives(adc_model, adc_data2b, q_int_ad, v_ad, tau_ad, mu_ad);

        ::casadi::SX g_q_2b(nv, 1), g_v_2b(nv, 1);
        for (int i = 0; i < nv; i++) {
            g_q_2b(i) = adc_data2b.ddq_dq_mod[i];
            g_v_2b(i) = adc_data2b.ddq_dv_mod[i];
        }

        ::casadi::Function eval_case2b(
            "case2b", cs_inputs,
            ::casadi::SXVector {
                jacobian(g_q_2b, cs_v_int),  // dqq
                jacobian(g_v_2b, cs_v),      // dvv
                jacobian(g_q_2b, cs_v),      // dqv
                jacobian(g_q_2b, cs_tau)     // dqtau
            });

        // ================================================================
        // Evaluate at a single point
        // ================================================================
        std::vector<double> q_vec((size_t)nq);
        std::vector<double> v_int_vec((size_t)nv, 0.0);
        std::vector<double> v_vec((size_t)nv);
        std::vector<double> tau_vec((size_t)nv);
        std::vector<double> mu_vec((size_t)nv);

        Eigen::Map<Model::ConfigVectorType>(q_vec.data(), nq, 1) = q;
        Eigen::Map<Model::TangentVectorType>(v_vec.data(), nv, 1) = v;
        Eigen::Map<Model::TangentVectorType>(tau_vec.data(), nv, 1) = tau;
        Eigen::Map<Model::TangentVectorType>(mu_vec.data(), nv, 1) = mu;

        ::casadi::DMVector input = {q_vec, v_int_vec, v_vec, tau_vec, mu_vec};

        auto casadi_to_eigen = [&](const ::casadi::DM& dm) -> MatrixXd {
            std::vector<double> vals = static_cast<std::vector<double>>(dm);
            return Eigen::Map<MatrixXd>(vals.data(), nv, nv);
        };

        // Case 3: Analytical chain-rule reference
        VectorXd qddot_fd = aba(model, data, q, v, tau);
        computeABADerivatives(model, data, q, v, tau);
        MatrixXd ddq_dq_mat = data.ddq_dq;
        MatrixXd ddq_dv_mat = data.ddq_dv;
        MatrixXd Minv_mat = data.Minv;
        Minv_mat.triangularView<Eigen::StrictlyLower>() =
            Minv_mat.transpose().triangularView<Eigen::StrictlyLower>();
        VectorXd lambda_fd = Minv_mat * mu;

        MatrixXd d2tau_dqq(MatrixXd::Zero(nv, nv)), d2tau_dvv(MatrixXd::Zero(nv, nv));
        MatrixXd d2tau_dqv(MatrixXd::Zero(nv, nv)), d2tau_dqa(MatrixXd::Zero(nv, nv));
        computeModRNEASecondOrderDerivatives(model, data, q, v,
            qddot_fd, lambda_fd, d2tau_dqq, d2tau_dvv, d2tau_dqv, d2tau_dqa);

        MatrixXd ana[] = {
            -d2tau_dqq - d2tau_dqa * ddq_dq_mat - ddq_dq_mat.transpose() * d2tau_dqa.transpose(),  // dqq
            -d2tau_dvv,                                                                              // dvv
            -d2tau_dqv - d2tau_dqa * ddq_dv_mat,                                                    // dqv
            -d2tau_dqa * Minv_mat                                                                    // dqtau
        };

        // Case 4: Full SO tensors + lambda contraction + FD chain-rule
        Eigen::Tensor<double, 3> d2tau_dqdq_t(nv, nv, nv);
        Eigen::Tensor<double, 3> d2tau_dvdv_t(nv, nv, nv);
        Eigen::Tensor<double, 3> d2tau_dqdv_t(nv, nv, nv);
        Eigen::Tensor<double, 3> d2tau_dadq_t(nv, nv, nv);
        d2tau_dqdq_t.setZero();
        d2tau_dvdv_t.setZero();
        d2tau_dqdv_t.setZero();
        d2tau_dadq_t.setZero();

        ComputeRNEASecondOrderDerivatives(model, data, q, v, qddot_fd,
            d2tau_dqdq_t, d2tau_dvdv_t, d2tau_dqdv_t, d2tau_dadq_t);

        // Contract each ID SO tensor with lambda_fd: result(i,j) = sum_k lambda_fd(k) * tensor(k,i,j)
        // Note: d2tau_dadq_t(k,i,j) = d²τ_k/(da_i dq_j), contraction gives d²(λ·τ)/(da_i dq_j)
        //   → transpose to get d²(λ·τ)/(dq_i da_j) matching dqa convention
        MatrixXd dqq_t(MatrixXd::Zero(nv, nv)), dvv_t(MatrixXd::Zero(nv, nv));
        MatrixXd dqv_t(MatrixXd::Zero(nv, nv)), dqa_t(MatrixXd::Zero(nv, nv));
        for (int i = 0; i < nv; i++) {
            for (int j = 0; j < nv; j++) {
                for (int k = 0; k < nv; k++) {
                    dqq_t(i, j) += lambda_fd(k) * d2tau_dqdq_t(k, i, j);
                    dvv_t(i, j) += lambda_fd(k) * d2tau_dvdv_t(k, i, j);
                    dqv_t(i, j) += lambda_fd(k) * d2tau_dqdv_t(k, i, j);
                    dqa_t(i, j) += lambda_fd(k) * d2tau_dadq_t(k, i, j);
                }
            }
        }
        dqa_t.transposeInPlace();  // dadq -> dqa

        // Apply FD chain-rule with tensor-derived mod ID SO matrices
        MatrixXd tens_fd[] = {
            -dqq_t - dqa_t * ddq_dq_mat - ddq_dq_mat.transpose() * dqa_t.transpose(),  // dqq
            -dvv_t,                                                                       // dvv
            -dqv_t - dqa_t * ddq_dv_mat,                                                 // dqv
            -dqa_t * Minv_mat                                                             // dqtau
        };

        auto res1 = eval_case1(input);
        auto res2a = eval_case2a(input);
        auto res2b = eval_case2b(input);

        string names[] = {"dqq", "dvv", "dqv", "dqtau"};

        double tol = 1e-6;
        bool all_pass = true;
        MatrixXd ana_arr[] = {ana[0], ana[1], ana[2], ana[3]};

        auto check_result = [&](const string& case_name, const ::casadi::DMVector& res, const MatrixXd ref[]) {
            std::cout << "\n" << case_name << ":" << std::endl;
            for (int k = 0; k < 4; k++) {
                MatrixXd mat = casadi_to_eigen(res[k]);
                bool has_nan = mat.hasNaN();
                double err = has_nan ? std::numeric_limits<double>::quiet_NaN() : (mat - ref[k]).norm();
                bool pass = !has_nan && (err < tol);
                if (!pass) all_pass = false;
                std::cout << "  " << names[k] << ": " << err
                          << (has_nan ? " [NAN!]" : (pass ? " [OK]" : " [FAIL]")) << std::endl;
            }
        };

        auto check_matrix = [&](const string& case_name, const MatrixXd mats[], const MatrixXd ref[]) {
            std::cout << "\n" << case_name << ":" << std::endl;
            for (int k = 0; k < 4; k++) {
                bool has_nan = mats[k].hasNaN();
                double err = has_nan ? std::numeric_limits<double>::quiet_NaN() : (mats[k] - ref[k]).norm();
                bool pass = !has_nan && (err < tol);
                if (!pass) all_pass = false;
                std::cout << "  " << names[k] << ": " << err
                          << (has_nan ? " [NAN!]" : (pass ? " [OK]" : " [FAIL]")) << std::endl;
            }
        };

        check_result("Case 1 vs 3 (full SO AD vs analytical)", res1, ana_arr);
        check_result("Case 2a vs 3 (AD over full FO vs analytical)", res2a, ana_arr);
        check_result("Case 2b vs 3 (AD over mod FO vs analytical)", res2b, ana_arr);
        check_matrix("Case 4 vs 3 (full SO tensor + lambda + FD chain-rule vs mod FD)", tens_fd, ana_arr);

        if (all_pass)
            std::cout << "\n>>> ALL PASSED for " << robot_name << " <<<" << std::endl;
        else
            std::cout << "\n>>> SOME CHECKS FAILED for " << robot_name << " <<<" << std::endl;

    } // end model loop

    return 0;
}
