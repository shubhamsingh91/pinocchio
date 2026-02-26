/* bench_modID_SO_accuracy.cpp
 * Accuracy check for modified ID second-order derivatives.
 *
 * Compares 3 CasADi approaches against the analytical reference:
 *   Case 1:  Full SO AD — trace modrnea(), Hessian (2 AD diffs)
 *   Case 2a: FO AD over full FO — trace computeRNEADerivativesFaster(), contract, Jacobian (1 AD diff)
 *   Case 2b: FO AD over mod FO — trace computeModRNEADerivatives(), Jacobian (1 AD diff)
 *   Case 3:  Analytical — computeModRNEASecondOrderDerivatives() (reference)
 */

#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/modrnea.hpp"
#include "pinocchio/algorithm/rnea-derivatives-faster.hpp"
#include "pinocchio/algorithm/mod-rnea-derivatives.hpp"
#include "pinocchio/codegen/code-generator-algo.hpp"
#include "pinocchio/algorithm/mod-rnea-second-order-derivatives.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/container/aligned-vector.hpp"

#include <casadi/casadi.hpp>
#include "pinocchio/autodiff/casadi.hpp"

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
        VectorXd a = VectorXd::Random(nv);
        VectorXd lambda = VectorXd::Random(nv);

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
        ::casadi::SX cs_a = ::casadi::SX::sym("a", nv);
        ::casadi::SX cs_lambda = ::casadi::SX::sym("lambda", nv);

        ConfigVectorAD q_ad(nq), v_int_ad(nv), q_int_ad(nq);
        q_ad = Eigen::Map<ConfigVectorAD>(static_cast<std::vector<ADcScalar>>(cs_q).data(), nq, 1);
        v_int_ad = Eigen::Map<ConfigVectorAD>(static_cast<std::vector<ADcScalar>>(cs_v_int).data(), nv, 1);
        pinocchio::integrate(adc_model, q_ad, v_int_ad, q_int_ad);

        TangentVectorAD v_ad(nv), a_ad(nv), lambda_ad(nv);
        v_ad = Eigen::Map<TangentVectorAD>(static_cast<std::vector<ADcScalar>>(cs_v).data(), nv, 1);
        a_ad = Eigen::Map<TangentVectorAD>(static_cast<std::vector<ADcScalar>>(cs_a).data(), nv, 1);
        lambda_ad = Eigen::Map<TangentVectorAD>(static_cast<std::vector<ADcScalar>>(cs_lambda).data(), nv, 1);

        // ================================================================
        // Case 1: Full SO AD — trace modrnea, take Hessian
        // ================================================================
        ADcData adc_data1(adc_model);
        modrnea(adc_model, adc_data1, q_int_ad, v_ad, a_ad, lambda_ad);
        ::casadi::SX cs_modtau = adc_data1.modtau;

        ::casadi::SX grad_q_1 = jacobian(cs_modtau, cs_v_int);
        ::casadi::SX grad_v_1 = jacobian(cs_modtau, cs_v);

        ::casadi::Function eval_case1(
            "case1",
            ::casadi::SXVector {cs_q, cs_v_int, cs_v, cs_a, cs_lambda},
            ::casadi::SXVector {
                jacobian(grad_q_1, cs_v_int),  // dqq
                jacobian(grad_v_1, cs_v),      // dvv
                jacobian(grad_q_1, cs_v),      // dvq
                jacobian(grad_q_1, cs_a)       // dqa
            });

        // ================================================================
        // Case 2a: FO AD over computeRNEADerivativesFaster + contract
        // ================================================================
        ADcData adc_data2a(adc_model);
        pinocchio::computeRNEADerivativesFaster(adc_model, adc_data2a, q_int_ad, v_ad, a_ad);
        (adc_data2a.M).template triangularView<Eigen::StrictlyLower>()
            = (adc_data2a.M).transpose().template triangularView<Eigen::StrictlyLower>();

        ::casadi::SX g_q_2a(nv, 1), g_v_2a(nv, 1);
        for (int i = 0; i < nv; i++) {
            g_q_2a(i) = 0; g_v_2a(i) = 0;
            for (int j = 0; j < nv; j++) {
                g_q_2a(i) += cs_lambda(j) * adc_data2a.dtau_dq(j, i);
                g_v_2a(i) += cs_lambda(j) * adc_data2a.dtau_dv(j, i);
            }
        }

        ::casadi::Function eval_case2a(
            "case2a",
            ::casadi::SXVector {cs_q, cs_v_int, cs_v, cs_a, cs_lambda},
            ::casadi::SXVector {
                jacobian(g_q_2a, cs_v_int),  // dqq
                jacobian(g_v_2a, cs_v),      // dvv
                jacobian(g_q_2a, cs_v),      // dvq
                jacobian(g_q_2a, cs_a)       // dqa
            });

        // ================================================================
        // Case 2b: FO AD over computeModRNEADerivatives
        // ================================================================
        ADcData adc_data2b(adc_model);
        pinocchio::computeModRNEADerivatives(adc_model, adc_data2b, q_int_ad, v_ad, a_ad, lambda_ad);

        ::casadi::SX g_q_2b(nv, 1), g_v_2b(nv, 1);
        for (int i = 0; i < nv; i++) {
            g_q_2b(i) = adc_data2b.dtau_dq_mod[i];
            g_v_2b(i) = adc_data2b.dtau_dv_mod[i];
        }

        ::casadi::Function eval_case2b(
            "case2b",
            ::casadi::SXVector {cs_q, cs_v_int, cs_v, cs_a, cs_lambda},
            ::casadi::SXVector {
                jacobian(g_q_2b, cs_v_int),  // dqq
                jacobian(g_v_2b, cs_v),      // dvv
                jacobian(g_q_2b, cs_v),      // dvq
                jacobian(g_q_2b, cs_a)       // dqa
            });

        // ================================================================
        // Evaluate at a single point
        // ================================================================
        std::vector<double> q_vec((size_t)nq);
        std::vector<double> v_int_vec((size_t)nv, 0.0);
        std::vector<double> v_vec((size_t)nv);
        std::vector<double> a_vec((size_t)nv);
        std::vector<double> lambda_vec((size_t)nv);

        Eigen::Map<Model::ConfigVectorType>(q_vec.data(), nq, 1) = q;
        Eigen::Map<Model::TangentVectorType>(v_vec.data(), nv, 1) = v;
        Eigen::Map<Model::TangentVectorType>(a_vec.data(), nv, 1) = a;
        Eigen::Map<Model::TangentVectorType>(lambda_vec.data(), nv, 1) = lambda;

        ::casadi::DMVector input = {q_vec, v_int_vec, v_vec, a_vec, lambda_vec};

        auto casadi_to_eigen = [&](const ::casadi::DM& dm) -> MatrixXd {
            std::vector<double> vals = static_cast<std::vector<double>>(dm);
            return Eigen::Map<MatrixXd>(vals.data(), nv, nv);
        };

        // Case 3: Analytical reference
        MatrixXd dqq_ana(MatrixXd::Zero(nv, nv)), dvv_ana(MatrixXd::Zero(nv, nv));
        MatrixXd dvq_ana(MatrixXd::Zero(nv, nv)), dqa_ana(MatrixXd::Zero(nv, nv));
        computeModRNEASecondOrderDerivatives(model, data, q, v, a, lambda,
            dqq_ana, dvv_ana, dvq_ana, dqa_ana);

        auto res1 = eval_case1(input);
        auto res2a = eval_case2a(input);
        auto res2b = eval_case2b(input);

        string names[] = {"dqq", "dvv", "dvq", "dqa"};
        MatrixXd ana[] = {dqq_ana, dvv_ana, dvq_ana, dqa_ana};

        double tol = 1e-6;
        bool all_pass = true;

        auto check_result = [&](const string& case_name, const ::casadi::DMVector& res, const MatrixXd ana_arr[]) {
            std::cout << "\n" << case_name << ":" << std::endl;
            for (int k = 0; k < 4; k++) {
                MatrixXd mat = casadi_to_eigen(res[k]);
                bool has_nan = mat.hasNaN();
                double err = has_nan ? std::numeric_limits<double>::quiet_NaN() : (mat - ana_arr[k]).norm();
                bool pass = !has_nan && (err < tol);
                if (!pass) all_pass = false;
                std::cout << "  " << names[k] << ": " << err
                          << (has_nan ? " [NAN!]" : (pass ? " [OK]" : " [FAIL]")) << std::endl;
            }
        };

        check_result("Case 1 vs 3 (full SO AD vs analytical)", res1, ana);
        check_result("Case 2a vs 3 (AD over full FO vs analytical)", res2a, ana);
        check_result("Case 2b vs 3 (AD over mod FO vs analytical)", res2b, ana);

        if (all_pass)
            std::cout << "\n>>> ALL PASSED for " << robot_name << " <<<" << std::endl;
        else
            std::cout << "\n>>> SOME CHECKS FAILED for " << robot_name << " <<<" << std::endl;

    } // end model loop

    return 0;
}
