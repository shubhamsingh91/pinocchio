/* bench_modID_SO.cpp
 * Modified ID Second-Order Derivatives Benchmark
 *
 * Compares 4 approaches for computing the Hessian of the scalar lambda*tau(q,v,a):
 *
 * Case 1:  Full SO AD — trace modrnea() with CasADi, take Hessian (2 AD diffs)
 * Case 2a: FO AD over full FO — trace computeRNEADerivatives(), contract with lambda, Jacobian (1 AD diff)
 * Case 2b: FO AD over mod FO — trace computeModRNEADerivatives(), Jacobian (1 AD diff)
 * Case 3:  Full analytical — computeModRNEASecondOrderDerivatives() (0 AD diffs)
 *
 * All 4 produce the same 4 nv x nv matrices: dqq, dvv, dvq, dqa
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
    int NBT = 100000;
#else
    int NBT = 1;
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

        VectorXd qmax = VectorXd::Ones(nq);

        // Random test data
        PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) qs(NBT);
        PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) qdots(NBT);
        PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) qddots(NBT);
        PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) lambdas(NBT);

        for (size_t i = 0; i < (size_t)NBT; ++i) {
            qs[i] = randomConfiguration(model, -qmax, qmax);
            qdots[i] = VectorXd::Random(nv);
            qddots[i] = VectorXd::Random(nv);
            lambdas[i] = VectorXd::Random(nv);
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
        ::casadi::SX cs_a = ::casadi::SX::sym("a", nv);

        // q_int = integrate(q, v_int)
        ConfigVectorAD q_ad(nq), v_int_ad(nv), q_int_ad(nq);
        q_ad = Eigen::Map<ConfigVectorAD>(static_cast<std::vector<ADcScalar>>(cs_q).data(), nq, 1);
        v_int_ad = Eigen::Map<ConfigVectorAD>(static_cast<std::vector<ADcScalar>>(cs_v_int).data(), nv, 1);
        pinocchio::integrate(adc_model, q_ad, v_int_ad, q_int_ad);

        TangentVectorAD v_ad(nv);
        v_ad = Eigen::Map<TangentVectorAD>(static_cast<std::vector<ADcScalar>>(cs_v).data(), nv, 1);

        TangentVectorAD a_ad(nv);
        a_ad = Eigen::Map<TangentVectorAD>(static_cast<std::vector<ADcScalar>>(cs_a).data(), nv, 1);

        // Lambda as symbolic variable
        ::casadi::SX cs_lambda = ::casadi::SX::sym("lambda", nv);
        TangentVectorAD lambda_ad(nv);
        lambda_ad = Eigen::Map<TangentVectorAD>(static_cast<std::vector<ADcScalar>>(cs_lambda).data(), nv, 1);

        // Evaluation vectors
        std::vector<double> q_vec((size_t)nq);
        std::vector<double> v_int_vec((size_t)nv, 0.0);
        std::vector<double> v_vec((size_t)nv);
        std::vector<double> a_vec((size_t)nv);
        std::vector<double> lambda_vec((size_t)nv);

        // ================================================================
        // Case 1: Full SO AD — trace modrnea, take Hessian
        // ================================================================
        std::cout << "\n--- Case 1: Full SO AD (trace modrnea) ---" << std::endl;

        // Need fresh adc_data for this trace
        ADcData adc_data1(adc_model);
        modrnea(adc_model, adc_data1, q_int_ad, v_ad, a_ad, lambda_ad);
        ::casadi::SX cs_modtau = adc_data1.modtau;  // scalar

        // Hessian blocks via 2 jacobian calls each
        ::casadi::SX grad_q_1 = jacobian(cs_modtau, cs_v_int);   // 1 x nv
        ::casadi::SX hess_qq_1 = jacobian(grad_q_1, cs_v_int);   // nv x nv: d2f/(dq_i dq_j)

        ::casadi::SX grad_v_1 = jacobian(cs_modtau, cs_v);       // 1 x nv
        ::casadi::SX hess_vv_1 = jacobian(grad_v_1, cs_v);       // nv x nv: d2f/(dv_i dv_j)

        // dvq: d2f/(dq_i dv_j) — diff q-gradient w.r.t. v
        ::casadi::SX hess_vq_1 = jacobian(grad_q_1, cs_v);       // nv x nv
        // dqa: d2f/(dq_i da_j) — diff q-gradient w.r.t. a
        ::casadi::SX hess_qa_1 = jacobian(grad_q_1, cs_a);       // nv x nv

        ::casadi::Function eval_case1(
            "case1_modrnea_SO",
            ::casadi::SXVector {cs_q, cs_v_int, cs_v, cs_a, cs_lambda},
            ::casadi::SXVector {hess_qq_1, hess_vv_1, hess_vq_1, hess_qa_1});

        // ================================================================
        // Case 2a: FO AD over computeRNEADerivatives + contract
        // ================================================================
        std::cout << "--- Case 2a: FO AD over full FO derivs ---" << std::endl;

        ADcData adc_data2a(adc_model);
        pinocchio::computeRNEADerivativesFaster(adc_model, adc_data2a, q_int_ad, v_ad, a_ad);
        // Symmetrize M
        (adc_data2a.M).template triangularView<Eigen::StrictlyLower>()
            = (adc_data2a.M).transpose().template triangularView<Eigen::StrictlyLower>();

        // Contract with lambda: g_q(i) = sum_j lambda(j) * dtau_dq(j,i)
        ::casadi::SX g_q_2a(nv, 1), g_v_2a(nv, 1), g_a_2a(nv, 1);
        for (int i = 0; i < nv; i++) {
            g_q_2a(i) = 0; g_v_2a(i) = 0; g_a_2a(i) = 0;
            for (int j = 0; j < nv; j++) {
                g_q_2a(i) += cs_lambda(j) * adc_data2a.dtau_dq(j, i);
                g_v_2a(i) += cs_lambda(j) * adc_data2a.dtau_dv(j, i);
                g_a_2a(i) += cs_lambda(j) * adc_data2a.M(j, i);
            }
        }

        ::casadi::SX hess_qq_2a = jacobian(g_q_2a, cs_v_int);
        ::casadi::SX hess_vv_2a = jacobian(g_v_2a, cs_v);
        ::casadi::SX hess_vq_2a = jacobian(g_q_2a, cs_v);       // d2f/(dq_i dv_j)
        ::casadi::SX hess_qa_2a = jacobian(g_q_2a, cs_a);       // d2f/(dq_i da_j)

        ::casadi::Function eval_case2a(
            "case2a_RNEAFO_contract",
            ::casadi::SXVector {cs_q, cs_v_int, cs_v, cs_a, cs_lambda},
            ::casadi::SXVector {hess_qq_2a, hess_vv_2a, hess_vq_2a, hess_qa_2a});

        // ================================================================
        // Case 2b: FO AD over computeModRNEADerivatives
        // ================================================================
        std::cout << "--- Case 2b: FO AD over mod FO derivs ---" << std::endl;

        ADcData adc_data2b(adc_model);
        pinocchio::computeModRNEADerivatives(adc_model, adc_data2b, q_int_ad, v_ad, a_ad, lambda_ad);

        // Extract gradient vectors from data
        ::casadi::SX g_q_2b(nv, 1), g_v_2b(nv, 1), g_a_2b(nv, 1);
        for (int i = 0; i < nv; i++) {
            g_q_2b(i) = adc_data2b.dtau_dq_mod[i];
            g_v_2b(i) = adc_data2b.dtau_dv_mod[i];
            g_a_2b(i) = adc_data2b.M_mod[i];
        }

        ::casadi::SX hess_qq_2b = jacobian(g_q_2b, cs_v_int);
        ::casadi::SX hess_vv_2b = jacobian(g_v_2b, cs_v);
        ::casadi::SX hess_vq_2b = jacobian(g_q_2b, cs_v);       // d2f/(dq_i dv_j)
        ::casadi::SX hess_qa_2b = jacobian(g_q_2b, cs_a);       // d2f/(dq_i da_j)

        ::casadi::Function eval_case2b(
            "case2b_modRNEAFO",
            ::casadi::SXVector {cs_q, cs_v_int, cs_v, cs_a, cs_lambda},
            ::casadi::SXVector {hess_qq_2b, hess_vv_2b, hess_vq_2b, hess_qa_2b});

        // ================================================================
        // ACCURACY CHECK (single evaluation)
        // ================================================================
        std::cout << "\n--- Accuracy check ---" << std::endl;

        // Set up evaluation point
        Eigen::Map<ConfigVector>(q_vec.data(), nq, 1) = qs[0];
        Eigen::Map<TangentVector>(v_vec.data(), nv, 1) = qdots[0];
        Eigen::Map<TangentVector>(a_vec.data(), nv, 1) = qddots[0];
        Eigen::Map<TangentVector>(lambda_vec.data(), nv, 1) = lambdas[0];

        // Case 3: Analytical reference
        MatrixXd dqq_ana(MatrixXd::Zero(nv, nv));
        MatrixXd dvv_ana(MatrixXd::Zero(nv, nv));
        MatrixXd dvq_ana(MatrixXd::Zero(nv, nv));
        MatrixXd dqa_ana(MatrixXd::Zero(nv, nv));
        computeModRNEASecondOrderDerivatives(model, data, qs[0], qdots[0], qddots[0], lambdas[0],
            dqq_ana, dvv_ana, dvq_ana, dqa_ana);

        // Helper to extract nv x nv from CasADi result
        auto casadi_to_eigen = [&](const ::casadi::DM& dm, int rows, int cols) -> MatrixXd {
            std::vector<double> v = static_cast<std::vector<double>>(dm);
            return Eigen::Map<MatrixXd>(v.data(), rows, cols);
        };

        // Case 1 evaluation
        auto res1 = eval_case1(::casadi::DMVector {q_vec, v_int_vec, v_vec, a_vec, lambda_vec});
        MatrixXd dqq_1 = casadi_to_eigen(res1[0], nv, nv);
        MatrixXd dvv_1 = casadi_to_eigen(res1[1], nv, nv);
        MatrixXd dvq_1 = casadi_to_eigen(res1[2], nv, nv);
        MatrixXd dqa_1 = casadi_to_eigen(res1[3], nv, nv);

        // Case 2a evaluation
        auto res2a = eval_case2a(::casadi::DMVector {q_vec, v_int_vec, v_vec, a_vec, lambda_vec});
        MatrixXd dqq_2a = casadi_to_eigen(res2a[0], nv, nv);
        MatrixXd dvv_2a = casadi_to_eigen(res2a[1], nv, nv);
        MatrixXd dvq_2a = casadi_to_eigen(res2a[2], nv, nv);
        MatrixXd dqa_2a = casadi_to_eigen(res2a[3], nv, nv);

        // Case 2b evaluation
        auto res2b = eval_case2b(::casadi::DMVector {q_vec, v_int_vec, v_vec, a_vec, lambda_vec});
        MatrixXd dqq_2b = casadi_to_eigen(res2b[0], nv, nv);
        MatrixXd dvv_2b = casadi_to_eigen(res2b[1], nv, nv);
        MatrixXd dvq_2b = casadi_to_eigen(res2b[2], nv, nv);
        MatrixXd dqa_2b = casadi_to_eigen(res2b[3], nv, nv);

        // Print accuracy
        std::cout << "Case 1 vs 3 (full SO AD vs analytical):" << std::endl;
        std::cout << "  dqq: " << (dqq_1 - dqq_ana).norm() << std::endl;
        std::cout << "  dvv: " << (dvv_1 - dvv_ana).norm() << std::endl;
        std::cout << "  dvq: " << (dvq_1 - dvq_ana).norm() << std::endl;
        std::cout << "  dqa: " << (dqa_1 - dqa_ana).norm() << std::endl;

        std::cout << "Case 2a vs 3 (AD over full FO vs analytical):" << std::endl;
        std::cout << "  dqq: " << (dqq_2a - dqq_ana).norm() << std::endl;
        std::cout << "  dvv: " << (dvv_2a - dvv_ana).norm() << std::endl;
        std::cout << "  dvq: " << (dvq_2a - dvq_ana).norm() << std::endl;
        std::cout << "  dqa: " << (dqa_2a - dqa_ana).norm() << std::endl;

        std::cout << "Case 2b vs 3 (AD over mod FO vs analytical):" << std::endl;
        std::cout << "  dqq: " << (dqq_2b - dqq_ana).norm() << std::endl;
        std::cout << "  dvv: " << (dvv_2b - dvv_ana).norm() << std::endl;
        std::cout << "  dvq: " << (dvq_2b - dvq_ana).norm() << std::endl;
        std::cout << "  dqa: " << (dqa_2b - dqa_ana).norm() << std::endl;

        // ================================================================
        // TIMING
        // ================================================================
        std::cout << "\n--- Timing (" << NBT << " iterations) ---" << std::endl;

        // Case 3: Analytical
        timer.tic();
        SMOOTH(NBT)
        {
            dqq_ana.setZero(); dvv_ana.setZero(); dvq_ana.setZero(); dqa_ana.setZero();
            computeModRNEASecondOrderDerivatives(model, data, qs[_smooth], qdots[_smooth], qddots[_smooth],
                lambdas[_smooth], dqq_ana, dvv_ana, dvq_ana, dqa_ana);
        }
        time_cases[0] = timer.toc() / NBT;
        std::cout << "Case 3  (analytical):          " << time_cases[0] << " us" << std::endl;

        // Case 1: Full SO AD
        timer.tic();
        SMOOTH(NBT)
        {
            Eigen::Map<ConfigVector>(q_vec.data(), nq, 1) = qs[_smooth];
            Eigen::Map<TangentVector>(v_vec.data(), nv, 1) = qdots[_smooth];
            Eigen::Map<TangentVector>(a_vec.data(), nv, 1) = qddots[_smooth];
            Eigen::Map<TangentVector>(lambda_vec.data(), nv, 1) = lambdas[_smooth];

            auto res = eval_case1(::casadi::DMVector {q_vec, v_int_vec, v_vec, a_vec, lambda_vec});
        }
        time_cases[1] = timer.toc() / NBT;
        std::cout << "Case 1  (full SO AD):          " << time_cases[1] << " us" << std::endl;

        // Case 2a: FO AD over full FO derivs
        timer.tic();
        SMOOTH(NBT)
        {
            Eigen::Map<ConfigVector>(q_vec.data(), nq, 1) = qs[_smooth];
            Eigen::Map<TangentVector>(v_vec.data(), nv, 1) = qdots[_smooth];
            Eigen::Map<TangentVector>(a_vec.data(), nv, 1) = qddots[_smooth];
            Eigen::Map<TangentVector>(lambda_vec.data(), nv, 1) = lambdas[_smooth];

            auto res = eval_case2a(::casadi::DMVector {q_vec, v_int_vec, v_vec, a_vec, lambda_vec});
        }
        time_cases[2] = timer.toc() / NBT;
        std::cout << "Case 2a (AD over full FO):     " << time_cases[2] << " us" << std::endl;

        // Case 2b: FO AD over mod FO derivs
        timer.tic();
        SMOOTH(NBT)
        {
            Eigen::Map<ConfigVector>(q_vec.data(), nq, 1) = qs[_smooth];
            Eigen::Map<TangentVector>(v_vec.data(), nv, 1) = qdots[_smooth];
            Eigen::Map<TangentVector>(a_vec.data(), nv, 1) = qddots[_smooth];
            Eigen::Map<TangentVector>(lambda_vec.data(), nv, 1) = lambdas[_smooth];

            auto res = eval_case2b(::casadi::DMVector {q_vec, v_int_vec, v_vec, a_vec, lambda_vec});
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
        string outdir = "data/modID_SO/";
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
