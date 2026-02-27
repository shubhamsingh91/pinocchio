/* bench_modFD_SO.cpp
 * Modified FD Second-Order Derivatives Benchmark (with CasADi codegen)
 *
 * Compares 6 approaches for computing modified FD SO derivatives:
 *
 * FO:      FO analytical — aba() + computeABADerivativesFaster() (baseline from RAL)
 * Full SO: Full FD SO analytical — full chain-rule with ID SO tensors (from T-Ro)
 * Mod SO:  Mod FD SO analytical — chain-rule via computeModRNEASecondOrderDerivatives() (our algo)
 * Case 1:  Full SO AD — trace modaba() with CasADi codegen (2 AD diffs)
 * Case 2a: FO AD over full FO — CasADi codegen (1 AD diff)
 * Case 2b: FO AD over mod FO — CasADi codegen (1 AD diff)
 *
 * Usage:
 *   ./bench_modFD_SO --codegen              Build CasADi .so files
 *   ./bench_modFD_SO --eval                 Run timing, save data
 *   ./bench_modFD_SO --eval --outdir <dir>  Save data to alternate directory
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
#include "pinocchio/utils/timer.hpp"
#include "pinocchio/utils/tensor_utils.hpp"

#include <casadi/casadi.hpp>
#include "pinocchio/autodiff/casadi.hpp"

#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdio>

using namespace std;
using namespace Eigen;
using namespace pinocchio;
using namespace pinocchio::casadi;

int main(int argc, const char* argv[])
{
    bool do_codegen = false;
    bool do_eval = false;
    string outdir = "data/modFD_SO";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--codegen") == 0) do_codegen = true;
        if (strcmp(argv[i], "--eval") == 0)    do_eval = true;
        if (strcmp(argv[i], "--outdir") == 0 && i + 1 < argc) outdir = argv[++i];
    }

    if (!do_codegen && !do_eval) {
        cout << "Usage: " << argv[0] << " [--codegen] [--eval] [--outdir <dir>]" << endl;
        cout << "  --codegen       Build CasADi symbolic functions, generate C code, compile .so" << endl;
        cout << "  --eval          Load compiled .so files, run timing benchmarks, save data" << endl;
        cout << "  --outdir <dir>  Output directory for timing data (default: data/modFD_SO)" << endl;
        return 0;
    }

    PinocchioTicToc timer(PinocchioTicToc::US);

#ifdef NDEBUG
    int NBT = 100000;
#else
    int NBT = 1;
    std::cout << "(the time score in debug mode is not relevant)" << std::endl;
#endif

    string codegen_dir = "codegen/";

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
        typedef Model::ConfigVectorType ConfigVector;
        typedef Model::TangentVectorType TangentVector;
        const int nv = model.nv;
        const int nq = model.nq;

        // Function name prefixes
        string fn_case1  = robot_name + "_modFD_SO_case1";
        string fn_case2a = robot_name + "_modFD_SO_case2a";
        string fn_case2b = robot_name + "_modFD_SO_case2b";

        // ================================================================
        // CODEGEN PHASE
        // ================================================================
        if (do_codegen) {
            // Skip if all 3 .so files already exist
            FILE* f1 = fopen((codegen_dir + fn_case1  + ".so").c_str(), "r");
            FILE* f2 = fopen((codegen_dir + fn_case2a + ".so").c_str(), "r");
            FILE* f3 = fopen((codegen_dir + fn_case2b + ".so").c_str(), "r");
            bool all_exist = (f1 && f2 && f3);
            if (f1) fclose(f1); if (f2) fclose(f2); if (f3) fclose(f3);

            if (all_exist) {
                std::cout << "\n--- Codegen: all .so files found, skipping ---" << std::endl;
            } else {
            std::cout << "\n--- Codegen phase ---" << std::endl;

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

            // Case 1: Full SO AD — trace modaba, take Hessian
            std::cout << "Case 1: trace modaba..." << std::flush;
            ADcData adc_data1(adc_model);
            ADcScalar cs_modqdd = modaba(adc_model, adc_data1, q_int_ad, v_ad, tau_ad, mu_ad);

            ::casadi::SX grad_q_1 = jacobian(cs_modqdd, cs_v_int);
            ::casadi::SX grad_v_1 = jacobian(cs_modqdd, cs_v);

            ::casadi::Function eval_case1(
                fn_case1, cs_inputs,
                ::casadi::SXVector {
                    jacobian(grad_q_1, cs_v_int),
                    jacobian(grad_v_1, cs_v),
                    jacobian(grad_q_1, cs_v),
                    jacobian(grad_q_1, cs_tau)
                });
            std::cout << " done" << std::endl;

            // Case 2a: FO AD over computeABADerivativesFaster + contract
            std::cout << "Case 2a: trace ABA FO + contract..." << std::flush;
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
                fn_case2a, cs_inputs,
                ::casadi::SXVector {
                    jacobian(g_q_2a, cs_v_int),
                    jacobian(g_v_2a, cs_v),
                    jacobian(g_q_2a, cs_v),
                    jacobian(g_q_2a, cs_tau)
                });
            std::cout << " done" << std::endl;

            // Case 2b: FO AD over computeModABADerivatives
            std::cout << "Case 2b: trace modABA FO..." << std::flush;
            ADcData adc_data2b(adc_model);
            pinocchio::computeModABADerivatives(adc_model, adc_data2b, q_int_ad, v_ad, tau_ad, mu_ad);

            ::casadi::SX g_q_2b(nv, 1), g_v_2b(nv, 1);
            for (int i = 0; i < nv; i++) {
                g_q_2b(i) = adc_data2b.ddq_dq_mod[i];
                g_v_2b(i) = adc_data2b.ddq_dv_mod[i];
            }

            ::casadi::Function eval_case2b(
                fn_case2b, cs_inputs,
                ::casadi::SXVector {
                    jacobian(g_q_2b, cs_v_int),
                    jacobian(g_v_2b, cs_v),
                    jacobian(g_q_2b, cs_v),
                    jacobian(g_q_2b, cs_tau)
                });
            std::cout << " done" << std::endl;

            // Generate C code, move to codegen/, compile .so in parallel
            system(("mkdir -p " + codegen_dir).c_str());

            std::cout << "Generating C code..." << std::flush;
            eval_case1.generate(fn_case1);
            eval_case2a.generate(fn_case2a);
            eval_case2b.generate(fn_case2b);
            // Move .c files into codegen/ immediately
            rename((fn_case1  + ".c").c_str(), (codegen_dir + fn_case1  + ".c").c_str());
            rename((fn_case2a + ".c").c_str(), (codegen_dir + fn_case2a + ".c").c_str());
            rename((fn_case2b + ".c").c_str(), (codegen_dir + fn_case2b + ".c").c_str());
            std::cout << " done" << std::endl;

            std::cout << "Compiling .so (sequential)..." << std::flush;
            string cmd1 = "gcc -fPIC -shared -O3 -march=native " + codegen_dir + fn_case1  + ".c -o " + codegen_dir + fn_case1  + ".so";
            string cmd2 = "gcc -fPIC -shared -O3 -march=native " + codegen_dir + fn_case2a + ".c -o " + codegen_dir + fn_case2a + ".so";
            string cmd3 = "gcc -fPIC -shared -O3 -march=native " + codegen_dir + fn_case2b + ".c -o " + codegen_dir + fn_case2b + ".so";
            int flag = system(cmd1.c_str());
            flag |= system(cmd2.c_str());
            flag |= system(cmd3.c_str());
            std::cout << (flag == 0 ? " ok" : " FAILED") << std::endl;
            } // end else (not all_exist)
        }

        // ================================================================
        // EVAL PHASE
        // ================================================================
        if (do_eval) {
            std::cout << "\n--- Eval phase ---" << std::endl;

            // Check which .so files exist
            bool has_case1 = false, has_case2a = false, has_case2b = false;
            {
                FILE* f;
                f = fopen((codegen_dir + fn_case1  + ".so").c_str(), "r");
                if (f) { has_case1  = true; fclose(f); }
                f = fopen((codegen_dir + fn_case2a + ".so").c_str(), "r");
                if (f) { has_case2a = true; fclose(f); }
                f = fopen((codegen_dir + fn_case2b + ".so").c_str(), "r");
                if (f) { has_case2b = true; fclose(f); }
            }
            std::cout << "  case1.so: " << (has_case1  ? "found" : "MISSING") << std::endl;
            std::cout << "  case2a.so: " << (has_case2a ? "found" : "MISSING") << std::endl;
            std::cout << "  case2b.so: " << (has_case2b ? "found" : "MISSING") << std::endl;

            // Load only available codegen'd functions
            ::casadi::Function cg_case1, cg_case2a, cg_case2b;
            if (has_case1)  cg_case1  = ::casadi::external(fn_case1,  codegen_dir + fn_case1  + ".so");
            if (has_case2a) cg_case2a = ::casadi::external(fn_case2a, codegen_dir + fn_case2a + ".so");
            if (has_case2b) cg_case2b = ::casadi::external(fn_case2b, codegen_dir + fn_case2b + ".so");

            VectorXd qmax = VectorXd::Ones(nq);

            // Random test data
            PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) qs(NBT);
            PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) qdots(NBT);
            PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) qddots(NBT);
            PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) taus(NBT);
            PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) mus(NBT);

            for (size_t i = 0; i < (size_t)NBT; ++i) {
                qs[i] = randomConfiguration(model, -qmax, qmax);
                qdots[i] = VectorXd::Random(nv);
                qddots[i] = VectorXd::Random(nv);
                taus[i] = VectorXd::Random(nv);
                mus[i] = VectorXd::Random(nv);
            }

            // Evaluation vectors
            std::vector<double> q_vec((size_t)nq);
            std::vector<double> v_int_vec((size_t)nv, 0.0);
            std::vector<double> v_vec((size_t)nv);
            std::vector<double> tau_vec((size_t)nv);
            std::vector<double> mu_vec((size_t)nv);

            // 6 timing slots: FO, Full FD SO, Mod FD SO, Case1, Case2a, Case2b
            double time_cases[6] = {-1, -1, -1, -1, -1, -1};

            // Temp matrices for mod FD SO analytical chain-rule
            MatrixXd d2tau_dqq(MatrixXd::Zero(nv, nv)), d2tau_dvv(MatrixXd::Zero(nv, nv));
            MatrixXd d2tau_dqv(MatrixXd::Zero(nv, nv)), d2tau_dqa(MatrixXd::Zero(nv, nv));

            // Pre-allocated tensors and matrices for full FD SO chain-rule
            Eigen::Tensor<double, 3> dtau2_dq_t(nv, nv, nv), dtau2_dv_t(nv, nv, nv);
            Eigen::Tensor<double, 3> dtau2_dqv_t(nv, nv, nv), M_FO_t(nv, nv, nv);
            Eigen::Tensor<double, 3> prodq(nv, nv, nv), prodv(nv, nv, nv), prodqdd(nv, nv, nv);
            Eigen::Tensor<double, 3> daba2_dq_t(nv, nv, nv), daba2_dv_t(nv, nv, nv);
            Eigen::Tensor<double, 3> daba2_qv_t(nv, nv, nv), daba2_tauq_t(nv, nv, nv);

            MatrixXd daba_dq_t(nv, nv), daba_dv_t(nv, nv), daba_dtau_t(nv, nv);
            MatrixXd Minv_full(nv, nv), Minv_neg(nv, nv);
            MatrixXd fso_mat1(nv, nv), fso_mat2(nv, nv), fso_mat3(nv, nv);
            VectorXd fso_vec1(nv), fso_vec2(nv);
            MatrixXd term_in(nv, 4 * nv * nv), term_out(nv, 4 * nv * nv);

            std::cout << "\n--- Timing (" << NBT << " iterations) ---" << std::endl;

            // FO analytical — aba + computeABADerivativesFaster
            timer.tic();
            SMOOTH(NBT)
            {
                aba(model, data, qs[_smooth], qdots[_smooth], taus[_smooth]);
                computeABADerivativesFaster(model, data, qs[_smooth], qdots[_smooth], taus[_smooth]);
            }
            time_cases[0] = timer.toc() / NBT;
            std::cout << "FO analytical:                 " << time_cases[0] << " us" << std::endl;

            // Full FD SO analytical — full chain-rule with ID SO tensors (T-Ro)
            timer.tic();
            SMOOTH(NBT)
            {
                pinocchio::computeABADerivatives(model, data, qs[_smooth], qdots[_smooth], taus[_smooth],
                    daba_dq_t, daba_dv_t, daba_dtau_t);

                dtau2_dq_t.setZero(); dtau2_dv_t.setZero();
                dtau2_dqv_t.setZero(); M_FO_t.setZero();
                ComputeRNEASecondOrderDerivatives(model, data, qs[_smooth], qdots[_smooth], qddots[_smooth],
                    dtau2_dq_t, dtau2_dv_t, dtau2_dqv_t, M_FO_t);

                Minv_full = daba_dtau_t;
                Minv_neg = -Minv_full;

                prodq.setZero(); prodv.setZero(); prodqdd.setZero();

                // Inner term — double loop (tensor slicing)
                for (int u = 0; u < nv; u++) {
                    get_mat_from_tens3_v1(M_FO_t, fso_mat1, nv, u);
                    for (int w = 0; w < nv; w++) {
                        get_mat_from_tens3_v1(M_FO_t, fso_mat2, nv, w);
                        fso_vec1 = fso_mat1 * daba_dq_t.col(w);
                        fso_vec2 = fso_mat2 * daba_dv_t.col(u);
                        hess_assign(prodq, fso_vec1, 0, u, w, 1, nv);
                        hess_assign(prodv, fso_vec2, 0, w, u, 1, nv);
                    }
                    fso_mat3.noalias() = fso_mat1 * Minv_full;
                    hess_assign_fd2(prodqdd, fso_mat3, nv, u);
                }

                // Inner term addition + build term_in
                for (int u = 0; u < nv; u++) {
                    get_mat_from_tens3_v1(dtau2_dq_t, fso_mat1, nv, u);
                    get_mat_from_tens2(prodq, fso_mat2, nv, u);
                    get_mat_from_tens3_v1(prodq, fso_mat3, nv, u);
                    fso_mat1 += fso_mat2 + fso_mat3;
                    term_in.middleCols(4 * u * nv, nv) = fso_mat1;

                    get_mat_from_tens3_v1(dtau2_dv_t, fso_mat2, nv, u);
                    term_in.middleCols((4 * u + 1) * nv, nv) = fso_mat2;

                    get_mat_from_tens3_v1(dtau2_dqv_t, fso_mat3, nv, u);
                    get_mat_from_tens3_v1(prodv, fso_mat2, nv, u);
                    fso_mat3 += fso_mat2;
                    term_in.middleCols((4 * u + 2) * nv, nv) = fso_mat3;

                    get_mat_from_tens2(prodqdd, fso_mat1, nv, u);
                    term_in.middleCols((4 * u + 3) * nv, nv) = fso_mat1;
                }

                // Outer term (DMM)
                term_out.noalias() = Minv_neg * term_in;

                // Final assignment
                for (int u = 0; u < nv; u++) {
                    for (int w = 0; w < nv; w++) {
                        hess_assign(daba2_dq_t, term_out.col(4 * u * nv + w), 0, w, u, 1, nv);
                        hess_assign(daba2_dv_t, term_out.col((4 * u + 1) * nv + w), 0, w, u, 1, nv);
                        hess_assign(daba2_qv_t, term_out.col((4 * u + 2) * nv + w), 0, w, u, 1, nv);
                        hess_assign(daba2_tauq_t, term_out.col((4 * u + 3) * nv + w), 0, w, u, 1, nv);
                    }
                }
            }
            time_cases[1] = timer.toc() / NBT;
            std::cout << "Full FD SO analytical (tensor):" << time_cases[1] << " us" << std::endl;

            // Mod FD SO analytical — chain-rule via computeModRNEASecondOrderDerivatives
            timer.tic();
            SMOOTH(NBT)
            {
                VectorXd qddot_t = aba(model, data, qs[_smooth], qdots[_smooth], taus[_smooth]);
                computeABADerivatives(model, data, qs[_smooth], qdots[_smooth], taus[_smooth]);
                MatrixXd Minv_t = data.Minv;
                Minv_t.triangularView<Eigen::StrictlyLower>() =
                    Minv_t.transpose().triangularView<Eigen::StrictlyLower>();
                VectorXd lambda_t = Minv_t * mus[_smooth];
                d2tau_dqq.setZero(); d2tau_dvv.setZero(); d2tau_dqv.setZero(); d2tau_dqa.setZero();
                computeModRNEASecondOrderDerivatives(model, data, qs[_smooth], qdots[_smooth],
                    qddot_t, lambda_t, d2tau_dqq, d2tau_dvv, d2tau_dqv, d2tau_dqa);
                MatrixXd dqq_t = -d2tau_dqq - d2tau_dqa * data.ddq_dq
                                  - data.ddq_dq.transpose() * d2tau_dqa.transpose();
                MatrixXd dvv_t = -d2tau_dvv;
                MatrixXd dqv_t = -d2tau_dqv - d2tau_dqa * data.ddq_dv;
                MatrixXd dqtau_t = -d2tau_dqa * Minv_t;
            }
            time_cases[2] = timer.toc() / NBT;
            std::cout << "Mod FD SO analytical:          " << time_cases[2] << " us" << std::endl;

            // Case 1: Full SO AD (codegen)
            if (has_case1) {
                timer.tic();
                SMOOTH(NBT)
                {
                    Eigen::Map<ConfigVector>(q_vec.data(), nq, 1) = qs[_smooth];
                    Eigen::Map<TangentVector>(v_vec.data(), nv, 1) = qdots[_smooth];
                    Eigen::Map<TangentVector>(tau_vec.data(), nv, 1) = taus[_smooth];
                    Eigen::Map<TangentVector>(mu_vec.data(), nv, 1) = mus[_smooth];
                    auto res = cg_case1(::casadi::DMVector {q_vec, v_int_vec, v_vec, tau_vec, mu_vec});
                }
                time_cases[3] = timer.toc() / NBT;
                std::cout << "Case 1  (full SO AD, cg):      " << time_cases[3] << " us" << std::endl;
            } else {
                std::cout << "Case 1  (full SO AD, cg):      SKIPPED (no .so)" << std::endl;
            }

            // Case 2a: FO AD over full FO (codegen)
            if (has_case2a) {
                timer.tic();
                SMOOTH(NBT)
                {
                    Eigen::Map<ConfigVector>(q_vec.data(), nq, 1) = qs[_smooth];
                    Eigen::Map<TangentVector>(v_vec.data(), nv, 1) = qdots[_smooth];
                    Eigen::Map<TangentVector>(tau_vec.data(), nv, 1) = taus[_smooth];
                    Eigen::Map<TangentVector>(mu_vec.data(), nv, 1) = mus[_smooth];
                    auto res = cg_case2a(::casadi::DMVector {q_vec, v_int_vec, v_vec, tau_vec, mu_vec});
                }
                time_cases[4] = timer.toc() / NBT;
                std::cout << "Case 2a (AD full FO, cg):      " << time_cases[4] << " us" << std::endl;
            } else {
                std::cout << "Case 2a (AD full FO, cg):      SKIPPED (no .so)" << std::endl;
            }

            // Case 2b: FO AD over mod FO (codegen)
            if (has_case2b) {
                timer.tic();
                SMOOTH(NBT)
                {
                    Eigen::Map<ConfigVector>(q_vec.data(), nq, 1) = qs[_smooth];
                    Eigen::Map<TangentVector>(v_vec.data(), nv, 1) = qdots[_smooth];
                    Eigen::Map<TangentVector>(tau_vec.data(), nv, 1) = taus[_smooth];
                    Eigen::Map<TangentVector>(mu_vec.data(), nv, 1) = mus[_smooth];
                    auto res = cg_case2b(::casadi::DMVector {q_vec, v_int_vec, v_vec, tau_vec, mu_vec});
                }
                time_cases[5] = timer.toc() / NBT;
                std::cout << "Case 2b (AD mod FO, cg):       " << time_cases[5] << " us" << std::endl;
            } else {
                std::cout << "Case 2b (AD mod FO, cg):       SKIPPED (no .so)" << std::endl;
            }

            // Speedups vs Mod FD SO
            std::cout << "\nSpeedups vs Mod FD SO analytical:" << std::endl;
            std::cout << "  FO      / Mod SO = " << time_cases[0] / time_cases[2] << "x" << std::endl;
            std::cout << "  Full SO / Mod SO = " << time_cases[1] / time_cases[2] << "x" << std::endl;
            if (has_case1)  std::cout << "  Case 1  / Mod SO = " << time_cases[3] / time_cases[2] << "x" << std::endl;
            else            std::cout << "  Case 1  / Mod SO = N/A" << std::endl;
            if (has_case2a) std::cout << "  Case 2a / Mod SO = " << time_cases[4] / time_cases[2] << "x" << std::endl;
            else            std::cout << "  Case 2a / Mod SO = N/A" << std::endl;
            if (has_case2b) std::cout << "  Case 2b / Mod SO = " << time_cases[5] / time_cases[2] << "x" << std::endl;
            else            std::cout << "  Case 2b / Mod SO = N/A" << std::endl;

            // Write timing data (6-line format, -1 for unavailable)
            system(("mkdir -p " + outdir).c_str());
            string outfile = outdir + "/" + robot_name + ".txt";
            std::ofstream ofs(outfile);
            if (ofs.is_open()) {
                ofs << time_cases[0] << std::endl;  // FO analytical
                ofs << time_cases[1] << std::endl;  // Full FD SO analytical
                ofs << time_cases[2] << std::endl;  // Mod FD SO analytical
                ofs << time_cases[3] << std::endl;  // Case 1 codegen
                ofs << time_cases[4] << std::endl;  // Case 2a codegen
                ofs << time_cases[5] << std::endl;  // Case 2b codegen
                ofs.close();
                std::cout << "Timing saved to " << outfile << std::endl;
            } else {
                std::cerr << "Warning: could not open " << outfile << " for writing" << std::endl;
            }
        }

    } // end model loop

    return 0;
}
