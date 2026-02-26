/* bench_modFD_SO.cpp
 * Modified FD Second-Order Derivatives Benchmark (with CasADi codegen)
 *
 * Compares 4 approaches for computing the Hessian of the scalar mu*qddot(q,v,tau):
 *
 * Case 1:  Full SO AD — trace modaba() with CasADi, Hessian (2 AD diffs)
 * Case 2a: FO AD over full FO — trace computeABADerivativesFaster(), contract, Jacobian (1 AD diff)
 * Case 2b: FO AD over mod FO — trace computeModABADerivatives(), Jacobian (1 AD diff)
 * Case 3:  Full analytical — chain-rule via computeModRNEASecondOrderDerivatives() + computeABADerivatives()
 *
 * Usage:
 *   ./bench_modFD_SO --codegen   Build CasADi functions, generate C, compile .so in codegen/
 *   ./bench_modFD_SO --eval      Load .so, run timing, save data
 *   ./bench_modFD_SO --codegen --eval   Both phases in one run
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

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--codegen") == 0) do_codegen = true;
        if (strcmp(argv[i], "--eval") == 0)    do_eval = true;
    }

    if (!do_codegen && !do_eval) {
        cout << "Usage: " << argv[0] << " [--codegen] [--eval]" << endl;
        cout << "  --codegen  Build CasADi symbolic functions, generate C code, compile .so" << endl;
        cout << "  --eval     Load compiled .so files, run timing benchmarks, save data" << endl;
        cout << "  Both flags can be combined." << endl;
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

            std::cout << "Compiling .so (3 in parallel)..." << std::flush;
            string par_cmd =
                "gcc -fPIC -shared -O3 -march=native " + codegen_dir + fn_case1  + ".c -o " + codegen_dir + fn_case1  + ".so & "
                "gcc -fPIC -shared -O3 -march=native " + codegen_dir + fn_case2a + ".c -o " + codegen_dir + fn_case2a + ".so & "
                "gcc -fPIC -shared -O3 -march=native " + codegen_dir + fn_case2b + ".c -o " + codegen_dir + fn_case2b + ".so & "
                "wait";
            int flag = system(par_cmd.c_str());
            std::cout << (flag == 0 ? " ok" : " FAILED") << std::endl;
            } // end else (not all_exist)
        }

        // ================================================================
        // EVAL PHASE
        // ================================================================
        if (do_eval) {
            std::cout << "\n--- Eval phase ---" << std::endl;

            // Load codegen'd functions
            std::cout << "Loading codegen .so files..." << std::endl;
            ::casadi::Function cg_case1  = ::casadi::external(fn_case1,  codegen_dir + fn_case1  + ".so");
            ::casadi::Function cg_case2a = ::casadi::external(fn_case2a, codegen_dir + fn_case2a + ".so");
            ::casadi::Function cg_case2b = ::casadi::external(fn_case2b, codegen_dir + fn_case2b + ".so");

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

            // Evaluation vectors
            std::vector<double> q_vec((size_t)nq);
            std::vector<double> v_int_vec((size_t)nv, 0.0);
            std::vector<double> v_vec((size_t)nv);
            std::vector<double> tau_vec((size_t)nv);
            std::vector<double> mu_vec((size_t)nv);

            int n_cases = 4;
            double time_cases[n_cases];

            // Temp matrices for analytical chain-rule
            MatrixXd d2tau_dqq(MatrixXd::Zero(nv, nv)), d2tau_dvv(MatrixXd::Zero(nv, nv));
            MatrixXd d2tau_dqv(MatrixXd::Zero(nv, nv)), d2tau_dqa(MatrixXd::Zero(nv, nv));

            std::cout << "\n--- Timing (" << NBT << " iterations) ---" << std::endl;

            // Case 3: Analytical (chain rule)
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
            time_cases[0] = timer.toc() / NBT;
            std::cout << "Case 3  (analytical):          " << time_cases[0] << " us" << std::endl;

            // Case 1: Full SO AD (codegen)
            timer.tic();
            SMOOTH(NBT)
            {
                Eigen::Map<ConfigVector>(q_vec.data(), nq, 1) = qs[_smooth];
                Eigen::Map<TangentVector>(v_vec.data(), nv, 1) = qdots[_smooth];
                Eigen::Map<TangentVector>(tau_vec.data(), nv, 1) = taus[_smooth];
                Eigen::Map<TangentVector>(mu_vec.data(), nv, 1) = mus[_smooth];

                auto res = cg_case1(::casadi::DMVector {q_vec, v_int_vec, v_vec, tau_vec, mu_vec});
            }
            time_cases[1] = timer.toc() / NBT;
            std::cout << "Case 1  (full SO AD, cg):      " << time_cases[1] << " us" << std::endl;

            // Case 2a: FO AD over full FO (codegen)
            timer.tic();
            SMOOTH(NBT)
            {
                Eigen::Map<ConfigVector>(q_vec.data(), nq, 1) = qs[_smooth];
                Eigen::Map<TangentVector>(v_vec.data(), nv, 1) = qdots[_smooth];
                Eigen::Map<TangentVector>(tau_vec.data(), nv, 1) = taus[_smooth];
                Eigen::Map<TangentVector>(mu_vec.data(), nv, 1) = mus[_smooth];

                auto res = cg_case2a(::casadi::DMVector {q_vec, v_int_vec, v_vec, tau_vec, mu_vec});
            }
            time_cases[2] = timer.toc() / NBT;
            std::cout << "Case 2a (AD full FO, cg):      " << time_cases[2] << " us" << std::endl;

            // Case 2b: FO AD over mod FO (codegen)
            timer.tic();
            SMOOTH(NBT)
            {
                Eigen::Map<ConfigVector>(q_vec.data(), nq, 1) = qs[_smooth];
                Eigen::Map<TangentVector>(v_vec.data(), nv, 1) = qdots[_smooth];
                Eigen::Map<TangentVector>(tau_vec.data(), nv, 1) = taus[_smooth];
                Eigen::Map<TangentVector>(mu_vec.data(), nv, 1) = mus[_smooth];

                auto res = cg_case2b(::casadi::DMVector {q_vec, v_int_vec, v_vec, tau_vec, mu_vec});
            }
            time_cases[3] = timer.toc() / NBT;
            std::cout << "Case 2b (AD mod FO, cg):       " << time_cases[3] << " us" << std::endl;

            // Speedups
            std::cout << "\nSpeedups vs analytical:" << std::endl;
            std::cout << "  Case 1  / Case 3 = " << time_cases[1] / time_cases[0] << "x" << std::endl;
            std::cout << "  Case 2a / Case 3 = " << time_cases[2] / time_cases[0] << "x" << std::endl;
            std::cout << "  Case 2b / Case 3 = " << time_cases[3] / time_cases[0] << "x" << std::endl;

            // Write timing data to file
            system("mkdir -p data/modFD_SO");
            string outfile = string("data/modFD_SO/") + robot_name + ".txt";
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
        }

    } // end model loop

    return 0;
}
