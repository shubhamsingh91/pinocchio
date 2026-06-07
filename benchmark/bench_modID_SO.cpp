/* bench_modID_SO.cpp
 * Modified ID Second-Order Derivatives Benchmark (with CasADi codegen)
 *
 * Compares 6 approaches for computing modified SO derivatives:
 *
 * FO:      FO analytical — computeRNEADerivativesFaster() (baseline from RAL)
 * Full SO: Full SO analytical — ComputeRNEASecondOrderDerivatives() (tensor, from T-Ro)
 * Mod SO:  Mod SO analytical — computeModRNEASecondOrderDerivatives() (our algo)
 * Case 1:  Full SO AD — trace modrnea() with CasADi codegen (2 AD diffs)
 * Case 2a: FO AD over full FO — CasADi codegen (1 AD diff)
 * Case 2b: FO AD over mod FO — CasADi codegen (1 AD diff)
 *
 * Usage:
 *   ./bench_modID_SO --codegen              Build CasADi .so files
 *   ./bench_modID_SO --eval                 Run timing, save data
 *   ./bench_modID_SO --eval --outdir <dir>  Save data to alternate directory
 */

#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/modrnea.hpp"
#include "pinocchio/algorithm/rnea-derivatives-faster.hpp"
#include "pinocchio/algorithm/mod-rnea-derivatives.hpp"
#include "pinocchio/codegen/code-generator-algo.hpp"
#include "pinocchio/algorithm/mod-rnea-second-order-derivatives.hpp"
#include "pinocchio/algorithm/rnea-second-order-derivatives.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/container/aligned-vector.hpp"
#include "pinocchio/utils/timer.hpp"

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
    string outdir = "data/modID_SO";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--codegen") == 0) do_codegen = true;
        if (strcmp(argv[i], "--eval") == 0)    do_eval = true;
        if (strcmp(argv[i], "--outdir") == 0 && i + 1 < argc) outdir = argv[++i];
    }

    if (!do_codegen && !do_eval) {
        cout << "Usage: " << argv[0] << " [--codegen] [--eval] [--outdir <dir>]" << endl;
        cout << "  --codegen       Build CasADi symbolic functions, generate C code, compile .so" << endl;
        cout << "  --eval          Load compiled .so files, run timing benchmarks, save data" << endl;
        cout << "  --outdir <dir>  Output directory for timing data (default: data/modID_SO)" << endl;
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
        string fn_case1  = robot_name + "_modID_SO_case1";
        string fn_case2a = robot_name + "_modID_SO_case2a";
        string fn_case2b = robot_name + "_modID_SO_case2b";

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

            ::casadi::SXVector cs_inputs = {cs_q, cs_v_int, cs_v, cs_a, cs_lambda};

            // Case 1: Full SO AD — trace modrnea, take Hessian
            std::cout << "Case 1: trace modrnea..." << std::flush;
            ADcData adc_data1(adc_model);
            modrnea(adc_model, adc_data1, q_int_ad, v_ad, a_ad, lambda_ad);
            ::casadi::SX cs_modtau = adc_data1.modtau;

            ::casadi::SX grad_q_1 = jacobian(cs_modtau, cs_v_int);
            ::casadi::SX grad_v_1 = jacobian(cs_modtau, cs_v);

            ::casadi::Function eval_case1(
                fn_case1, cs_inputs,
                ::casadi::SXVector {
                    jacobian(grad_q_1, cs_v_int),
                    jacobian(grad_v_1, cs_v),
                    jacobian(grad_q_1, cs_v),
                    jacobian(grad_q_1, cs_a)
                });
            std::cout << " done" << std::endl;

            // Case 2a: FO AD over computeRNEADerivativesFaster + contract
            std::cout << "Case 2a: trace RNEA FO + contract..." << std::flush;
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
                fn_case2a, cs_inputs,
                ::casadi::SXVector {
                    jacobian(g_q_2a, cs_v_int),
                    jacobian(g_v_2a, cs_v),
                    jacobian(g_q_2a, cs_v),
                    jacobian(g_q_2a, cs_a)
                });
            std::cout << " done" << std::endl;

            // Case 2b: FO AD over computeModRNEADerivatives
            std::cout << "Case 2b: trace modRNEA FO..." << std::flush;
            ADcData adc_data2b(adc_model);
            pinocchio::computeModRNEADerivatives(adc_model, adc_data2b, q_int_ad, v_ad, a_ad, lambda_ad);

            ::casadi::SX g_q_2b(nv, 1), g_v_2b(nv, 1);
            for (int i = 0; i < nv; i++) {
                g_q_2b(i) = adc_data2b.dtau_dq_mod[i];
                g_v_2b(i) = adc_data2b.dtau_dv_mod[i];
            }

            ::casadi::Function eval_case2b(
                fn_case2b, cs_inputs,
                ::casadi::SXVector {
                    jacobian(g_q_2b, cs_v_int),
                    jacobian(g_v_2b, cs_v),
                    jacobian(g_q_2b, cs_v),
                    jacobian(g_q_2b, cs_a)
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
            PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) lambdas(NBT);

            for (size_t i = 0; i < (size_t)NBT; ++i) {
                qs[i] = randomConfiguration(model, -qmax, qmax);
                qdots[i] = VectorXd::Random(nv);
                qddots[i] = VectorXd::Random(nv);
                lambdas[i] = VectorXd::Random(nv);
            }

            // Evaluation vectors
            std::vector<double> q_vec((size_t)nq);
            std::vector<double> v_int_vec((size_t)nv, 0.0);
            std::vector<double> v_vec((size_t)nv);
            std::vector<double> a_vec((size_t)nv);
            std::vector<double> lambda_vec((size_t)nv);

            // 6 timing slots: FO, Full SO, Mod SO, Case1, Case2a, Case2b
            double time_cases[6] = {-1, -1, -1, -1, -1, -1};

            // Temp matrices for mod SO analytical
            MatrixXd dqq_ana(MatrixXd::Zero(nv, nv)), dvv_ana(MatrixXd::Zero(nv, nv));
            MatrixXd dvq_ana(MatrixXd::Zero(nv, nv)), dqa_ana(MatrixXd::Zero(nv, nv));

            // Tensors for full SO analytical (allocate once outside loop)
            Eigen::Tensor<double, 3> d2tau_dqdq(nv, nv, nv);
            Eigen::Tensor<double, 3> d2tau_dvdv(nv, nv, nv);
            Eigen::Tensor<double, 3> d2tau_dqdv(nv, nv, nv);
            Eigen::Tensor<double, 3> d2tau_dadq(nv, nv, nv);

            std::cout << "\n--- Timing (" << NBT << " iterations) ---" << std::endl;

            // FO analytical — computeRNEADerivativesFaster
            timer.tic();
            SMOOTH(NBT)
            {
                computeRNEADerivativesFaster(model, data, qs[_smooth], qdots[_smooth], qddots[_smooth]);
            }
            time_cases[0] = timer.toc() / NBT;
            std::cout << "FO analytical:                 " << time_cases[0] << " us" << std::endl;

            // Full SO analytical — ComputeRNEASecondOrderDerivatives (tensor)
            timer.tic();
            SMOOTH(NBT)
            {
                d2tau_dqdq.setZero(); d2tau_dvdv.setZero();
                d2tau_dqdv.setZero(); d2tau_dadq.setZero();
                ComputeRNEASecondOrderDerivatives(model, data, qs[_smooth], qdots[_smooth], qddots[_smooth],
                    d2tau_dqdq, d2tau_dvdv, d2tau_dqdv, d2tau_dadq);
            }
            time_cases[1] = timer.toc() / NBT;
            std::cout << "Full SO analytical (tensor):   " << time_cases[1] << " us" << std::endl;

            // Mod SO analytical — computeModRNEASecondOrderDerivatives
            timer.tic();
            SMOOTH(NBT)
            {
                dqq_ana.setZero(); dvv_ana.setZero(); dvq_ana.setZero(); dqa_ana.setZero();
                computeModRNEASecondOrderDerivatives(model, data, qs[_smooth], qdots[_smooth], qddots[_smooth],
                    lambdas[_smooth], dqq_ana, dvv_ana, dvq_ana, dqa_ana);
            }
            time_cases[2] = timer.toc() / NBT;
            std::cout << "Mod SO analytical:             " << time_cases[2] << " us" << std::endl;

            // Case 1: Full SO AD (codegen)
            if (has_case1) {
                timer.tic();
                SMOOTH(NBT)
                {
                    Eigen::Map<ConfigVector>(q_vec.data(), nq, 1) = qs[_smooth];
                    Eigen::Map<TangentVector>(v_vec.data(), nv, 1) = qdots[_smooth];
                    Eigen::Map<TangentVector>(a_vec.data(), nv, 1) = qddots[_smooth];
                    Eigen::Map<TangentVector>(lambda_vec.data(), nv, 1) = lambdas[_smooth];
                    auto res = cg_case1(::casadi::DMVector {q_vec, v_int_vec, v_vec, a_vec, lambda_vec});
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
                    Eigen::Map<TangentVector>(a_vec.data(), nv, 1) = qddots[_smooth];
                    Eigen::Map<TangentVector>(lambda_vec.data(), nv, 1) = lambdas[_smooth];
                    auto res = cg_case2a(::casadi::DMVector {q_vec, v_int_vec, v_vec, a_vec, lambda_vec});
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
                    Eigen::Map<TangentVector>(a_vec.data(), nv, 1) = qddots[_smooth];
                    Eigen::Map<TangentVector>(lambda_vec.data(), nv, 1) = lambdas[_smooth];
                    auto res = cg_case2b(::casadi::DMVector {q_vec, v_int_vec, v_vec, a_vec, lambda_vec});
                }
                time_cases[5] = timer.toc() / NBT;
                std::cout << "Case 2b (AD mod FO, cg):       " << time_cases[5] << " us" << std::endl;
            } else {
                std::cout << "Case 2b (AD mod FO, cg):       SKIPPED (no .so)" << std::endl;
            }

            // Speedups vs Mod SO
            std::cout << "\nSpeedups vs Mod SO analytical:" << std::endl;
            std::cout << "  FO     / Mod SO = " << time_cases[0] / time_cases[2] << "x" << std::endl;
            std::cout << "  Full SO/ Mod SO = " << time_cases[1] / time_cases[2] << "x" << std::endl;
            if (has_case1)  std::cout << "  Case 1 / Mod SO = " << time_cases[3] / time_cases[2] << "x" << std::endl;
            else            std::cout << "  Case 1 / Mod SO = N/A" << std::endl;
            if (has_case2a) std::cout << "  Case 2a/ Mod SO = " << time_cases[4] / time_cases[2] << "x" << std::endl;
            else            std::cout << "  Case 2a/ Mod SO = N/A" << std::endl;
            if (has_case2b) std::cout << "  Case 2b/ Mod SO = " << time_cases[5] / time_cases[2] << "x" << std::endl;
            else            std::cout << "  Case 2b/ Mod SO = N/A" << std::endl;

            // Write timing data (6-line format, -1 for unavailable)
            system(("mkdir -p " + outdir).c_str());
            string outfile = outdir + "/" + robot_name + ".txt";
            std::ofstream ofs(outfile);
            if (ofs.is_open()) {
                ofs << time_cases[0] << std::endl;  // FO analytical
                ofs << time_cases[1] << std::endl;  // Full SO analytical
                ofs << time_cases[2] << std::endl;  // Mod SO analytical
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
