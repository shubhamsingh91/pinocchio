
/* Date created on- 8/29/22
* Modified on 10/14/22
Modified by- Shubham Singh, singh281@utexas.edu
version - ADmdof_v3

This version compares the CPU Runtime for

1. ABA FO analytical derivatives- Faster                               - Done
2. ABA FO partials using CppAD w/out codegen                           - Done (Not timed)
3. ABA FO partials using CasADi w/out codegen                          - Done (Not timed)
4. ABA SO partials using CppAD w/out codegen                           - Done (Not timed)
5. ABA SO partials using CasADi w/out codegen   + generating code      - Done (Not timed)
6. ABA SO partials using Analytical + generating code (+No run-time)   - Done (Not timed- NOT USED )

ToDo

*/

#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/centroidal.hpp"
#include "pinocchio/algorithm/aba.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/cholesky.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/compute-all-terms.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/parsers/sample-models.hpp"
#include "pinocchio/container/aligned-vector.hpp"
// #include "pinocchio/algorithm/rnea_SO_v7.hpp"
#include "pinocchio/algorithm/rnea-derivatives-faster.hpp"
#include "pinocchio/codegen/code-generator-algo.hpp"
#include <iostream>
#include "pinocchio/utils/timer.hpp"
#include <casadi/casadi.hpp>
#include "pinocchio/autodiff/casadi.hpp"
#include <dlfcn.h>
#include "pinocchio/algorithm/rnea-derivatives.hpp"
#include "pinocchio/algorithm/aba-derivatives.hpp"
#include "pinocchio/utils/tensor_utils.hpp"

using namespace std;
using namespace pinocchio;
using namespace pinocchio::casadi;

bool replace(std::string& str, const std::string& from, const std::string& to);

int main(int argc, const char* argv[])
{
    using namespace Eigen;

    using CppAD::AD;
    using CppAD::NearEqual;

    PinocchioTicToc timer(PinocchioTicToc::US);

#ifdef NDEBUG
    // int NBT= 1; // 50000 initially
    int NBT = 10000; // 50000 initially, then 1000*100
    int NBT_SO = 1;  // 50000 initially, then 1000*100

#else
    int NBT = 1;
    std::cout << "(the time score in debug mode is not relevant) " << std::endl;
#endif

    int n_models = 5;
    string str_robotname[n_models];

    // multi-case run

    str_robotname[0] = "double_pendulum"; // double pendulum
    // str_robotname[1] = "ur3_robot";       // UR3
    // str_robotname[2] = "hyq";             // hyq
    // str_robotname[3] = "baxter_simple";   // baxter_simple
    // str_robotname[4] = "atlas";           // atlas
    // str_robotname[5] = "talos_full_v2";   // talos_full_v2

    // single-case run
    // str_robotname[0] = "atlas"; // double pendulum

    char tmp[256];
    getcwd(tmp, 256);

    double time_vec[n_models];

    for (int mm = 0; mm < n_models; mm++) {

        Model model;
        bool with_ff = false;

        string str_file_ext;
        string robot_name = "";
        string str_urdf;

        robot_name = str_robotname[mm];
        std ::string filename = "../models/" + robot_name + std::string(".urdf");

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

        std::cout << "nq = " << model.nq << std::endl;
        std::cout << "nv = " << model.nv << std::endl;
        cout << "Model is" << robot_name << endl;

        //-- opening filename here
        ofstream file1;
        string filewrite = tmp;

        if (*argv[1] == 'c') {
         //   replace(filewrite, "pinocchio/benchmark/AD_cases", "Data/AD_v3/avx/clang/mdof/");
        } else if (*argv[1] == 'g') {
       //     replace(filewrite, "pinocchio/benchmark/AD_cases", "Data/AD_v3/avx/gcc/mdof/");
        } else {
        }

        filewrite.append(robot_name);
        filewrite.append(".txt");

        file1.open(filewrite);

        Data data(model);
        typedef Model::ConfigVectorType ConfigVector;
        typedef Model::TangentVectorType TangentVector;

        VectorXd qmax = Eigen::VectorXd::Ones(model.nq);

        PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) qs(NBT);
        PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) qdots(NBT);
        PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) taus(NBT);
        PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) qddots(NBT);

        // randomizing input data here

        for (size_t i = 0; i < NBT; ++i) {
            qs[i] = randomConfiguration(model, -qmax, qmax);
            qdots[i] = Eigen::VectorXd::Random(model.nv);
            qddots[i] = Eigen::VectorXd::Random(model.nv);
            taus[i] = Eigen::VectorXd::Random(model.nv);
        }

        int n_cases = 5;
        double time_ABA[n_cases];
        int code_gen = 1;
        int code_gen_FO = 1;
        int code_gen_ana_SO = 1;

        //-----------------------1 --------------------------//
        //----------------------------------------------------//
        // Compute RNEA FO derivatives faster-----------------//
        //----------------------------------------------------//

        Eigen::MatrixXd daba_dq(model.nv, model.nv);
        daba_dq.setZero();
        Eigen::MatrixXd daba_dv(model.nv, model.nv);
        daba_dv.setZero();
        Eigen::MatrixXd daba_dtau(model.nv, model.nv);
        daba_dtau.setZero();

        timer.tic();
        SMOOTH(NBT)
        {
            pinocchio::computeABADerivativesFaster(
                model, data, qs[_smooth], qdots[_smooth], taus[_smooth], daba_dq, daba_dv, daba_dtau);
        }
        time_ABA[0] = timer.toc() / NBT; // ABA timing
        std::cout << "ABA derivative = \t\t\t" << time_ABA[0] << endl;

        //-----------------------2 ---------------------------------//
        //----------------------------------------------------------//
        // Compute ABA FO derivatives using CppAD (w/out codegen)--//
        //----------------------------------------------------------//
        typedef double Scalar;
        typedef AD<Scalar> ADScalar;
        typedef pinocchio::ModelTpl<ADScalar> ADModel;
        typedef ADModel::Data ADData;
        ADModel ad_model = model.cast<ADScalar>();
        ADData ad_data(ad_model);

        typedef ADModel::ConfigVectorType ADConfigVectorType;
        typedef ADModel::TangentVectorType ADTangentVectorType;
        pinocchio::container::aligned_vector<ADConfigVectorType> ad_q(NBT);
        ADTangentVectorType ad_dq = ADTangentVectorType::Zero(model.nv);
        pinocchio::container::aligned_vector<ADConfigVectorType> ad_v(NBT);
        pinocchio::container::aligned_vector<ADConfigVectorType> ad_tau(NBT);
        typedef Eigen::Matrix<ADScalar, Eigen::Dynamic, 1> VectorXAD;
        ADConfigVectorType ad_q_plus;
        VectorXAD Y(model.nv);
        CPPAD_TESTVECTOR(Scalar) x((size_t)model.nv);

        for (size_t i = 0; i < NBT; ++i) {
            ad_q[i] = qs[i].cast<ADScalar>();
            ad_v[i] = qdots[i].cast<ADScalar>();
            ad_tau[i] = taus[i].cast<ADScalar>();
        }
        Data::MatrixXs dqdd_dq_mat;
        Data::MatrixXs dqdd_dv_mat;
        CPPAD_TESTVECTOR(Scalar) dqdd_dq_AD;
        Eigen::Map<Data::TangentVectorType>(x.data(), model.nv, 1).setZero();
        CPPAD_TESTVECTOR(Scalar) dqdd_dv_AD;

        timer.tic();
        SMOOTH(NBT_SO) // NBT_SO because don't need to time it, for timing it use NBT
        {
            CppAD::Independent(ad_dq);
            ad_q_plus = pinocchio::integrate(ad_model, ad_q[_smooth], ad_dq);
            pinocchio::aba(ad_model, ad_data, ad_q_plus, ad_v[_smooth], ad_tau[_smooth]);
            Eigen::Map<ADData::TangentVectorType>(Y.data(), model.nv, 1) = ad_data.ddq;
            CppAD::ADFun<Scalar> ad_fun1(ad_dq, Y);
            dqdd_dq_AD = ad_fun1.Jacobian(x);
            dqdd_dq_mat = Eigen::Map<PINOCCHIO_EIGEN_PLAIN_ROW_MAJOR_TYPE(Data::MatrixXs)>(
                dqdd_dq_AD.data(), model.nv, model.nv);

            CppAD::Independent(ad_v[_smooth]);
            pinocchio::aba(ad_model, ad_data, ad_q[_smooth], ad_v[_smooth], ad_tau[_smooth]);
            Eigen::Map<ADData::TangentVectorType>(Y.data(), model.nv, 1) = ad_data.ddq;
            CppAD::ADFun<Scalar> ad_fun2(ad_v[_smooth], Y);
            Eigen::Map<Data::TangentVectorType>(x.data(), model.nv, 1) = qdots[_smooth];
            dqdd_dv_AD = ad_fun2.Jacobian(x);
            dqdd_dv_mat = Eigen::Map<PINOCCHIO_EIGEN_PLAIN_ROW_MAJOR_TYPE(Data::MatrixXs)>(
                dqdd_dv_AD.data(), model.nv, model.nv);
        }
        time_ABA[1] = timer.toc() / NBT_SO; //
        std::cout << "FD FO partials using CppAD NO Codegen= " << time_ABA[1] << endl;

        //-----------------------3 ----------------------------------//
        //-----------------------------------------------------------//
        // Compute ABA FO derivatives using CasADi (w/out codegen)---//
        //-----------------------------------------------------------//

        typedef double Scalar;
        typedef ::casadi::SX ADcScalar;
        typedef pinocchio::ModelTpl<ADcScalar> ADcModel;
        typedef ADcModel::Data ADDcata;

        typedef ADcModel::ConfigVectorType ConfigVectorAD;
        typedef ADcModel::TangentVectorType TangentVectorAD;
        ADcModel adc_model = model.cast<ADcScalar>();
        ADDcata adc_data(adc_model);

        ::casadi::SX cs_q = ::casadi::SX::sym("q", model.nq);
        ::casadi::SX cs_v_int = ::casadi::SX::sym("v_inc", model.nv);
        ConfigVectorAD q_ad(model.nq), v_int_ad(model.nv), q_int_ad(model.nq);
        q_ad = Eigen::Map<ConfigVectorAD>(static_cast<std::vector<ADcScalar>>(cs_q).data(), model.nq, 1);
        v_int_ad = Eigen::Map<ConfigVectorAD>(static_cast<std::vector<ADcScalar>>(cs_v_int).data(), model.nv, 1);
        pinocchio::integrate(adc_model, q_ad, v_int_ad, q_int_ad);
        ::casadi::SX cs_q_int(model.nq, 1);
        pinocchio::casadi::copy(q_int_ad, cs_q_int);
        std::vector<double> q_vec((size_t)model.nq);
        std::vector<double> v_int_vec((size_t)model.nv);
        Eigen::Map<TangentVector>(v_int_vec.data(), model.nv, 1).setZero();

        ::casadi::SX cs_v = ::casadi::SX::sym("v", model.nv);
        TangentVectorAD v_ad(model.nv);
        v_ad = Eigen::Map<TangentVectorAD>(static_cast<std::vector<ADcScalar>>(cs_v).data(), model.nv, 1);

        ::casadi::SX cs_tau = ::casadi::SX::sym("tau", model.nv);
        TangentVectorAD tau_ad(model.nv);
        tau_ad = Eigen::Map<TangentVectorAD>(static_cast<std::vector<ADcScalar>>(cs_tau).data(), model.nv, 1);

        ::casadi::SX cs_a = ::casadi::SX::sym("a", model.nv);
        TangentVectorAD a_ad(model.nv);
        a_ad = Eigen::Map<TangentVectorAD>(static_cast<std::vector<ADcScalar>>(cs_a).data(), model.nv, 1);

        aba(adc_model, adc_data, q_int_ad, v_ad, tau_ad);
        ::casadi::SX cs_qdd(model.nv, 1);
        for (Eigen::DenseIndex k = 0; k < model.nv; ++k) {
            cs_qdd(k) = adc_data.ddq[k];
        }
        std::vector<double> v_vec((size_t)model.nv);
        std::vector<double> tau_vec((size_t)model.nv);

        // check with respect to q+dq
        std::string strfun_dqdddq = robot_name + std::string("_dqdd_dq");
        ::casadi::SX dqdd_dq = jacobian(cs_qdd, cs_v_int);
        ::casadi::Function eval_dqdd_dq(
            strfun_dqdddq, ::casadi::SXVector {cs_q, cs_v_int, cs_v, cs_tau}, ::casadi::SXVector {dqdd_dq});
        // check with respect to v+dv
        std::string strfun_dqdddv = robot_name + std::string("_dqdd_dv");
        ::casadi::SX dqdd_dv = jacobian(cs_qdd, cs_v);
        ::casadi::Function eval_dqdd_dv(
            strfun_dqdddv, ::casadi::SXVector {cs_q, cs_v_int, cs_v, cs_tau}, ::casadi::SXVector {dqdd_dv});
        // check with respect to a+da
        std::string strfun_dqdddtau = robot_name + std::string("_dqdd_dtau");
        ::casadi::SX dqdd_dtau = jacobian(cs_qdd, cs_tau);
        ::casadi::Function eval_dqdd_dtau(
            strfun_dqdddtau, ::casadi::SXVector {cs_q, cs_v_int, cs_v, cs_tau}, ::casadi::SXVector {dqdd_dtau});

        ::casadi::DM dqdd_dq_res, dqdd_dv_res, dqdd_dtau_res;
        Eigen::MatrixXd dqdd_dqADc(model.nv, model.nv), dqdd_dvADc(model.nv, model.nv),
            dqdd_dtauADc(model.nv, model.nv);
        std::vector<double> dqdd_dq_vec(model.nv), dqdd_dv_vec(model.nv), dqdd_dtau_vec(model.nv);
        int flag;
        if (code_gen_FO == 1) {
            cout << "Generating FO code" << endl; // generating function for dtau-dq
            eval_dqdd_dq.generate(strfun_dqdddq); // to generate the function
            // Compile the C-code to a shared library
            string compile_command
                = "gcc -fPIC -shared -O3 -march=native " + strfun_dqdddq + ".c -o " + strfun_dqdddq + ".so";
            flag = system(compile_command.c_str());
            // generating function for dtau-dv
            eval_dqdd_dv.generate(strfun_dqdddv); // to generate the function
            // Compile the C-code to a shared library
            compile_command
                = "gcc -fPIC -shared -O3 -march=native " + strfun_dqdddv + ".c -o " + strfun_dqdddv + ".so ";
            flag = system(compile_command.c_str());
            // generating function for dtau-da
            eval_dqdd_dtau.generate(strfun_dqdddtau); // to generate the function
            // Compile the C-code to a shared library
            compile_command
                = "gcc -fPIC -shared -O3 -march=native " + strfun_dqdddtau + ".c -o " + strfun_dqdddtau + ".so ";
            flag = system(compile_command.c_str());
        }

        timer.tic();
        SMOOTH(NBT_SO) // NBT_SO because don't need to time it, for timing it use NBT
        {
            Eigen::Map<ConfigVector>(q_vec.data(), model.nq, 1) = qs[_smooth];
            Eigen::Map<TangentVector>(v_vec.data(), model.nv, 1) = qdots[_smooth];
            Eigen::Map<TangentVector>(tau_vec.data(), model.nv, 1) = taus[_smooth];

            dqdd_dq_res = eval_dqdd_dq(::casadi::DMVector {q_vec, v_int_vec, v_vec, tau_vec})[0];
            dqdd_dq_vec = static_cast<std::vector<double>>(dqdd_dq_res);
            dqdd_dqADc = Eigen::Map<Data::MatrixXs>(dqdd_dq_vec.data(), model.nv, model.nv);

            dqdd_dv_res = eval_dqdd_dv(::casadi::DMVector {q_vec, v_int_vec, v_vec, tau_vec})[0];
            dqdd_dv_vec = static_cast<std::vector<double>>(dqdd_dv_res);
            dqdd_dvADc = Eigen::Map<Data::MatrixXs>(dqdd_dv_vec.data(), model.nv, model.nv);

            dqdd_dtau_res = eval_dqdd_dtau(::casadi::DMVector {q_vec, v_int_vec, v_vec, tau_vec})[0];
            dqdd_dtau_vec = static_cast<std::vector<double>>(dqdd_dtau_res);
            dqdd_dtauADc = Eigen::Map<Data::MatrixXs>(dqdd_dtau_vec.data(), model.nv, model.nv);
        }
        time_ABA[2] = timer.toc() / NBT_SO; //
        std::cout << "FD FO partials using CasADi NO Codegen= " << time_ABA[2] << endl;

        // //-----------------------4 ---------------------------//
        // //----------------------------------------------------//
        // // FD SO derivatives using CppAD (w/out codegen) -----//
        // //----------------------------------------------------//

        // VectorXAD Y_SO(model.nv * model.nv);

        // CPPAD_TESTVECTOR(Scalar) d2qdd_dq_ad, d2qdd_dv_ad, d2qdd_dqv_ad, d2qdd_dqtau_ad;
        // Eigen::Tensor<double, 3> d2qdd_dq2_AD(model.nv, model.nv, model.nv), d2qdd_dv2_AD(model.nv, model.nv, model.nv);
        // Eigen::Tensor<double, 3> d2qdd_dqv2_AD(model.nv, model.nv, model.nv),
        //     d2tau_dqtau2_AD(model.nv, model.nv, model.nv);
        // Eigen::VectorXd v1(model.nv);

        // timer.tic();
        // SMOOTH(NBT_SO) // NBT_SO because don't need to time it, for timing it use NBT
        // {
        //     // d2dqdd_dq2
        //     CppAD::Independent(ad_dq);
        //     ad_q_plus = pinocchio::integrate(ad_model, ad_q[_smooth], ad_dq);
        //     computeABADerivativesFaster(ad_model, ad_data, ad_q_plus, ad_v[_smooth], ad_tau[_smooth]);
        //     Eigen::Map<ADData::TangentVectorType>(Y_SO.data(), model.nv * model.nv, 1) = ad_data.ddq_dq;
        //     Eigen::Map<Data::TangentVectorType>(x.data(), model.nv, 1).setZero();
        //     CppAD::ADFun<Scalar> ad_fun1(ad_dq, Y_SO);
        //     d2qdd_dq_ad = ad_fun1.Jacobian(x);
        //     for (int k = 0; k < model.nv; k++) {
        //         for (int j = 0; j < model.nv; j++) {
        //             for (int i = 0; i < model.nv; i++) {
        //                 v1[i] = d2qdd_dq_ad[i + j * model.nv + k * model.nv * model.nv];
        //             }
        //             hess_assign(d2qdd_dq2_AD, v1, 0, k, j, 1, model.nv);
        //         }
        //     }

        //     // d2qdd_dv2
        //     CppAD::Independent(ad_v[_smooth]);
        //     computeABADerivativesFaster(ad_model, ad_data, ad_q[_smooth], ad_v[_smooth], ad_tau[_smooth]);
        //     Eigen::Map<ADData::TangentVectorType>(Y_SO.data(), model.nv * model.nv, 1) = ad_data.ddq_dq;
        //     Eigen::Map<Data::TangentVectorType>(x.data(), model.nv, 1) = qdots[_smooth];
        //     CppAD::ADFun<Scalar> ad_fun2(ad_v[_smooth], Y_SO);
        //     d2qdd_dv_ad = ad_fun2.Jacobian(x);
        //     for (int k = 0; k < model.nv; k++) {
        //         for (int j = 0; j < model.nv; j++) {
        //             for (int i = 0; i < model.nv; i++) {
        //                 v1[i] = d2qdd_dv_ad[i + j * model.nv + k * model.nv * model.nv];
        //             }
        //             hess_assign(d2qdd_dv2_AD, v1, 0, k, j, 1, model.nv);
        //         }
        //     }
        //     // d2qdd_dqdv
        //     CppAD::Independent(ad_dq);
        //     ad_q_plus = pinocchio::integrate(ad_model, ad_q[_smooth], ad_dq);
        //     computeABADerivativesFaster(ad_model, ad_data, ad_q_plus, ad_v[_smooth], ad_tau[_smooth]);
        //     Eigen::Map<ADData::TangentVectorType>(Y_SO.data(), model.nv * model.nv, 1) = ad_data.ddq_dv;
        //     CppAD::ADFun<Scalar> ad_fun3(ad_dq, Y_SO);
        //     Eigen::Map<Data::TangentVectorType>(x.data(), model.nv, 1).setZero();
        //     d2qdd_dqv_ad = ad_fun3.Jacobian(x);

        //     for (int k = 0; k < model.nv; k++) {
        //         for (int j = 0; j < model.nv; j++) {
        //             for (int i = 0; i < model.nv; i++) {
        //                 v1[i] = d2qdd_dqv_ad[i + j * model.nv + k * model.nv * model.nv];
        //             }
        //             hess_assign(d2qdd_dqv2_AD, v1, k, 0, j, 2, model.nv);
        //         }
        //     }
        //     // d2datu_dqdtau
        //     CppAD::Independent(ad_tau[_smooth]);
        //     computeABADerivativesFaster(ad_model, ad_data, ad_q[_smooth], ad_v[_smooth], ad_tau[_smooth]);
        //     Eigen::Map<ADData::TangentVectorType>(Y_SO.data(), model.nv * model.nv, 1) = ad_data.ddq_dq;
        //     CppAD::ADFun<Scalar> ad_fun4(ad_tau[_smooth], Y_SO);
        //     Eigen::Map<Data::TangentVectorType>(x.data(), model.nv, 1) = taus[_smooth];
        //     d2qdd_dqtau_ad = ad_fun4.Jacobian(x);

        //     for (int k = 0; k < model.nv; k++) {
        //         for (int j = 0; j < model.nv; j++) {
        //             for (int i = 0; i < model.nv; i++) {
        //                 v1[i] = d2qdd_dqtau_ad[i + j * model.nv + k * model.nv * model.nv];
        //             }
        //             hess_assign(d2tau_dqtau2_AD, v1, 0, j, k, 1, model.nv);
        //         }
        //     }
        // }
        // time_ABA[3] = timer.toc() / NBT_SO; //
        // std::cout << "FD SO partials using CppAD NO Codegen= \t" << time_ABA[3] << endl;

        //-----------------------5 ---------------------------//
        //----------------------------------------------------//
        // FD SO derivatives using CasADi (w/out codegen) ----//
        //----------------------------------------------------//

        computeABADerivativesFaster(adc_model, adc_data, q_int_ad, v_ad, tau_ad);
        ::casadi::SX cs_dqdd_dq(model.nv, model.nv), cs_dqdd_dv(model.nv, model.nv), cs_dqdd_dtau(model.nv, model.nv);

        for (Eigen::DenseIndex i = 0; i < model.nv; ++i) {
            for (Eigen::DenseIndex j = 0; j < model.nv; ++j) {
                cs_dqdd_dq(i, j) = adc_data.ddq_dq(i, j);
                cs_dqdd_dv(i, j) = adc_data.ddq_dv(i, j);
                cs_dqdd_dtau(i, j) = adc_data.Minv(i, j);
            }
        }
        std::string strfun_d2qdddq = robot_name + std::string("_d2qdd_dq");
        ::casadi::SX d2qdd_dq = jacobian(cs_dqdd_dq, cs_v_int);
        std::string strfun_d2qdddv = robot_name + std::string("_d2qdd_dv");
        ::casadi::SX d2qdd_dv = jacobian(cs_dqdd_dv, cs_v);
        std::string strfun_d2qdddqv = robot_name + std::string("_d2qdd_dqv");
        ::casadi::SX d2qdd_dqv = jacobian(cs_dqdd_dq, cs_v);
        std::string strfun_d2qdddtauq = robot_name + std::string("_d2qdd_dtauq");
        ::casadi::SX d2qdd_dtauq = jacobian(cs_dqdd_dtau, cs_v_int);

        ::casadi::Function eval_d2qdd_dq(
            strfun_d2qdddq, ::casadi::SXVector {cs_q, cs_v_int, cs_v, cs_tau}, ::casadi::SXVector {d2qdd_dq});
        ::casadi::Function eval_d2qdd_dv(
            strfun_d2qdddv, ::casadi::SXVector {cs_q, cs_v_int, cs_v, cs_tau}, ::casadi::SXVector {d2qdd_dv});
        ::casadi::Function eval_d2qdd_dqv(
            strfun_d2qdddqv, ::casadi::SXVector {cs_q, cs_v_int, cs_v, cs_tau}, ::casadi::SXVector {d2qdd_dqv});
        ::casadi::Function eval_d2qdd_dtauq(
            strfun_d2qdddtauq, ::casadi::SXVector {cs_q, cs_v_int, cs_v, cs_tau}, ::casadi::SXVector {d2qdd_dtauq});

        std::vector<double> v2(model.nv * model.nv), v3(model.nv * model.nv);
        std::vector<double> v4(model.nv * model.nv), v5(model.nv * model.nv);

        Eigen::MatrixXd mat1(model.nv, model.nv);
        Eigen::MatrixXd mat2(model.nv, model.nv);
        Eigen::MatrixXd mat3(model.nv, model.nv);
        Eigen::MatrixXd mat4(model.nv, model.nv);
        ::casadi::DM d2qdd_dq_res, d2qdd_dv_res, d2qdd_dqv_res, d2qdd_dtauq_res;
        std::vector<double> d2qdd_dq_vec, d2qdd_dv_vec, d2qdd_dqv_vec, d2qdd_dtauq_vec;
        Eigen::Tensor<double, 3> d2qdd_dq2_ADc(model.nv, model.nv, model.nv);
        Eigen::Tensor<double, 3> d2qdd_dv2_ADc(model.nv, model.nv, model.nv);
        Eigen::Tensor<double, 3> d2qdd_dqv_ADc(model.nv, model.nv, model.nv);
        Eigen::Tensor<double, 3> d2qdd_dtauq_ADc(model.nv, model.nv, model.nv);
        int n2 = model.nv * model.nv;

        timer.tic();
        SMOOTH(NBT_SO) // NBT_SO because don't need to time it, for timing it use NBT
        {
            Eigen::Map<ConfigVector>(q_vec.data(), model.nq, 1) = qs[_smooth];
            Eigen::Map<TangentVector>(v_vec.data(), model.nv, 1) = qdots[_smooth];
            Eigen::Map<TangentVector>(tau_vec.data(), model.nv, 1) = taus[_smooth];

            d2qdd_dq_res = eval_d2qdd_dq(::casadi::DMVector {q_vec, v_int_vec, v_vec, tau_vec})[0];
            d2qdd_dq_vec = static_cast<std::vector<double>>(d2qdd_dq_res);
            d2qdd_dv_res = eval_d2qdd_dv(::casadi::DMVector {q_vec, v_int_vec, v_vec, tau_vec})[0];
            d2qdd_dv_vec = static_cast<std::vector<double>>(d2qdd_dv_res);
            d2qdd_dqv_res = eval_d2qdd_dqv(::casadi::DMVector {q_vec, v_int_vec, v_vec, tau_vec})[0];
            d2qdd_dqv_vec = static_cast<std::vector<double>>(d2qdd_dqv_res);
            d2qdd_dtauq_res = eval_d2qdd_dtauq(::casadi::DMVector {q_vec, v_int_vec, v_vec, tau_vec})[0];
            d2qdd_dtauq_vec = static_cast<std::vector<double>>(d2qdd_dtauq_res);
            for (int j = 0; j < model.nv; j++) {
                for (int i = 0; i < n2; i++) {
                    v2[i] = d2qdd_dq_vec[j + i * model.nv];
                    v3[i] = d2qdd_dv_vec[j + i * model.nv];
                    v4[i] = d2qdd_dqv_vec[j + i * model.nv];
                }

                mat1 = Eigen::Map<Eigen::Matrix<double, Dynamic, Dynamic>>(v2.data(), model.nv, model.nv);
                hess_assign_fd_v1(d2qdd_dq2_ADc, mat1, model.nv, j);
                mat2 = Eigen::Map<Eigen::Matrix<double, Dynamic, Dynamic>>(v3.data(), model.nv, model.nv);
                hess_assign_fd_v1(d2qdd_dv2_ADc, mat2, model.nv, j);
                mat3 = Eigen::Map<Eigen::Matrix<double, Dynamic, Dynamic, RowMajor>>(v4.data(), model.nv, model.nv);
                hess_assign_fd_v1(d2qdd_dqv_ADc, mat3, model.nv, j);

                for (int i = 0; i < n2; i++) {
                    v5[i] = d2qdd_dtauq_vec[j * n2 + i];
                }
                mat4 = Eigen::Map<Eigen::Matrix<double, Dynamic, Dynamic>>(v5.data(), model.nv, model.nv);
                hess_assign_fd_v1(d2qdd_dtauq_ADc, mat4, model.nv, j);
            }
        }
        time_ABA[4] = timer.toc() / NBT_SO; //
        std::cout << "FD SO partials using CasADi NO Codegen = " << time_ABA[4] << endl;

        // Code-generation for SO partials
        if (code_gen == 1) { // generating function for dt2au-dq
            cout << "Generating SO code" << endl;
            // for q
            eval_d2qdd_dq.generate(strfun_d2qdddq); // to generate the function
            // Compile the C-code to a shared library
            string compile_command
                = "gcc -fPIC -shared -O3 -march=native " + strfun_d2qdddq + ".c -o " + strfun_d2qdddq + ".so ";
            flag = system(compile_command.c_str());
            // for v
            eval_d2qdd_dv.generate(strfun_d2qdddv); // to generate the function
            // Compile the C-code to a shared library
            compile_command
                = "gcc -fPIC -shared -O3 -march=native " + strfun_d2qdddv + ".c -o " + strfun_d2qdddv + ".so ";
            flag = system(compile_command.c_str());
            // for q,v
            eval_d2qdd_dqv.generate(strfun_d2qdddqv); // to generate the function
            // Compile the C-code to a shared library
            compile_command
                = "gcc -fPIC -shared -O3 -march=native " + strfun_d2qdddqv + ".c -o " + strfun_d2qdddqv + ".so ";
            flag = system(compile_command.c_str());
            // for q,a
            eval_d2qdd_dtauq.generate(strfun_d2qdddtauq); // to generate the function
            // Compile the C-code to a shared library
            compile_command
                = "gcc -fPIC -shared -O3 -march=native " + strfun_d2qdddtauq + ".c -o " + strfun_d2qdddtauq + ".so ";
            flag = system(compile_command.c_str());
        }

        //-----------------------6 ---------------------------//
        //----------------------------------------------------//
        // FD SO derivatives using Analytical+ generating code//
        //----------------------------------------------------//
        //--------------NOT TIMED ----------------------------//
        //--------------NOT USED ----------------------------//

        // computeRNEA_SO_v7(adc_model, adc_data, q_int_ad, v_ad, a_ad);
        // // std::string strfun_d2tdq_ana_cg = robot_name + std::string("_d2tau_dq_ana_cg");
        // // std::string strfun_d2tdv_ana_cg = robot_name + std::string("_d2tau_dv_ana_cg");
        // // std::string strfun_d2tdqv_ana_cg = robot_name + std::string("_d2tau_dqv_ana_cg");
        // // std::string strfun_MFO_ana_cg = robot_name + std::string("_MFO_ana_cg");

        // ::casadi::SX cs_d2tau_dq_ana_cg(model.nv, model.nv * model.nv);
        // ::casadi::SX cs_d2tau_dv_ana_cg(model.nv, model.nv * model.nv);
        // ::casadi::SX cs_d2tau_dqv_ana_cg(model.nv, model.nv * model.nv);
        // ::casadi::SX cs_MFO_ana_cg(model.nv, model.nv * model.nv);
        // ::casadi::SX cas_mat1(model.nv, model.nv), cas_mat2(model.nv, model.nv), cas_mat3(model.nv, model.nv),
        //     cas_mat4(model.nv, model.nv);

        // for (int k = 0; k < model.nv; k++) {
        //     get_mat_from_tens3_v2(adc_data.d2tau_dqdq, cas_mat1, model.nv, k);
        //     get_mat_from_tens3_v2(adc_data.d2tau_dvdv, cas_mat2, model.nv, k);
        //     get_mat_from_tens3_v2(adc_data.d2tau_dqdv, cas_mat3, model.nv, k);
        //     get_mat_from_tens3_v2(adc_data.M_FO, cas_mat4, model.nv, k);

        //     for (int j = 0; j < model.nv; j++) {
        //         for (int i = 0; i < model.nv; i++) {
        //             cs_d2tau_dq_ana_cg(i, j + k * model.nv) = cas_mat1(i, j);
        //             cs_d2tau_dv_ana_cg(i, j + k * model.nv) = cas_mat2(i, j);
        //             cs_d2tau_dqv_ana_cg(i, j + k * model.nv) = cas_mat3(i, j);
        //             cs_MFO_ana_cg(i, j + k * model.nv) = cas_mat4(i, j);
        //         }
        //     }
        // }

        // ::casadi::SX Minv_cg(model.nv, model.nv);
        // ::casadi::SX Minv_neg_cg(model.nv, model.nv);

        // computeMinverse(adc_model, adc_data, q_int_ad);
        // for (int i = 0; i < model.nv; i++) {
        //     for (int j = 0; j <= i; j++) {
        //         Minv_cg(j, i) = adc_data.Minv(j, i);
        //         Minv_cg(i, j) = Minv_cg(j, i);
        //     }
        // }
        // Minv_neg_cg = -Minv_cg;

        // ::casadi::SX prodq_v2_cg(model.nv, model.nv * model.nv);
        // ::casadi::SX prodqd_v2_cg(model.nv, model.nv * model.nv);
        // ::casadi::SX prodqdd_v2_cg(model.nv, model.nv * model.nv);
        // ::casadi::SX mat1_cg(model.nv, model.nv), mat2_cg(model.nv, model.nv), mat3_cg(model.nv, model.nv);
        // ::casadi::SX vec1_cg(model.nv, 1), vec2_cg(model.nv, 1);
        // ::casadi::SX vec3_cg(model.nv, 1), vec4_cg(model.nv, 1);

        // // Inner term Compute using DTM (or DMM)
        // for (int u1 = 0; u1 < model.nv; u1++) {
        //     get_mat_from_flattens3_v2(cs_MFO_ana_cg, mat1_cg, model.nv, u1);
        //     for (int u2 = 0; u2 < model.nv; u2++) {
        //         get_mat_from_flattens3_v2(cs_MFO_ana_cg, mat2_cg, model.nv, u2);

        //         get_col(cs_dqdd_dq, vec3_cg, model.nv, u2);
        //         get_col(cs_dqdd_dv, vec4_cg, model.nv, u1);
        //         vec1_cg = mtimes(mat1_cg, vec3_cg);
        //         vec2_cg = mtimes(mat2_cg, vec4_cg);
        //         hess_assign_flat(prodq_v2_cg, vec1_cg, 0, u1, u2, 1, model.nv);
        //         hess_assign_flat(prodqd_v2_cg, vec2_cg, 0, u2, u1, 1, model.nv);
        //     }

        //     mat3_cg = mtimes(mat1_cg, Minv_cg);
        //     flattens_assign_fd2(prodqdd_v2_cg, mat3_cg, model.nv, u1);
        // }
        // // SO partials of qdd w.r.t q
        // ::casadi::SX mat_aza_in_q_v2_cg(model.nv, model.nv * model.nv);
        // ::casadi::SX mat_aza_out_q_v2_cg(model.nv, model.nv * model.nv);

        // ::casadi::SX mat_aza_in_v_v2_cg(model.nv, model.nv * model.nv);
        // ::casadi::SX mat_aza_out_v_v2_cg(model.nv, model.nv * model.nv);

        // ::casadi::SX mat_aza_in_qv_v2_cg(model.nv, model.nv * model.nv);
        // ::casadi::SX mat_aza_out_qv_v2_cg(model.nv, model.nv * model.nv);

        // ::casadi::SX mat_aza_in_qtau_v2_cg(model.nv, model.nv * model.nv);
        // ::casadi::SX mat_aza_out_qtau_v2_cg(model.nv, model.nv * model.nv);

        // // Inner term addition using single loop
        // for (int u = 0; u < model.nv; u++) {
        //     // partial w.r.t q
        //     get_mat_from_flattens3_v2(cs_d2tau_dq_ana_cg, mat1_cg, model.nv, u);
        //     get_mat_from_flat_tens2(prodq_v2_cg, mat2_cg, model.nv, u);
        //     get_mat_from_flattens3_v2(prodq_v2_cg, mat3_cg, model.nv, u);
        //     mat1_cg += mat2_cg + mat3_cg;
        //     mat_middle_cols(mat_aza_in_q_v2_cg, mat1_cg, model.nv, model.nv, u * model.nv);
        //     // partial w.r.t qd
        //     get_mat_from_flattens3_v2(cs_d2tau_dv_ana_cg, mat2_cg, model.nv, u);
        //     mat_middle_cols(mat_aza_in_v_v2_cg, mat2_cg, model.nv, model.nv, u * model.nv);
        //     // partial w.r.t q/qd
        //     get_mat_from_flattens3_v2(cs_d2tau_dqv_ana_cg, mat3_cg, model.nv, u);
        //     get_mat_from_flattens3_v2(prodqd_v2_cg, mat2_cg, model.nv, u);
        //     mat3_cg += mat2_cg;
        //     mat_middle_cols(mat_aza_in_qv_v2_cg, mat3_cg, model.nv, model.nv, u * model.nv);
        //     // partial w.r.t q/tau
        //     get_mat_from_flattens3_v2(prodqdd_v2_cg, mat1_cg, model.nv, u);
        //     mat_middle_cols(mat_aza_in_qtau_v2_cg, mat1_cg, model.nv, model.nv, u * model.nv);
        // }

        // mat_aza_out_q_v2_cg = mtimes(Minv_neg_cg, mat_aza_in_q_v2_cg);
        // mat_aza_out_v_v2_cg = mtimes(Minv_neg_cg, mat_aza_in_v_v2_cg);
        // mat_aza_out_qv_v2_cg = mtimes(Minv_neg_cg, mat_aza_in_qv_v2_cg);
        // mat_aza_out_qtau_v2_cg = mtimes(Minv_neg_cg, mat_aza_in_qtau_v2_cg);

        // std::string strfun_d2qdddq_ana_cg = robot_name + std::string("_d2qdd_dq_ana_cg");
        // ::casadi::Function eval_d2qdd_dq_ana_cg(strfun_d2qdddq_ana_cg,
        //     ::casadi::SXVector {cs_q, cs_v_int, cs_v, cs_tau, cs_a}, ::casadi::SXVector {mat_aza_out_q_v2_cg});

        // std::string strfun_d2qdddv_ana_cg = robot_name + std::string("_d2qdd_dv_ana_cg");
        // ::casadi::Function eval_d2qdd_dv_ana_cg(strfun_d2qdddv_ana_cg,
        //     ::casadi::SXVector {cs_q, cs_v_int, cs_v, cs_tau, cs_a}, ::casadi::SXVector {mat_aza_out_v_v2_cg});

        // std::string strfun_d2qdddqv_ana_cg = robot_name + std::string("_d2qdd_dqv_ana_cg");
        // ::casadi::Function eval_d2qdd_dqv_ana_cg(strfun_d2qdddqv_ana_cg,
        //     ::casadi::SXVector {cs_q, cs_v_int, cs_v, cs_tau, cs_a}, ::casadi::SXVector {mat_aza_out_qv_v2_cg});

        // std::string strfun_d2qdddqtau_ana_cg = robot_name + std::string("_d2qdd_dqtau_ana_cg");
        // ::casadi::Function eval_d2qdd_dqtau_ana_cg(strfun_d2qdddqtau_ana_cg,
        //     ::casadi::SXVector {cs_q, cs_v_int, cs_v, cs_tau, cs_a}, ::casadi::SXVector {mat_aza_out_qtau_v2_cg});

        // // Code-generation for SO partials
        // if (code_gen_ana_SO == 1) { // generating function for dt2au-dq
        //     cout << "Generating ANA SO code" << endl;
        //     // for q
        //     eval_d2qdd_dq_ana_cg.generate(strfun_d2qdddq_ana_cg); // to generate the function
        //     // Compile the C-code to a shared library
        //     string compile_command = "gcc -fPIC -shared -O3 -march=native " + strfun_d2qdddq_ana_cg + ".c -o "
        //                              + strfun_d2qdddq_ana_cg + ".so ";
        //     flag = system(compile_command.c_str());
        //     // for v
        //     eval_d2qdd_dv_ana_cg.generate(strfun_d2qdddv_ana_cg); // to generate the function
        //                                                           // Compile the C-code to a shared library
        //     compile_command = "gcc -fPIC -shared -O3 -march=native " + strfun_d2qdddv_ana_cg + ".c -o "
        //                       + strfun_d2qdddv_ana_cg + ".so ";
        //     flag = system(compile_command.c_str());
        //     // for q,v
        //     eval_d2qdd_dqv_ana_cg.generate(strfun_d2qdddqv_ana_cg); // to generate the function
        //                                                             // Compile the C-code to a shared library
        //     compile_command = "gcc -fPIC -shared -O3 -march=native " + strfun_d2qdddqv_ana_cg + ".c -o "
        //                       + strfun_d2qdddqv_ana_cg + ".so ";
        //     flag = system(compile_command.c_str());
        //     // for q,tau
        //     eval_d2qdd_dqtau_ana_cg.generate(strfun_d2qdddqtau_ana_cg); // to generate the function
        //                                                                 // Compile the C-code to a shared library
        //     compile_command = "gcc -fPIC -shared -O3 -march=native " + strfun_d2qdddqtau_ana_cg + ".c -o "
        //                       + strfun_d2qdddqtau_ana_cg + ".so ";
        //     flag = system(compile_command.c_str());
        // }

        //------------------------------------------------//
        // Writing all the timings to the file
        //------------------------------------------------//

        // for (int ii = 0; ii < n_cases; ii++) {
        //     file1 << time_ABA[ii] << endl;
        // }
        // file1.close();
    }

    return 0;
}

bool replace(std::string& str, const std::string& from, const std::string& to)
{
    size_t start_pos = str.find(from);
    if (start_pos == std::string::npos)
        return false;
    str.replace(start_pos, from.length(), to);
    return true;
}