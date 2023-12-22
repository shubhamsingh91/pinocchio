/* Date created on- 8/22/22
* Modified on 1/3/23
Modified by- Shubham Singh, singh281@utexas.edu
version - ADmdof_v1

This version compares the CPU Runtime for

1. RNEA FO analytical derivatives- Faster                        - Done (TIMED)
2. RNEA FO partials using CppAD w/out codegen                    - Done (NOT TIMED)
3. RNEA FO partials using CasADi w/out codegen + generating code - Done (NOT TIMED)
4. RNEA SO analytical derivatives -                              - Done (TIMED)
5. RNEA SO partials using CppAD w/out codegen                    - Done (NOT TIMED)
6. RNEA SO partials using CasADi w/out codegen + generating code - Done (TIMED)
7. RNEA SO partials using IDSVA + generating code (No run-time)  - Done (NOT TIMED)

// Modifications
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
#include "pinocchio/algorithm/rnea-derivatives-SO.hpp"
#include "pinocchio/algorithm/rnea-derivatives-faster.hpp"
#include "pinocchio/codegen/code-generator-algo.hpp"
#include <iostream>
#include "pinocchio/utils/timer.hpp"
#include <casadi/casadi.hpp>
#include "pinocchio/autodiff/casadi.hpp"
#include <dlfcn.h>
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
    int NBT = 100000;    // 50000 initially, then 1000*100
    int NBT_SO = 100000; // 50000 initially, then 1000*100
    int NBT_1 = 1;       // 50000 initially, then 1000*100
    int NBT_SO_l = 1000; // 50000 initially, then 1000

#else
    int NBT = 1;
    std::cout << "(the time score in debug mode is not relevant) " << std::endl;
#endif

    int n_models = 1;
    string str_robotname[n_models];

    str_robotname[0] = "double_pendulum"; // double pendulum
    // str_robotname[1] = "ur3_robot";       // UR3
    // str_robotname[2] = "hyq";             // hyq
    // str_robotname[3] = "baxter_simple";   // baxter_simple
    // str_robotname[4] = "atlas";           // atlas
    // str_robotname[5] = "talos_full_v2";   // talos_full_v2

    char tmp[256];
    getcwd(tmp, 256);

    double time_vec[n_models];

    for (int mm = 0; mm < n_models; mm++) {

        Model model;
        bool with_ff = false; // false originally

        if ((mm == 2) || (mm == 4) || (mm == 5)) {
            with_ff = true; // True for hyQ and atlas, talos_full_v2
        }

        string str_file_ext;
        string robot_name = "";
        string str_urdf;

        robot_name = str_robotname[mm];
        std ::string filename = "../models/" + robot_name + std::string(".urdf");

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
        int code_gen;

        if (*argv[1] == 'c') {
           // replace(filewrite, "pinocchio/benchmark/AD_cases", "Data/AD_v1/avx/clang/mdof/");
            code_gen = 1;
        } else if (*argv[1] == 'g') {
           // replace(filewrite, "pinocchio/benchmark/AD_cases", "Data/AD_v1/avx/gcc/mdof/");
            code_gen = 0;
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
        PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) qdots_zero(NBT);
        PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) taus(NBT);
        PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) qddots(NBT);
        PINOCCHIO_ALIGNED_STD_VECTOR(MatrixXd) tau_mat(NBT);

        // randomizing input data here

        for (size_t i = 0; i < NBT; ++i) {
            qs[i] = randomConfiguration(model, -qmax, qmax);
            qdots[i] = Eigen::VectorXd::Random(model.nv);
            qddots[i] = Eigen::VectorXd::Random(model.nv);
        }

        int n_cases = 7;
        double time_ABA[n_cases];

        //-----------------------1 --------------------------//
        //----------------------------------------------------//
        // Compute RNEA FO derivatives faster-----------------//
        //----------------------------------------------------//
        std::cout << "Running first case!" << std::endl;

        MatrixXd drnea_dq(MatrixXd::Zero(model.nv, model.nv));
        MatrixXd drnea_dv(MatrixXd::Zero(model.nv, model.nv));
        MatrixXd drnea_da(MatrixXd::Zero(model.nv, model.nv));
        std::cout << "Running first case algo!" << std::endl;

        timer.tic();
        SMOOTH(NBT)
        {
            computeRNEADerivativesFaster(
                model, data, qs[_smooth], qdots[_smooth], qddots[_smooth], drnea_dq, drnea_dv, drnea_da);
        }
        time_ABA[0] = timer.toc() / NBT; // RNEAF timing
        std::cout << "RNEA derivativeF= \t\t\t" << time_ABA[0] << endl;

        //-----------------------2 ---------------------------------//
        //----------------------------------------------------------//
        // Compute RNEA FO derivatives using CppAD (w/out codegen)--//
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
        pinocchio::container::aligned_vector<ADConfigVectorType> ad_a(NBT);
        typedef Eigen::Matrix<ADScalar, Eigen::Dynamic, 1> VectorXAD;
        ADConfigVectorType ad_q_plus;
        VectorXAD Y(model.nv);
        CPPAD_TESTVECTOR(Scalar) x((size_t)model.nv);

        Data::MatrixXs dtau_dq_mat;
        Data::MatrixXs dtau_dv_mat;
        CPPAD_TESTVECTOR(Scalar) dtau_dq_AD;
        Eigen::Map<Data::TangentVectorType>(x.data(), model.nv, 1).setZero();
        CPPAD_TESTVECTOR(Scalar) dtau_dv_AD;

        // dtau_dq
        timer.tic();
        SMOOTH(NBT)
        {
            ad_q[_smooth] = qs[_smooth].cast<ADScalar>();
            ad_v[_smooth] = qdots[_smooth].cast<ADScalar>();
            ad_a[_smooth] = qddots[_smooth].cast<ADScalar>();

            CppAD::Independent(ad_dq);
            ad_q_plus = pinocchio::integrate(ad_model, ad_q[_smooth], ad_dq);
            pinocchio::rnea(ad_model, ad_data, ad_q_plus, ad_v[_smooth], ad_a[_smooth]);
            Eigen::Map<ADData::TangentVectorType>(Y.data(), model.nv, 1) = ad_data.tau;
            CppAD::ADFun<Scalar> ad_fun1(ad_dq, Y);
            dtau_dq_AD = ad_fun1.Jacobian(x);
            dtau_dq_mat = Eigen::Map<PINOCCHIO_EIGEN_PLAIN_ROW_MAJOR_TYPE(Data::MatrixXs)>(
                dtau_dq_AD.data(), model.nv, model.nv);

            CppAD::Independent(ad_v[_smooth]);
            pinocchio::rnea(ad_model, ad_data, ad_q[_smooth], ad_v[_smooth], ad_a[_smooth]);
            Eigen::Map<ADData::TangentVectorType>(Y.data(), model.nv, 1) = ad_data.tau;
            CppAD::ADFun<Scalar> ad_fun2(ad_v[_smooth], Y);
            Eigen::Map<Data::TangentVectorType>(x.data(), model.nv, 1) = qdots[_smooth];
            dtau_dv_AD = ad_fun2.Jacobian(x);
            dtau_dv_mat = Eigen::Map<PINOCCHIO_EIGEN_PLAIN_ROW_MAJOR_TYPE(Data::MatrixXs)>(
                dtau_dv_AD.data(), model.nv, model.nv);
        }
        time_ABA[1] = timer.toc() / NBT; //
        std::cout << "ID FO partials using CppAD No codegen= " << time_ABA[1] << endl;

        //-----------------------3 ----------------------------------//
        //-----------------------------------------------------------//
        // Compute RNEA FO derivatives using CasADi (w/out codegen)--//
        //-----------------------------------------------------------//

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

        ::casadi::SX cs_a = ::casadi::SX::sym("a", model.nv);
        TangentVectorAD a_ad(model.nv);
        a_ad = Eigen::Map<TangentVectorAD>(static_cast<std::vector<ADcScalar>>(cs_a).data(), model.nv, 1);

        rnea(adc_model, adc_data, q_int_ad, v_ad, a_ad);

        ::casadi::SX cs_tau(model.nv, 1);
        for (Eigen::DenseIndex k = 0; k < model.nv; ++k) {
            cs_tau(k) = adc_data.tau[k];
        }
        std::vector<double> v_vec((size_t)model.nv);
        std::vector<double> a_vec((size_t)model.nv);

        // check with respect to q+dq
        std::string strfun_dtdq = robot_name + std::string("_dtau_dq");
        ::casadi::SX dtau_dq = jacobian(cs_tau, cs_v_int);
        ::casadi::Function eval_dtau_dq(
            strfun_dtdq, ::casadi::SXVector {cs_q, cs_v_int, cs_v, cs_a}, ::casadi::SXVector {dtau_dq});

        // check with respect to v+dv
        std::string strfun_dtdv = robot_name + std::string("_dtau_dv");
        ::casadi::SX dtau_dv = jacobian(cs_tau, cs_v);
        ::casadi::Function eval_dtau_dv(
            strfun_dtdv, ::casadi::SXVector {cs_q, cs_v_int, cs_v, cs_a}, ::casadi::SXVector {dtau_dv});

        // check with respect to a+da
        std::string strfun_dtda = robot_name + std::string("_dtau_da");
        ::casadi::SX dtau_da = jacobian(cs_tau, cs_a);
        ::casadi::Function eval_dtau_da(
            strfun_dtda, ::casadi::SXVector {cs_q, cs_v_int, cs_v, cs_a}, ::casadi::SXVector {dtau_da});

        ::casadi::DM dtau_dq_res, dtau_dv_res, dtau_da_res;
        Eigen::MatrixXd dtau_dqAD(model.nv, model.nv), dtau_dvAD(model.nv, model.nv), dtau_daAD(model.nv, model.nv);
        std::vector<double> dtau_dq_vec, dtau_dv_vec, dtau_da_vec;
        int flag;

        if (code_gen == 1) {
            cout << "Generating FO code" << endl; // generating function for dtau-dq
            eval_dtau_dq.generate(strfun_dtdq);   // to generate the function
            // Compile the C-code to a shared library
            string compile_command
                = "gcc -fPIC -shared -O3 -march=native " + strfun_dtdq + ".c -o " + strfun_dtdq + ".so ";
           flag = system(compile_command.c_str());
            // generating function for dtau-dv
            eval_dtau_dv.generate(strfun_dtdv); // to generate the function
            // Compile the C-code to a shared library
            compile_command = "gcc -fPIC -shared -O3 -march=native " + strfun_dtdv + ".c -o " + strfun_dtdv + ".so ";
          flag = system(compile_command.c_str());
            // generating function for dtau-da
            eval_dtau_da.generate(strfun_dtda); // to generate the function
            // Compile the C-code to a shared library
            compile_command = "gcc -fPIC -shared -O3 -march=native " + strfun_dtda + ".c -o " + strfun_dtda + ".so ";
           flag = system(compile_command.c_str());
        }

        timer.tic();
        SMOOTH(NBT_1)
        {
            Eigen::Map<ConfigVector>(q_vec.data(), model.nq, 1) = qs[_smooth];
            Eigen::Map<TangentVector>(v_vec.data(), model.nv, 1) = qdots[_smooth];
            Eigen::Map<TangentVector>(a_vec.data(), model.nv, 1) = qddots[_smooth];

            dtau_dq_res = eval_dtau_dq(::casadi::DMVector {q_vec, v_int_vec, v_vec, a_vec})[0];
            dtau_dq_vec = static_cast<std::vector<double>>(dtau_dq_res);
            dtau_dqAD = Eigen::Map<Data::MatrixXs>(dtau_dq_vec.data(), model.nv, model.nv);

            dtau_dv_res = eval_dtau_dv(::casadi::DMVector {q_vec, v_int_vec, v_vec, a_vec})[0];
            dtau_dv_vec = static_cast<std::vector<double>>(dtau_dv_res);
            dtau_dvAD = Eigen::Map<Data::MatrixXs>(dtau_dv_vec.data(), model.nv, model.nv);

            dtau_da_res = eval_dtau_da(::casadi::DMVector {q_vec, v_int_vec, v_vec, a_vec})[0];
            dtau_da_vec = static_cast<std::vector<double>>(dtau_da_res);
            dtau_daAD = Eigen::Map<Data::MatrixXs>(dtau_da_vec.data(), model.nv, model.nv);
        }
        time_ABA[2] = timer.toc() / NBT_1; //
        std::cout << "ID FO partials using CasADi No Codegen= " << time_ABA[2] << endl;

        //-----------------------4 ---------------------------//
        //----------------------------------------------------//
        // Compute RNEA SO derivatives (analytical method) ---//
        //----------------------------------------------------//

        Eigen::Tensor<double, 3> dtau2_dq_ana(model.nv, model.nv, model.nv);
        Eigen::Tensor<double, 3> dtau2_dv_ana(model.nv, model.nv, model.nv);
        Eigen::Tensor<double, 3> dtau2_dqv_ana(model.nv, model.nv, model.nv);
        Eigen::Tensor<double, 3> M_FO_ana(model.nv, model.nv, model.nv);
        M_FO_ana.setZero();
        timer.tic();
        SMOOTH(NBT_SO)
        {
            computeRNEADerivativesSO(model, data, qs[_smooth], qdots[_smooth], qddots[_smooth], dtau2_dq_ana, dtau2_dv_ana,
                dtau2_dqv_ana, M_FO_ana);
        }
        time_ABA[3] = timer.toc() / NBT_SO; //
        std::cout << "ID SO partials using IDSVA SO= \t\t" << time_ABA[3] << endl;

        // //-----------------------5 ---------------------------//
        // //----------------------------------------------------//
        // // ID SO derivatives using CppAD (w/out codegen) -----//
        // //----------------------------------------------------//
        // Eigen::MatrixXd temp1(model.nv, model.nv);
        // Eigen::MatrixXd temp2(model.nv, model.nv);
        // VectorXAD Y_SO(model.nv * model.nv);

        // // d2dtau_dq2
        // CPPAD_TESTVECTOR(Scalar) d2tau_dq_ad, d2tau_dv_ad, d2tau_dqv_ad, d2tau_daq_ad;
        // typename Data::Tensor3x d2tau_dq2_AD1, d2tau_dv2_AD, d2tau_dqv2_AD, d2tau_daq2_AD;
        // Eigen::Tensor<double, 3> d2tau_dq2_AD(model.nv, model.nv, model.nv);

        // timer.tic();
        // SMOOTH(NBT_1)
        // {
        //     // d2datu_dq2
        //     CppAD::Independent(ad_dq);
        //     ad_q_plus = pinocchio::integrate(ad_model, ad_q[_smooth], ad_dq);
        //     computeRNEADerivativesFaster(ad_model, ad_data, ad_q_plus, ad_v[_smooth], ad_a[_smooth]);
        //     Eigen::Map<ADData::TangentVectorType>(Y_SO.data(), model.nv * model.nv, 1) = ad_data.dtau_dq;
        //     Eigen::Map<Data::TangentVectorType>(x.data(), model.nv, 1).setZero();
        //     CppAD::ADFun<Scalar> ad_fun1(ad_dq, Y_SO);
        //     d2tau_dq_ad = ad_fun1.Jacobian(x);
        //     d2tau_dq2_AD1 = Eigen::TensorMap<Eigen::Tensor<double, 3, Eigen::ColMajor>>(
        //         &d2tau_dq_ad[0], model.nv, model.nv, model.nv);
        //     for (unsigned int i = 0; i < model.nv; i++) {
        //         get_mat_from_tens3_v1(d2tau_dq2_AD1, temp1, model.nv, i);
        //         temp1.transposeInPlace();
        //         hess_assign_fd2(d2tau_dq2_AD, temp1, model.nv, i);
        //     }

        //     // d2datu_dv2
        //     CppAD::Independent(ad_v[_smooth]);
        //     computeRNEADerivativesFaster(ad_model, ad_data, ad_q[_smooth], ad_v[_smooth], ad_a[_smooth]);
        //     Eigen::Map<ADData::TangentVectorType>(Y_SO.data(), model.nv * model.nv, 1) = ad_data.dtau_dv;
        //     Eigen::Map<Data::TangentVectorType>(x.data(), model.nv, 1) = qdots[_smooth];
        //     CppAD::ADFun<Scalar> ad_fun2(ad_v[_smooth], Y_SO);
        //     d2tau_dv_ad = ad_fun2.Jacobian(x);
        //     d2tau_dv2_AD = Eigen::TensorMap<Eigen::Tensor<double, 3, Eigen::ColMajor>>(
        //         &d2tau_dv_ad[0], model.nv, model.nv, model.nv);

        //     for (unsigned int i = 0; i < model.nv; i++) {
        //         get_mat_from_tens3_v1(d2tau_dv2_AD, temp2, model.nv, i);
        //         temp2.transposeInPlace();
        //         hess_assign_fd_v1(d2tau_dv2_AD, temp2, model.nv, i);
        //     }
        //     // d2datu_dqdv
        //     CppAD::Independent(ad_dq);
        //     ad_q_plus = pinocchio::integrate(ad_model, ad_q[_smooth], ad_dq);
        //     computeRNEADerivativesFaster(ad_model, ad_data, ad_q_plus, ad_v[_smooth], ad_a[_smooth]);
        //     Eigen::Map<ADData::TangentVectorType>(Y_SO.data(), model.nv * model.nv, 1) = ad_data.dtau_dv;
        //     CppAD::ADFun<Scalar> ad_fun3(ad_dq, Y_SO);
        //     Eigen::Map<Data::TangentVectorType>(x.data(), model.nv, 1).setZero();
        //     d2tau_dqv_ad = ad_fun3.Jacobian(x);
        //     d2tau_dqv2_AD = Eigen::TensorMap<Eigen::Tensor<double, 3, Eigen::ColMajor>>(
        //         &d2tau_dqv_ad[0], model.nv, model.nv, model.nv);

        //     for (unsigned int i = 0; i < model.nv; i++) {
        //         get_mat_from_tens3_v1(d2tau_dqv2_AD, temp1, model.nv, i);
        //         temp1.transposeInPlace();
        //         hess_assign_fd_v1(d2tau_dqv2_AD, temp1, model.nv, i);
        //     }
        //     // d2datu_dqda
        //     CppAD::Independent(ad_a[_smooth]);
        //     ad_q_plus = pinocchio::integrate(ad_model, ad_q[_smooth], ad_dq);
        //     computeRNEADerivativesFaster(ad_model, ad_data, ad_q[_smooth], ad_v[_smooth], ad_a[_smooth]);
        //     Eigen::Map<ADData::TangentVectorType>(Y_SO.data(), model.nv * model.nv, 1) = ad_data.dtau_dq;
        //     CppAD::ADFun<Scalar> ad_fun4(ad_a[_smooth], Y_SO);
        //     Eigen::Map<Data::TangentVectorType>(x.data(), model.nv, 1) = qddots[_smooth];
        //     d2tau_daq_ad = ad_fun4.Jacobian(x);
        //     d2tau_daq2_AD = Eigen::TensorMap<Eigen::Tensor<double, 3, Eigen::ColMajor>>(
        //         &d2tau_daq_ad[0], model.nv, model.nv, model.nv);
        //     for (unsigned int i = 0; i < model.nv; i++) {
        //         get_mat_from_tens3_v1(d2tau_daq2_AD, temp1, model.nv, i);
        //         temp1.transposeInPlace();
        //         hess_assign_fd_v1(d2tau_daq2_AD, temp1, model.nv, i);
        //     }
        // }
        // time_ABA[4] = timer.toc() / NBT_1; //
        // std::cout << "ID SO partials using CppAD No codegen= \t" << time_ABA[4] << endl;

        //-----------------------6 ---------------------------//
        //----------------------------------------------------//
        // ID SO derivatives using CasADi (w/out codegen) ----//
        //----------------------------------------------------//

        pinocchio::computeRNEADerivatives(adc_model, adc_data, q_int_ad, v_ad, a_ad);
        (adc_data.M).triangularView<Eigen::StrictlyLower>()
            = (adc_data.M).transpose().triangularView<Eigen::StrictlyLower>();

        ::casadi::SX cs_dtau_dq(model.nv, model.nv), cs_dtau_dv(model.nv, model.nv), cs_dtau_da(model.nv, model.nv);

        for (Eigen::DenseIndex l = 0; l < model.nv; ++l) {
            for (Eigen::DenseIndex j = 0; j < model.nv; ++j) {
                cs_dtau_dq(l, j) = adc_data.dtau_dq(l, j);
                cs_dtau_dv(l, j) = adc_data.dtau_dv(l, j);
                cs_dtau_da(l, j) = adc_data.M(l, j);
            }
        }
        std::vector<double> d2tau_dq_vec, d2tau_dv_vec, d2tau_dqv_vec, d2tau_daq_vec;
        ::casadi::DM d2tau_dq_res, d2tau_dv_res, d2tau_dqv_res, d2tau_daq_res;
        typename Data::Tensor3x d2tau_dq2_ADc, d2tau_dv2_ADc, d2tau_dqv_ADc, d2tau_daq_ADc;

        std::string strfun_d2tdq = robot_name + std::string("_d2tau_dq");
        ::casadi::SX d2tau_dqdq = jacobian(cs_dtau_dq, cs_v_int);
        ::casadi::Function eval_d2tau_dq(
            strfun_d2tdq, ::casadi::SXVector {cs_q, cs_v_int, cs_v, cs_a}, ::casadi::SXVector {d2tau_dqdq});

        // d2tau_dv2
        std::string strfun_d2tdv = robot_name + std::string("_d2tau_dv");
        ::casadi::SX d2tau_dvdv = jacobian(cs_dtau_dv, cs_v);
        ::casadi::Function eval_d2tau_dv(
            strfun_d2tdv, ::casadi::SXVector {cs_q, cs_v_int, cs_v, cs_a}, ::casadi::SXVector {d2tau_dvdv});

        // d2tau_dqdv
        std::string strfun_d2tdqv = robot_name + std::string("_d2tau_dqv");
        ::casadi::SX d2tau_dqdv = jacobian(cs_dtau_dq, cs_v);
        ::casadi::Function eval_d2tau_dqv(
            strfun_d2tdqv, ::casadi::SXVector {cs_q, cs_v_int, cs_v, cs_a}, ::casadi::SXVector {d2tau_dqdv});

        // d2tau_dqda
        std::string strfun_d2tdaq = robot_name + std::string("_d2tau_daq");
        ::casadi::SX d2tau_dadq = jacobian(cs_dtau_da, cs_v_int);
        ::casadi::Function eval_d2tau_daq(
            strfun_d2tdaq, ::casadi::SXVector {cs_q, cs_v_int, cs_v, cs_a}, ::casadi::SXVector {d2tau_dadq});

        string compile_command;

        // Code-generation for SO partials
        if (code_gen == 1) { // generating function for dt2au-dq
            cout << "Generating SO code" << endl;
            // for q
            eval_d2tau_dq.generate(strfun_d2tdq); // to generate the function
                                                  // Compile the C-code to a shared library
            compile_command = "gcc -fPIC -shared -O3 -march=native " + strfun_d2tdq + ".c -o " + strfun_d2tdq + ".so ";
            flag = system(compile_command.c_str());
            // for v
            eval_d2tau_dv.generate(strfun_d2tdv); // to generate the function
            // Compile the C-code to a shared library
            compile_command = "gcc -fPIC -shared -O3 -march=native " + strfun_d2tdv + ".c -o " + strfun_d2tdv + ".so";
            flag = system(compile_command.c_str());
            // for q,v
            eval_d2tau_dqv.generate(strfun_d2tdqv); // to generate the function
            // Compile the C-code to a shared library
            compile_command
                = "gcc -fPIC -shared -O3 -march=native " + strfun_d2tdqv + ".c -o " + strfun_d2tdqv + ".so ";
            flag = system(compile_command.c_str());
            // for q,a
            eval_d2tau_daq.generate(strfun_d2tdaq); // to generate the function
            // Compile the C-code to a shared library
            compile_command
                = "gcc -fPIC -shared -O3 -march=native " + strfun_d2tdaq + ".c -o " + strfun_d2tdaq + ".so ";
            flag = system(compile_command.c_str());
        }

        timer.tic();
        SMOOTH(NBT_SO_l)
        {
            Eigen::Map<ConfigVector>(q_vec.data(), model.nq, 1) = qs[_smooth];
            Eigen::Map<TangentVector>(v_vec.data(), model.nv, 1) = qdots[_smooth];
            Eigen::Map<TangentVector>(a_vec.data(), model.nv, 1) = qddots[_smooth];
            // w.r.t q
            d2tau_dq_res = eval_d2tau_dq(::casadi::DMVector {q_vec, v_int_vec, v_vec, a_vec})[0];
            d2tau_dq_vec = static_cast<std::vector<double>>(d2tau_dq_res);
            d2tau_dq2_ADc = Eigen::TensorMap<Eigen::Tensor<double, 3, Eigen::ColMajor>>(
                &d2tau_dq_vec[0], model.nv, model.nv, model.nv);
            // w.r.t v
            d2tau_dv_res = eval_d2tau_dv(::casadi::DMVector {q_vec, v_int_vec, v_vec, a_vec})[0];
            d2tau_dv_vec = static_cast<std::vector<double>>(d2tau_dv_res);
            d2tau_dv2_ADc = Eigen::TensorMap<Eigen::Tensor<double, 3, Eigen::ColMajor>>(
                &d2tau_dv_vec[0], model.nv, model.nv, model.nv);
            // w.r.t q/v
            d2tau_dqv_res = eval_d2tau_dqv(::casadi::DMVector {q_vec, v_int_vec, v_vec, a_vec})[0];
            d2tau_dqv_vec = static_cast<std::vector<double>>(d2tau_dqv_res);
            d2tau_dqv_ADc = Eigen::TensorMap<Eigen::Tensor<double, 3, Eigen::ColMajor>>(
                &d2tau_dqv_vec[0], model.nv, model.nv, model.nv);
            // w.r.t q/a
            d2tau_daq_res = eval_d2tau_daq(::casadi::DMVector {q_vec, v_int_vec, v_vec, a_vec})[0];
            d2tau_daq_vec = static_cast<std::vector<double>>(d2tau_daq_res);
            d2tau_daq_ADc = Eigen::TensorMap<Eigen::Tensor<double, 3, Eigen::ColMajor>>(
                &d2tau_daq_vec[0], model.nv, model.nv, model.nv);
        }
        time_ABA[5] = timer.toc() / NBT_SO_l; //
        std::cout << "ID SO partials using CasADi No codegen = " << time_ABA[5] << endl;

        //-----------------------7 ---------------------------------------------//
        //----------------------------------------------------------------------//
        // Compute RNEA SO derivatives (analytical method - generating code) ---//
        //----------------------------------------------------------------------//
        //--------------NOT TIMED ----------------------------------------------//

       // computeRNEADerivativesSO(adc_model, adc_data, q_int_ad, v_ad, a_ad);

        std::string strfun_d2tdq_ana_cg = robot_name + std::string("_d2tau_dq_ana_cg");
        std::string strfun_d2tdv_ana_cg = robot_name + std::string("_d2tau_dv_ana_cg");
        std::string strfun_d2tdqv_ana_cg = robot_name + std::string("_d2tau_dqv_ana_cg");
        std::string strfun_MFO_ana_cg = robot_name + std::string("_MFO_ana_cg");

        ::casadi::SX cs_d2tau_dq_ana_cg(model.nv, model.nv * model.nv);
        ::casadi::SX cs_d2tau_dv_ana_cg(model.nv, model.nv * model.nv);
        ::casadi::SX cs_d2tau_dqv_ana_cg(model.nv, model.nv * model.nv);
        ::casadi::SX cs_MFO_ana_cg(model.nv, model.nv * model.nv);
        ::casadi::SX cas_mat1(model.nv, model.nv), cas_mat2(model.nv, model.nv), cas_mat3(model.nv, model.nv),
            cas_mat4(model.nv, model.nv);

        for (int k = 0; k < model.nv; k++) {
            get_mat_from_tens3_v2(adc_data.d2tau_dqdq, cas_mat1, model.nv, k);
            get_mat_from_tens3_v2(adc_data.d2tau_dvdv, cas_mat2, model.nv, k);
            get_mat_from_tens3_v2(adc_data.d2tau_dqdv, cas_mat3, model.nv, k);
            get_mat_from_tens3_v2(adc_data.d2tau_dadq, cas_mat4, model.nv, k);

            for (int j = 0; j < model.nv; j++) {
                for (int i = 0; i < model.nv; i++) {
                    cs_d2tau_dq_ana_cg(i, j + k * model.nv) = cas_mat1(i, j);
                    cs_d2tau_dv_ana_cg(i, j + k * model.nv) = cas_mat2(i, j);
                    cs_d2tau_dqv_ana_cg(i, j + k * model.nv) = cas_mat3(i, j);
                    cs_MFO_ana_cg(i, j + k * model.nv) = cas_mat4(i, j);
                }
            }
        }

        ::casadi::Function eval_d2tau_dq_ana_cg(strfun_d2tdq_ana_cg, ::casadi::SXVector {cs_q, cs_v_int, cs_v, cs_a},
            ::casadi::SXVector {cs_d2tau_dq_ana_cg});
        ::casadi::Function eval_d2tau_dv_ana_cg(strfun_d2tdv_ana_cg, ::casadi::SXVector {cs_q, cs_v_int, cs_v, cs_a},
            ::casadi::SXVector {cs_d2tau_dv_ana_cg});
        ::casadi::Function eval_d2tau_dqv_ana_cg(strfun_d2tdqv_ana_cg, ::casadi::SXVector {cs_q, cs_v_int, cs_v, cs_a},
            ::casadi::SXVector {cs_d2tau_dqv_ana_cg});
        ::casadi::Function eval_MFO_ana_cg(
            strfun_MFO_ana_cg, ::casadi::SXVector {cs_q, cs_v_int, cs_v, cs_a}, ::casadi::SXVector {cs_MFO_ana_cg});

        if (code_gen == 1) {
            cout << "Generating Ana ID SO code" << endl;        // generating function for dtau-dq
            eval_d2tau_dq_ana_cg.generate(strfun_d2tdq_ana_cg); // to generate the function
                                                                // Compile the C-code to a shared library
            string compile_command = "gcc -fPIC -shared -O3 -march=native " + strfun_d2tdq_ana_cg + ".c -o "
                                     + strfun_d2tdq_ana_cg + ".so ";
            flag = system(compile_command.c_str());
            // generating function for d2tau-dv
            eval_d2tau_dv_ana_cg.generate(strfun_d2tdv_ana_cg); // to generate the function
            // Compile the C-code to a shared library
            compile_command = "gcc -fPIC -shared -O3 -march=native " + strfun_d2tdv_ana_cg + ".c -o "
                              + strfun_d2tdv_ana_cg + ".so ";
            flag = system(compile_command.c_str());
            // generating function for d2tau-dqv
            eval_d2tau_dqv_ana_cg.generate(strfun_d2tdqv_ana_cg); // to generate the function
                                                                  // Compile the C-code to a shared library
            compile_command = "gcc -fPIC -shared -O3 -march=native " + strfun_d2tdqv_ana_cg + ".c -o "
                              + strfun_d2tdqv_ana_cg + ".so ";
            flag = system(compile_command.c_str());
            // generating function for M FO
            eval_MFO_ana_cg.generate(strfun_MFO_ana_cg); // to generate the function
            // Compile the C-code to a shared library
            compile_command
                = "gcc -fPIC -shared -O3 -march=native " + strfun_MFO_ana_cg + ".c -o " + strfun_MFO_ana_cg + ".so ";
            flag = system(compile_command.c_str());
        }
        
     
     
     
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