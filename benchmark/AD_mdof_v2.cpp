/* Date created on- 8/23/22
* Modified on 1/3/23
Modified by- Shubham Singh, singh281@utexas.edu
version - ADmdof_v2

This version compares the CPU Runtime for
1. RNEA FO analytical derivatives- Faster       -- Done (TIMED)
2. RNEA FO partials using CasADi w codegen      -- Done (TIMED)
3. RNEA SO partials using CasADi w codegen      -- Done (TIMED)
4. RNEA SO partials using analytical w codegen  -- Done (TIMED)

*/

#include "pinocchio/codegen/cppadcg.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/rnea-derivatives.hpp"
#include "pinocchio/algorithm/rnea-derivatives-faster.hpp"
#include "pinocchio/algorithm/cholesky.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/parsers/sample-models.hpp"
#include "pinocchio/container/aligned-vector.hpp"
#include "pinocchio/codegen/code-generator-algo.hpp"
#include "pinocchio/algorithm/aza_so_v2.hpp"
#include "pinocchio/algorithm/ID_FO_AZA.hpp"
#include "pinocchio/algorithm/M_FO_v1.hpp"
#include <fstream>
#include "pinocchio/utils/timer.hpp"
#include <string>
#include <iostream>
#include <ctime>
#include "pinocchio/utils/tensor_utils.hpp"
#include <casadi/casadi.hpp>
#include "pinocchio/autodiff/casadi.hpp"
#include "pinocchio/utils/tensor_utils.hpp"

using namespace std;
using namespace pinocchio;
using namespace pinocchio::casadi;

bool replace(std::string& str, const std::string& from, const std::string& to);

int main(int argc, const char* argv[])
{
    using CppAD::AD;
    using CppAD::NearEqual;

    using namespace Eigen;

    PinocchioTicToc timer(PinocchioTicToc::US);

#ifdef NDEBUG
    // int NBT= 1; // 50000 initially
    int NBT = 100000;    // 50000 initially, then 1000*100
    int NBT_SO = 100000; // 50000 initially, then 1000*100

#else
    int NBT = 1;
    std::cout << "(the time score in debug mode is not relevant) " << std::endl;
#endif

    int n_models = 4; // no of robots to be used
    string str_robotname[n_models];

    str_robotname[0] = "double_pendulum"; // double pendulum
    str_robotname[1] = "ur3_robot";       // UR3
    str_robotname[2] = "hyq";             // hyq
    str_robotname[3] = "baxter_simple";   // baxter_simple
    str_robotname[4] = "atlas";           // atlas
    // str_robotname[5] = "talos_full_v2";   // Talos

    char tmp[256];
    getcwd(tmp, 256);

    double time_vec[n_models];

    for (int mm = 0; mm < n_models; mm++) {

        Model model;

        string str_file_ext;
        string robot_name = "";
        string str_urdf;

        robot_name = str_robotname[mm];
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
        std::cout << "nq = " << model.nq << std::endl;
        std::cout << "nv = " << model.nv << std::endl;
        cout << "Model is" << robot_name << endl;

        //-- opening filename here
        ofstream file1;
        string filewrite = tmp;

        if (*argv[1] == 'c') {
           // replace(filewrite, "pinocchio/benchmark/AD_cases", "Data/AD_v2/avx/clang/mdof/");
        } else if (*argv[1] == 'g') {
          //  replace(filewrite, "pinocchio/benchmark/AD_cases", "Data/AD_v2/avx/gcc/mdof/");
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

        int n_cases = 4;
        double time_ABA[n_cases];

        //-----------------------1 ---------------------------//
        //----------------------------------------------------//
        // Compute RNEA FO derivatives faster-----------------//
        //----------------------------------------------------//

        MatrixXd drnea_dq(MatrixXd::Zero(model.nv, model.nv));
        MatrixXd drnea_dv(MatrixXd::Zero(model.nv, model.nv));
        MatrixXd drnea_da(MatrixXd::Zero(model.nv, model.nv));

        timer.tic();
        SMOOTH(NBT)
        {
            computeRNEADerivativesFaster(
                model, data, qs[_smooth], qdots[_smooth], qddots[_smooth], drnea_dq, drnea_dv, drnea_da);
        }
        time_ABA[0] = timer.toc() / NBT; // RNEAF timing
        std::cout << "RNEA derivativeF= \t\t\t\t" << time_ABA[0] << endl;

        //-----------------------2 -------------------------------//
        //--------------------------------------------------------//
        // Compute RNEA FO derivatives using CasADI (w codegen)---//
        //--------------------------------------------------------//
        // ---- CodeGenRNEA --------------------------------------//

        typedef double Scalar;
        typedef ::casadi::SX ADScalar;
        typedef pinocchio::ModelTpl<ADScalar> ADModel;
        typedef ADModel::Data ADData;

        typedef ADModel::ConfigVectorType ConfigVectorAD;
        typedef ADModel::TangentVectorType TangentVectorAD;
        ADModel ad_model = model.cast<ADScalar>();
        ADData ad_data(ad_model);

        ::casadi::SX cs_q = ::casadi::SX::sym("q", model.nq);
        ::casadi::SX cs_v_int = ::casadi::SX::sym("v_inc", model.nv);
        ConfigVectorAD q_ad(model.nq), v_int_ad(model.nv), q_int_ad(model.nq);
        q_ad = Eigen::Map<ConfigVectorAD>(static_cast<std::vector<ADScalar>>(cs_q).data(), model.nq, 1);
        v_int_ad = Eigen::Map<ConfigVectorAD>(static_cast<std::vector<ADScalar>>(cs_v_int).data(), model.nv, 1);
        pinocchio::integrate(ad_model, q_ad, v_int_ad, q_int_ad);

        std::vector<double> q_vec((size_t)model.nq);
        std::vector<double> v_int_vec((size_t)model.nv);
        Eigen::Map<TangentVector>(v_int_vec.data(), model.nv, 1).setZero();
        std::vector<double> v_vec((size_t)model.nv);
        std::vector<double> a_vec((size_t)model.nv);

        // check with respect to q+dq
        std::string strfun_dtdq = robot_name + std::string("_dtau_dq");
        std::string strfun_dtdv = robot_name + std::string("_dtau_dv");
        std::string strfun_dtda = robot_name + std::string("_dtau_da");

        ::casadi::Function eval_dtau_dq = ::casadi::external(strfun_dtdq);
        // check with respect to v+dv
        ::casadi::Function eval_dtau_dv = ::casadi::external(strfun_dtdv);
        // check with respect to a+da
        ::casadi::Function eval_dtau_da = ::casadi::external(strfun_dtda);

        ::casadi::DM dtau_dq_res, dtau_dv_res, dtau_da_res;
        Eigen::MatrixXd dtau_dqAD(model.nv, model.nv), dtau_dvAD(model.nv, model.nv), dtau_daAD(model.nv, model.nv);
        std::vector<double> dtau_dq_vec, dtau_dv_vec, dtau_da_vec;

        timer.tic();
        SMOOTH(NBT)
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
        time_ABA[1] = timer.toc() / NBT;
        std::cout << "RNEA FO derivatives CasADi codegen= \t\t" << time_ABA[1] << endl;

        //-----------------------3 ------------------------------//
        //-------------------------------------------------------//
        // Compute RNEA SO derivatives using CasADI (w codegen)---//
        //-------------------------------------------------------//
        // ---- CodeGenRNEA -------------------------------------//

        std::string strfun_d2tdq = robot_name + std::string("_d2tau_dq");
        std::string strfun_d2tdv = robot_name + std::string("_d2tau_dv");
        std::string strfun_d2tdqv = robot_name + std::string("_d2tau_dqv");
        std::string strfun_d2tdaq = robot_name + std::string("_d2tau_daq");

        ::casadi::Function eval_d2tau_dq = ::casadi::external(strfun_d2tdq);
        ::casadi::Function eval_d2tau_dv = ::casadi::external(strfun_d2tdv);
        ::casadi::Function eval_d2tau_dqv = ::casadi::external(strfun_d2tdqv);
        ::casadi::Function eval_d2tau_daq = ::casadi::external(strfun_d2tdaq);

        ::casadi::DM d2tau_dq_res, d2tau_dv_res, d2tau_dqv_res, d2tau_daq_res;
        typename Data::Tensor3x d2tau_dq2_AD, d2tau_dv2_AD, d2tau_dqv_AD, d2tau_daq_AD;
        std::vector<double> d2tau_dq_vec, d2tau_dv_vec, d2tau_dqv_vec, d2tau_daq_vec;

        timer.tic();
        SMOOTH(NBT_SO)
        {
            Eigen::Map<ConfigVector>(q_vec.data(), model.nq, 1) = qs[_smooth];
            Eigen::Map<TangentVector>(v_vec.data(), model.nv, 1) = qdots[_smooth];
            Eigen::Map<TangentVector>(a_vec.data(), model.nv, 1) = qddots[_smooth];

            // w.r.t q
            d2tau_dq_res = eval_d2tau_dq(::casadi::DMVector {q_vec, v_int_vec, v_vec, a_vec})[0];
            d2tau_dq_vec = static_cast<std::vector<double>>(d2tau_dq_res);
            d2tau_dq2_AD = Eigen::TensorMap<Eigen::Tensor<double, 3, Eigen::ColMajor>>(
                &d2tau_dq_vec[0], model.nv, model.nv, model.nv);

            // w.r.t v
            d2tau_dv_res = eval_d2tau_dv(::casadi::DMVector {q_vec, v_int_vec, v_vec, a_vec})[0];
            d2tau_dv_vec = static_cast<std::vector<double>>(d2tau_dv_res);
            d2tau_dv2_AD = Eigen::TensorMap<Eigen::Tensor<double, 3, Eigen::ColMajor>>(
                &d2tau_dv_vec[0], model.nv, model.nv, model.nv);

            // w.r.t q/v
            d2tau_dqv_res = eval_d2tau_dqv(::casadi::DMVector {q_vec, v_int_vec, v_vec, a_vec})[0];
            d2tau_dqv_vec = static_cast<std::vector<double>>(d2tau_dqv_res);
            d2tau_dqv_AD = Eigen::TensorMap<Eigen::Tensor<double, 3, Eigen::ColMajor>>(
                &d2tau_dqv_vec[0], model.nv, model.nv, model.nv);

            // w.r.t q/a
            d2tau_daq_res = eval_d2tau_daq(::casadi::DMVector {q_vec, v_int_vec, v_vec, a_vec})[0];
            d2tau_daq_vec = static_cast<std::vector<double>>(d2tau_daq_res);
            d2tau_daq_AD = Eigen::TensorMap<Eigen::Tensor<double, 3, Eigen::ColMajor>>(
                &d2tau_daq_vec[0], model.nv, model.nv, model.nv);
        }
        time_ABA[2] = timer.toc() / NBT_SO;
        std::cout << "RNEA SO derivatives CasADi codegen= \t\t" << time_ABA[2] << endl;

        //-----------------------4 -----------------------------------//
        //------------------------------------------------------------//
        // Compute RNEA SO derivatives using Analytical (w codegen)---//
        //------------------------------------------------------------//

        std::string strfun_d2tdq_ana_cg = robot_name + std::string("_d2tau_dq_ana_cg");
        std::string strfun_d2tdv_ana_cg = robot_name + std::string("_d2tau_dv_ana_cg");
        std::string strfun_d2tdqv_ana_cg = robot_name + std::string("_d2tau_dqv_ana_cg");
        std::string strfun_MFO_ana_cg = robot_name + std::string("_MFO_ana_cg");

        ::casadi::Function eval_d2tau_dq_ana_cg = ::casadi::external(strfun_d2tdq_ana_cg);
        ::casadi::Function eval_d2tau_dv_ana_cg = ::casadi::external(strfun_d2tdv_ana_cg);
        ::casadi::Function eval_d2tau_dqv_ana_cg = ::casadi::external(strfun_d2tdqv_ana_cg);
        ::casadi::Function eval_MFO_ana_cg = ::casadi::external(strfun_MFO_ana_cg);
        ::casadi::DM d2tau_dq_res_ana_cg, d2tau_dv_res_ana_cg, d2tau_dqv_res_ana_cg, MFO_res_ana_cg;

        typename Data::Tensor3x d2tau_dq_cg, d2tau_dv_cg, d2tau_dqv_cg, MFO_cg;
        std::vector<double> d2tau_dq_vec_ana_cg, d2tau_dv_vec_ana_cg, d2tau_dqv_vec_ana_cg, MFO_vec_ana_cg;

        timer.tic();
        SMOOTH(NBT_SO)
        {
            Eigen::Map<ConfigVector>(q_vec.data(), model.nq, 1) = qs[_smooth];
            Eigen::Map<TangentVector>(v_vec.data(), model.nv, 1) = qdots[_smooth];
            Eigen::Map<TangentVector>(a_vec.data(), model.nv, 1) = qddots[_smooth];

            // w.r.t q
            d2tau_dq_res_ana_cg = eval_d2tau_dq_ana_cg(::casadi::DMVector {q_vec, v_int_vec, v_vec, a_vec})[0];
            d2tau_dq_vec_ana_cg = static_cast<std::vector<double>>(d2tau_dq_res_ana_cg);
            d2tau_dq_cg = Eigen::TensorMap<Eigen::Tensor<double, 3, Eigen::ColMajor>>(
                &d2tau_dq_vec_ana_cg[0], model.nv, model.nv, model.nv);

            // w.r.t v
            d2tau_dv_res_ana_cg = eval_d2tau_dv_ana_cg(::casadi::DMVector {q_vec, v_int_vec, v_vec, a_vec})[0];
            d2tau_dv_vec_ana_cg = static_cast<std::vector<double>>(d2tau_dv_res_ana_cg);
            d2tau_dv_cg = Eigen::TensorMap<Eigen::Tensor<double, 3, Eigen::ColMajor>>(
                &d2tau_dv_vec_ana_cg[0], model.nv, model.nv, model.nv);

            // w.r.t q/v
            d2tau_dqv_res_ana_cg = eval_d2tau_dqv_ana_cg(::casadi::DMVector {q_vec, v_int_vec, v_vec, a_vec})[0];
            d2tau_dqv_vec_ana_cg = static_cast<std::vector<double>>(d2tau_dqv_res_ana_cg);
            d2tau_dqv_cg = Eigen::TensorMap<Eigen::Tensor<double, 3, Eigen::ColMajor>>(
                &d2tau_dqv_vec_ana_cg[0], model.nv, model.nv, model.nv);

            // w.r.t q/a
            MFO_res_ana_cg = eval_MFO_ana_cg(::casadi::DMVector {q_vec, v_int_vec, v_vec, a_vec})[0];
            MFO_vec_ana_cg = static_cast<std::vector<double>>(MFO_res_ana_cg);
            MFO_cg = Eigen::TensorMap<Eigen::Tensor<double, 3, Eigen::ColMajor>>(
                &MFO_vec_ana_cg[0], model.nv, model.nv, model.nv);
        }
        time_ABA[3] = timer.toc() / NBT_SO;
        std::cout << "RNEA SO derivatives Analytical codegen= \t" << time_ABA[3] << endl;

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