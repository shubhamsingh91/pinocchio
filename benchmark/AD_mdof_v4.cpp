/* Date created on- 8/29/22
* Modified on 1/3/22
Modified by- Shubham Singh, singh281@utexas.edu
version - ADmdof_v4

This version compares the CPU Runtime for
1. ABA FO analytical derivatives- Faster -- Done (Timed)
2. ABA FO partials using CasADi w codegen-- Done (Timed)
3. ABA SO analytical derivatives         -- Done (Timed)
4. ABA SO partials using CasADi w codegen-- Done (Timed)
5. ABA SO partials using Analytical + codegen-- Done (Timed)

* Modifications
*/

#include "pinocchio/codegen/cppadcg.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/rnea-derivatives.hpp"
#include "pinocchio/algorithm/rnea-derivatives-faster.hpp"
#include "pinocchio/algorithm/cholesky.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/parsers/sample-models.hpp"
#include "pinocchio/container/aligned-vector.hpp"
#include "pinocchio/algorithm/rnea-derivatives-SO.hpp"
#include "pinocchio/algorithm/rnea-second-order-derivatives.hpp"
#include "pinocchio/codegen/code-generator-algo.hpp"
// #include "pinocchio/algorithm/aza_so_v2.hpp"
#include "pinocchio/algorithm/ID_FO_AZA.hpp"
// #include "pinocchio/algorithm/M_FO_v1.hpp"
#include <fstream>
#include "pinocchio/utils/timer.hpp"
#include <string>
#include <iostream>
#include <ctime>
#include "pinocchio/utils/tensor_utils.hpp"
#include <casadi/casadi.hpp>
#include "pinocchio/autodiff/casadi.hpp"
#include "pinocchio/utils/tensor_utils.hpp"
// #include "pinocchio/algorithm/aba_v2.hpp"
#include "pinocchio/algorithm/aba-derivatives-faster.hpp"
#include "pinocchio/algorithm/aba-derivatives.hpp"

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
    int NBT = 10000;    // 50000 initially, then 1000*100
    int NBT_SO = 10000; // 50000 initially, then 1000*100

#else
    int NBT = 1;
    std::cout << "(the time score in debug mode is not relevant) " << std::endl;
#endif

    int n_models = 1; // no of robots to be used
    string str_robotname[n_models];

    str_robotname[0] = "double_pendulum"; // double pendulum
    // str_robotname[1] = "ur3_robot";       // UR3
    // str_robotname[2] = "hyq";             // hyq
    // str_robotname[3] = "baxter_simple";   // baxter_simple
    // str_robotname[4] = "atlas";           // atlas
    // str_robotname[5] = "talos_full_v2";   // talos

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
       //     replace(filewrite, "pinocchio/benchmark/AD_cases", "Data/AD_v4/avx/clang/mdof/");
        } else if (*argv[1] == 'g') {
      //      replace(filewrite, "pinocchio/benchmark/AD_cases", "Data/AD_v4/avx/gcc/mdof/");
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
        PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) v_zero(NBT);
        PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) taus(NBT);
        PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) qddots(NBT);
        PINOCCHIO_ALIGNED_STD_VECTOR(MatrixXd) tau_mat(NBT);

        // randomizing input data here

        for (size_t i = 0; i < NBT; ++i) {
            qs[i] = randomConfiguration(model, -qmax, qmax);
            qdots[i] = Eigen::VectorXd::Random(model.nv);
            qddots[i] = Eigen::VectorXd::Random(model.nv);
            taus[i] = Eigen::VectorXd::Random(model.nv);
            v_zero[i] = Eigen::VectorXd::Zero(model.nv);
        }

        int n_cases = 5;
        double time_ABA[n_cases];

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
        std::cout << "ABA FO derivative analytical= \t\t\t" << time_ABA[0] << endl;

        //-----------------------2 ------------------------------//
        //-------------------------------------------------------//
        // Compute ABA FO derivatives using CasADI (w codegen)---//
        //-------------------------------------------------------//
        //-------------------------------------------------------//

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
        std::vector<double> tau_vec((size_t)model.nv);
        std::vector<double> a_vec((size_t)model.nv);

        std::string strfun_dqdddq = robot_name + std::string("_dqdd_dq");
        std::string strfun_dqdddv = robot_name + std::string("_dqdd_dv");
        std::string strfun_dqdddtau = robot_name + std::string("_dqdd_dtau");

        // check with respect to q+dq
        ::casadi::Function eval_dqdd_dq = ::casadi::external(strfun_dqdddq);
        // check with respect to v+dv
        ::casadi::Function eval_dqdd_dv = ::casadi::external(strfun_dqdddv);
        // check with respect to tau+dtau
        ::casadi::Function eval_dqdd_dtau = ::casadi::external(strfun_dqdddtau);

        ::casadi::DM dqdd_dq_res, dqdd_dv_res, dqdd_dtau_res;
        Eigen::MatrixXd dqdd_dqAD(model.nv, model.nv), dqdd_dvAD(model.nv, model.nv), dqdd_dtauAD(model.nv, model.nv);
        std::vector<double> dqdd_dq_vec(model.nv), dqdd_dv_vec(model.nv), dqdd_dtau_vec(model.nv);

        timer.tic();
        SMOOTH(NBT)
        {
            Eigen::Map<ConfigVector>(q_vec.data(), model.nq, 1) = qs[_smooth];
            Eigen::Map<TangentVector>(v_vec.data(), model.nv, 1) = qdots[_smooth];
            Eigen::Map<TangentVector>(tau_vec.data(), model.nv, 1) = taus[_smooth];

            dqdd_dq_res = eval_dqdd_dq(::casadi::DMVector {q_vec, v_int_vec, v_vec, tau_vec})[0];
            dqdd_dq_vec = static_cast<std::vector<double>>(dqdd_dq_res);
            dqdd_dqAD = Eigen::Map<Data::MatrixXs>(dqdd_dq_vec.data(), model.nv, model.nv);

            dqdd_dv_res = eval_dqdd_dv(::casadi::DMVector {q_vec, v_int_vec, v_vec, tau_vec})[0];
            dqdd_dv_vec = static_cast<std::vector<double>>(dqdd_dv_res);
            dqdd_dvAD = Eigen::Map<Data::MatrixXs>(dqdd_dv_vec.data(), model.nv, model.nv);

            dqdd_dtau_res = eval_dqdd_dtau(::casadi::DMVector {q_vec, v_int_vec, v_vec, tau_vec})[0];
            dqdd_dtau_vec = static_cast<std::vector<double>>(dqdd_dtau_res);
            dqdd_dtauAD = Eigen::Map<Data::MatrixXs>(dqdd_dtau_vec.data(), model.nv, model.nv);
        }
        time_ABA[1] = timer.toc() / NBT;
        std::cout << "ABA FO derivatives CasADi + codegen= \t\t" << time_ABA[1] << endl;

        //-----------------------3 ---------------------------//
        //----------------------------------------------------//
        // Compute ABA SO derivatives (FDSVA SO)--------------//
        //----------------------------------------------------//
        Eigen::Tensor<double, 3> dtau2_dq_ana(model.nv, model.nv, model.nv);
        Eigen::Tensor<double, 3> dtau2_dv_ana(model.nv, model.nv, model.nv);
        Eigen::Tensor<double, 3> dtau2_dqv_ana(model.nv, model.nv, model.nv);
        Eigen::Tensor<double, 3> M_FO(model.nv, model.nv, model.nv);
        dtau2_dq_ana.setZero();
        dtau2_dv_ana.setZero();
        dtau2_dqv_ana.setZero();
        M_FO.setZero();
        Eigen::Tensor<double, 3> daba2_dq_ana(model.nv, model.nv, model.nv);
        Eigen::Tensor<double, 3> daba2_dv_ana(model.nv, model.nv, model.nv);
        Eigen::Tensor<double, 3> daba2_qv_ana(model.nv, model.nv, model.nv);
        Eigen::Tensor<double, 3> daba2_tauq_ana(model.nv, model.nv, model.nv);

        Eigen::Tensor<double, 3> prodq(model.nv, model.nv, model.nv);
        Eigen::Tensor<double, 3> prodv(model.nv, model.nv, model.nv);
        Eigen::Tensor<double, 3> prodqdd(model.nv, model.nv, model.nv);

        Eigen::MatrixXd mat_out(MatrixXd::Zero(model.nv, model.nv));
        Eigen::MatrixXd Minv_mat_prod_v6_temp(MatrixXd::Zero(model.nv, 4 * model.nv * model.nv));
        // Some temp variables here

        Eigen::VectorXd vec1(model.nv);
        Eigen::VectorXd vec2(model.nv);

        Eigen::MatrixXd mat1(model.nv, model.nv);
        Eigen::MatrixXd mat2(model.nv, model.nv);
        Eigen::MatrixXd mat3(model.nv, model.nv);
        Eigen::MatrixXd mat4(model.nv, model.nv);

        MatrixXd Minv(MatrixXd::Zero(model.nv, model.nv));
        MatrixXd Minv_neg(MatrixXd::Zero(model.nv, model.nv));
        Eigen::MatrixXd term_in(model.nv, 4 * model.nv * model.nv);  // For DMM
        Eigen::MatrixXd term_out(model.nv, 4 * model.nv * model.nv); // For DMM
        Eigen::MatrixXd mat_in_id_fo_aza(model.nv, 3 * model.nv);

        timer.tic();
        SMOOTH(NBT_SO)
        {
            pinocchio::computeABADerivatives(
                model, data, qs[_smooth], qdots[_smooth], taus[_smooth], daba_dq, daba_dv, daba_dtau);

            ComputeRNEASecondOrderDerivatives(model, data, qs[_smooth], qdots[_smooth], qddots[_smooth], dtau2_dq_ana, dtau2_dv_ana,
                dtau2_dqv_ana, M_FO);

            Minv = daba_dtau;
            Minv_neg = -Minv;
            //--- For models N<30------------------//
            // Inner term Compute using DTM (or DMM)
            if (model.nv <= 30) {
                for (int u = 0; u < model.nv; u++) {
                    get_mat_from_tens3_v1(M_FO, mat1, model.nv, u);
                    for (int w = 0; w < model.nv; w++) {
                        get_mat_from_tens3_v1(M_FO, mat2, model.nv, w);
                        vec1 = mat1 * daba_dq.col(w);
                        vec2 = mat2 * daba_dv.col(u);
                        hess_assign(prodq, vec1, 0, u, w, 1, model.nv); // slicing a vector in
                        hess_assign(prodv, vec2, 0, w, u, 1, model.nv); // slicing a vector in
                    }
                    mat3.noalias() = mat1 * Minv;
                    hess_assign_fd2(prodqdd, mat3, model.nv, u);
                }
            }
            //--- For models N>30------------------//// Used for ATLAS and talos_full_v2
            // Inner term Compute using IDFOZA
            else {
                mat_in_id_fo_aza << daba_dq, daba_dv, Minv; // concatenating FO partial of FD wrt q and qdot
                for (int ii = 0; ii < 3 * model.nv; ii++) {
                    computeID_FO_AZA(model, data, qs[_smooth], qdots[_smooth], mat_in_id_fo_aza.col(ii), mat_out);
                    if (ii < model.nv) {
                        hess_assign_fd_v1(prodq, mat_out, model.nv, ii);
                    } else if (ii >= model.nv && ii < 2 * model.nv) {
                        hess_assign_fd_v1(prodv, mat_out, model.nv, ii - model.nv);
                    } else {
                        hess_assign_fd_v1(prodqdd, mat_out, model.nv, ii - 2 * model.nv);
                    }
                }
            }

            // Inner term addition using single loop- overall cheaper than double loop inner-term add
            for (int u = 0; u < model.nv; u++) {
                get_mat_from_tens3_v1(dtau2_dq_ana, mat1, model.nv, u); // changed
                get_mat_from_tens2(prodq, mat2, model.nv, u);
                get_mat_from_tens3_v1(prodq, mat3, model.nv, u);
                mat1 += mat2 + mat3;
                term_in.middleCols(4 * u * model.nv, model.nv) = mat1;
                // partial w.r.t v
                get_mat_from_tens3_v1(dtau2_dv_ana, mat2, model.nv, u); // changed
                term_in.middleCols((4 * u + 1) * model.nv, model.nv) = mat2;
                // partial w.r.t q/v
                get_mat_from_tens3_v1(dtau2_dqv_ana, mat3, model.nv, u); // changed
                get_mat_from_tens3_v1(prodv, mat2, model.nv, u);
                mat3 += mat2;
                term_in.middleCols((4 * u + 2) * model.nv, model.nv) = mat3;
                // partial w.r.t tau/q
                get_mat_from_tens2(prodqdd, mat1, model.nv, u); // changed
                term_in.middleCols((4 * u + 3) * model.nv, model.nv) = mat1;
            }
            // outer term compute using DTM
            term_out = Minv_neg * term_in; // DMM here with -Minv

            // final assign using double loop- overall cheaper than single loop assign
            for (int u = 0; u < model.nv; u++) {
                for (int w = 0; w < model.nv; w++) {
                    hess_assign(daba2_dq_ana, term_out.col(4 * u * model.nv + w), 0, w, u, 1, model.nv);
                    hess_assign(daba2_dv_ana, term_out.col((4 * u + 1) * model.nv + w), 0, w, u, 1, model.nv);
                    hess_assign(daba2_qv_ana, term_out.col((4 * u + 2) * model.nv + w), 0, w, u, 1, model.nv);
                    hess_assign(daba2_tauq_ana, term_out.col((4 * u + 3) * model.nv + w), 0, w, u, 1, model.nv);
                }
            }
        }
        time_ABA[2] = timer.toc() / NBT_SO;
        std::cout << "ABA SO partials using Analytical = \t\t" << time_ABA[2] << endl;

        //-----------------------4 ------------------------------//
        //-------------------------------------------------------//
        // Compute ABA SO derivatives using CasADI (w codegen)---//
        //-------------------------------------------------------//

        std::vector<double> v2(model.nv * model.nv), v3(model.nv * model.nv);
        std::vector<double> v4(model.nv * model.nv), v5(model.nv * model.nv);

        std::string strfun_d2qdddq = robot_name + std::string("_d2qdd_dq");
        std::string strfun_d2qdddv = robot_name + std::string("_d2qdd_dv");
        std::string strfun_d2qdddqv = robot_name + std::string("_d2qdd_dqv");
        std::string strfun_d2qdddtauq = robot_name + std::string("_d2qdd_dtauq");

        ::casadi::DM d2qdd_dq_res, d2qdd_dv_res, d2qdd_dqv_res, d2qdd_dtauq_res;
        std::vector<double> d2qdd_dq_vec, d2qdd_dv_vec, d2qdd_dqv_vec, d2qdd_dtauq_vec;
        Eigen::Tensor<double, 3> d2qdd_dq2_AD(model.nv, model.nv, model.nv);
        Eigen::Tensor<double, 3> d2qdd_dv2_AD(model.nv, model.nv, model.nv);
        Eigen::Tensor<double, 3> d2qdd_dqv_AD(model.nv, model.nv, model.nv);
        Eigen::Tensor<double, 3> d2qdd_dtauq_AD(model.nv, model.nv, model.nv);
        int n2 = model.nv * model.nv;

        if (mm == 5) { // Not running this for Talos_full_v2
            time_ABA[3] = 0.0;
        } else {

            ::casadi::Function eval_d2qdd_dq = ::casadi::external(strfun_d2qdddq);
            ::casadi::Function eval_d2qdd_dv = ::casadi::external(strfun_d2qdddv);
            ::casadi::Function eval_d2qdd_dqv = ::casadi::external(strfun_d2qdddqv);
            ::casadi::Function eval_d2qdd_dtauq = ::casadi::external(strfun_d2qdddtauq);

            timer.tic();
            SMOOTH(NBT_SO)
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
                        v2[i] = d2qdd_dq_vec[j * n2 + i];
                        v3[i] = d2qdd_dv_vec[j * n2 + i];
                        v4[i] = d2qdd_dqv_vec[j * n2 + i];
                        v5[i] = d2qdd_dtauq_vec[j * n2 + i];
                    }

                    mat1 = Eigen::Map<Eigen::Matrix<double, Dynamic, Dynamic>>(v2.data(), model.nv, model.nv);
                    hess_assign_fd_v1(d2qdd_dq2_AD, mat1, model.nv, j);
                    mat2 = Eigen::Map<Eigen::Matrix<double, Dynamic, Dynamic>>(v3.data(), model.nv, model.nv);
                    hess_assign_fd_v1(d2qdd_dv2_AD, mat2, model.nv, j);
                    mat3 = Eigen::Map<Eigen::Matrix<double, Dynamic, Dynamic>>(v4.data(), model.nv, model.nv);
                    hess_assign_fd_v1(d2qdd_dqv_AD, mat3, model.nv, j);
                    mat4 = Eigen::Map<Eigen::Matrix<double, Dynamic, Dynamic>>(v5.data(), model.nv, model.nv);
                    hess_assign_fd_v1(d2qdd_dtauq_AD, mat4, model.nv, j);
                }
            }
            time_ABA[3] = timer.toc() / NBT_SO;
            std::cout << "ABA SO derivatives CasADi + codegen= \t\t" << time_ABA[3] << endl;
        }

        // //-----------------------5 -----------------------------------//
        // //------------------------------------------------------------//
        // // Compute ABA SO derivatives using Analytical (w codegen)---//
        // //------------------------------------------------------------//

        // std::string strfun_d2qdddq_ana_cg = robot_name + std::string("_d2qdd_dq_ana_cg");
        // std::string strfun_d2qdddv_ana_cg = robot_name + std::string("_d2qdd_dv_ana_cg");
        // std::string strfun_d2qdddqv_ana_cg = robot_name + std::string("_d2qdd_dqv_ana_cg");
        // std::string strfun_d2qdddtauq_ana_cg = robot_name + std::string("_d2qdd_dtauq_ana_cg");

        // ::casadi::DM d2qdd_dq_res_ana_cg, d2qdd_dv_res_ana_cg, d2qdd_dqv_res_ana_cg, d2qdd_dtauq_res_ana_cg;
        // std::vector<double> d2qdd_dq_vec_ana_cg, d2qdd_dv_vec_ana_cg, d2qdd_dqv_vec_ana_cg, d2qdd_dtauq_vec_ana_cg;
        // Eigen::Tensor<double, 3> d2qdd_dq_ana_cg(model.nv, model.nv, model.nv);
        // Eigen::Tensor<double, 3> d2qdd_dv_ana_cg(model.nv, model.nv, model.nv);
        // Eigen::Tensor<double, 3> d2qdd_dqv_ana_cg(model.nv, model.nv, model.nv);
        // Eigen::Tensor<double, 3> d2qdd_dtauq_ana_cg(model.nv, model.nv, model.nv);


        //     ::casadi::Function eval_d2qdd_dq_ana_cg = ::casadi::external(strfun_d2qdddq_ana_cg);
        //     ::casadi::Function eval_d2qdd_dv_ana_cg = ::casadi::external(strfun_d2qdddv_ana_cg);
        //     ::casadi::Function eval_d2qdd_dqv_ana_cg = ::casadi::external(strfun_d2qdddqv_ana_cg);
        //     ::casadi::Function eval_d2qdd_dtauq_ana_cg = ::casadi::external(strfun_d2qdddtauq_ana_cg);

        //     timer.tic();
        //     SMOOTH(NBT_SO)
        //     {
        //         Eigen::Map<ConfigVector>(q_vec.data(), model.nq, 1) = qs[_smooth];
        //         Eigen::Map<TangentVector>(v_vec.data(), model.nv, 1) = qdots[_smooth];
        //         Eigen::Map<TangentVector>(tau_vec.data(), model.nv, 1) = taus[_smooth];
        //         Eigen::Map<TangentVector>(a_vec.data(), model.nv, 1) = qddots[_smooth];

        //         // partials w.r.t q
        //         d2qdd_dq_res_ana_cg
        //             = eval_d2qdd_dq_ana_cg(::casadi::DMVector {q_vec, v_int_vec, v_vec, tau_vec, a_vec})[0];
        //         d2qdd_dq_vec_ana_cg = static_cast<std::vector<double>>(d2qdd_dq_res_ana_cg);
        //         d2qdd_dq_ana_cg = Eigen::TensorMap<Eigen::Tensor<double, 3, Eigen::ColMajor>>(
        //             &d2qdd_dq_vec_ana_cg[0], model.nv, model.nv, model.nv);

        //         // partials w.r.t v
        //         d2qdd_dv_res_ana_cg
        //             = eval_d2qdd_dv_ana_cg(::casadi::DMVector {q_vec, v_int_vec, v_vec, tau_vec, a_vec})[0];
        //         d2qdd_dv_vec_ana_cg = static_cast<std::vector<double>>(d2qdd_dv_res_ana_cg);
        //         d2qdd_dv_ana_cg = Eigen::TensorMap<Eigen::Tensor<double, 3, Eigen::ColMajor>>(
        //             &d2qdd_dv_vec_ana_cg[0], model.nv, model.nv, model.nv);

        //         // partials w.r.t q,v
        //         d2qdd_dqv_res_ana_cg
        //             = eval_d2qdd_dqv_ana_cg(::casadi::DMVector {q_vec, v_int_vec, v_vec, tau_vec, a_vec})[0];
        //         d2qdd_dqv_vec_ana_cg = static_cast<std::vector<double>>(d2qdd_dqv_res_ana_cg);
        //         d2qdd_dqv_ana_cg = Eigen::TensorMap<Eigen::Tensor<double, 3, Eigen::ColMajor>>(
        //             &d2qdd_dqv_vec_ana_cg[0], model.nv, model.nv, model.nv);

        //         // partials w.r.t q,tau
        //         d2qdd_dtauq_res_ana_cg
        //             = eval_d2qdd_dtauq_ana_cg(::casadi::DMVector {q_vec, v_int_vec, v_vec, tau_vec, a_vec})[0];
        //         d2qdd_dtauq_vec_ana_cg = static_cast<std::vector<double>>(d2qdd_dtauq_res_ana_cg);
        //         d2qdd_dtauq_ana_cg = Eigen::TensorMap<Eigen::Tensor<double, 3, Eigen::ColMajor>>(
        //             &d2qdd_dtauq_vec_ana_cg[0], model.nv, model.nv, model.nv);
        //     }
        //     time_ABA[4] = timer.toc() / NBT_SO;
        //     std::cout << "ABA SO derivatives Analytical + codegen= \t" << time_ABA[4] << endl;
        

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