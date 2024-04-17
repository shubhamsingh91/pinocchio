/* Date created on- 4/14/24
Modified by- Shubham Singh, singh281@utexas.edu

This version compares the CPU Accuracy for
1. ABA SO analytical derivatives         
2. ABA SO partials using FD SO derivatives

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
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/ID_FO_AZA.hpp"
#include <fstream>
#include "pinocchio/utils/timer.hpp"
#include <string>
#include <iostream>
#include <ctime>
#include "pinocchio/utils/tensor_utils.hpp"
#include <casadi/casadi.hpp>
#include "pinocchio/autodiff/casadi.hpp"
#include "pinocchio/utils/tensor_utils.hpp"
#include "pinocchio/algorithm/aba-derivatives-faster.hpp"
#include "pinocchio/algorithm/aba-derivatives.hpp"

using namespace std;
using namespace pinocchio;
using namespace pinocchio::casadi;


int main(int argc, const char* argv[])
{
    using CppAD::AD;
    using CppAD::NearEqual;

    using namespace Eigen;

    PinocchioTicToc timer(PinocchioTicToc::US);

#ifdef NDEBUG
    // int NBT= 1; // 50000 initially
    int NBT = 1;    // 50000 initially, then 1000*100
    int NBT_SO = 1; // 50000 initially, then 1000*100

#else
    int NBT = 1;
    std::cout << "(the time score in debug mode is not relevant) " << std::endl;
#endif

    char tmp[256];
    getcwd(tmp, 256);

    Model model;
    bool with_ff;

    std::string model_name;
    std::cout << "Enter the model name " << std::endl;
    std::cin >> model_name;
    cout << "with_ff = " << endl;
    cin >> with_ff;

    string str_file_ext;
    string robot_name = "";
    string str_urdf;

    std ::string filename = "../models/" + model_name + std::string(".urdf");

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
    } else if (*argv[1] == 'g') {
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
    PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) v(NBT);
    PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) v_zero(NBT);
    PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) taus(NBT);
    PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) qdd(NBT);
    PINOCCHIO_ALIGNED_STD_VECTOR(MatrixXd) tau_mat(NBT);

        // randomizing input data here

        for (size_t i = 0; i < NBT; ++i) {
            qs[i] = randomConfiguration(model, -qmax, qmax);
            v[i] = Eigen::VectorXd::Random(model.nv);
            qdd[i] = Eigen::VectorXd::Random(model.nv);
            taus[i] = Eigen::VectorXd::Random(model.nv);
            v_zero[i] = Eigen::VectorXd::Zero(model.nv);
        }

        //-----------------------3 ---------------------------//
        //----------------------------------------------------//
        // Compute ABA SO derivatives (FDSVA SO)--------------//
        //----------------------------------------------------//

        Eigen::MatrixXd daba_dq(model.nv, model.nv);
        daba_dq.setZero();
        Eigen::MatrixXd daba_dv(model.nv, model.nv);
        daba_dv.setZero();
        Eigen::MatrixXd daba_dtau(model.nv, model.nv);
        daba_dtau.setZero();

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

        SMOOTH(NBT_SO)
        {
            pinocchio::computeABADerivatives(
                model, data, qs[_smooth], v[_smooth], taus[_smooth], daba_dq, daba_dv, daba_dtau);

            ComputeRNEASecondOrderDerivatives(model, data, qs[_smooth], v[_smooth], qdd[_smooth], dtau2_dq_ana, dtau2_dv_ana,
                dtau2_dqv_ana, M_FO);

            Minv = daba_dtau;
            Minv_neg = -Minv;
            //--- For models N<30------------------//
            // Inner term Compute using DTM (or DMM)
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

        //-----------------------4 ------------------------------//
        //-------------------------------------------------------//
        // Compute ABA SO derivatives using finite-derivatives---//
        //-------------------------------------------------------//

        SMOOTH(NBT) {
        pinocchio::computeABADerivatives(
                model, data, qs[_smooth], v[_smooth], taus[_smooth], daba_dq, daba_dv, daba_dtau);
        }

        // perturbed variables here

        PINOCCHIO_EIGEN_PLAIN_ROW_MAJOR_TYPE(MatrixXd)
        daba_dq_plus(MatrixXd::Zero(model.nv, model.nv));
        PINOCCHIO_EIGEN_PLAIN_ROW_MAJOR_TYPE(MatrixXd)
        daba_dv_plus(MatrixXd::Zero(model.nv, model.nv));
        MatrixXd daba_dtau_plus(MatrixXd::Zero(model.nv, model.nv));

        Data::Tensor3x daba2_dq(model.nv, model.nv, model.nv);
        Data::Tensor3x daba2_dv(model.nv, model.nv, model.nv);
        Data::Tensor3x daba2_qv(model.nv, model.nv, model.nv);
        Data::Tensor3x daba2_tauq(model.nv, model.nv, model.nv);

        VectorXd v_eps(VectorXd::Zero(model.nv));
        VectorXd a_eps(VectorXd::Zero(model.nv));
        VectorXd q_plus(model.nq);
        VectorXd qd_plus(model.nv);
        VectorXd tau_plus(model.nv);

        MatrixXd temp_mat1(MatrixXd::Zero(model.nv, model.nv));
        MatrixXd temp_mat2(MatrixXd::Zero(model.nv, model.nv));
        MatrixXd temp_mat3(MatrixXd::Zero(model.nv, model.nv));

        double alpha = 1e-7; // performs well

        // Partial wrt q
        SMOOTH(NBT) {
        for (int k = 0; k < model.nv; ++k) {
            v_eps[k] += alpha;
            q_plus = integrate( model, qs[_smooth], v_eps); // This is used to add the v_eps to q in the k^th direction
            computeABADerivatives(model, data, q_plus, v[_smooth], taus[_smooth], daba_dq_plus, daba_dv_plus, daba_dtau_plus);
            temp_mat1 = (daba_dq_plus - daba_dq) / alpha;
            temp_mat2 = (daba_dtau_plus - daba_dtau) / alpha; // MSO partial of dqdd_dq wrt tau
            temp_mat2.triangularView<Eigen::StrictlyLower>() =
                temp_mat2.transpose().triangularView<Eigen::StrictlyLower>();
            hess_assign_fd_v1(daba2_dq, temp_mat1, model.nv, k);
            hess_assign_fd_v1(daba2_tauq, temp_mat2, model.nv, k);
            v_eps[k] -= alpha;
        }

        }
        // // Partial wrt qd
        // for (int k = 0; k < model.nv; ++k) {
        //     v_eps[k] += alpha;
        //     qd_plus =
        //         v + v_eps; // This is used to add the v_eps to q in the k^th direction
        //     computeRNEADerivatives(model, data, q, qd_plus, a, drnea_dq_plus,
        //                         drnea_dv_plus, drnea_da_plus);
        //     temp_mat1 = (drnea_dv_plus - drnea_dv) / alpha; // SO partial wrt qdot
        //     temp_mat2 =
        //         (drnea_dq_plus - drnea_dq) / alpha; // MSO partial of dtau_dq wrt qdot
        //     hess_assign_fd_v1(dtau2_dqd, temp_mat1, model.nv, k);
        //     hess_assign_fd_v1(dtau2_qv, temp_mat2, model.nv, k);
        //     v_eps[k] -= alpha;
        // }

        // difference variables
        Data::Tensor3x temptens1(model.nv, model.nv, model.nv);
        Data::Tensor3x temptens2(model.nv, model.nv, model.nv);
        Data::Tensor3x temptens3(model.nv, model.nv, model.nv);
        Data::Tensor3x temptens4(model.nv, model.nv, model.nv);
        double temp_q_SO, temp_v_SO, temp_qv_SO, temp_tauq_SO;

        temptens1 = daba2_dq_ana - daba2_dq;
        // temptens2 = dtau2_dv_ana - dtau2_dqd;
        // temptens3 = dtau2_dqv_ana - dtau2_qv;
        // temptens4 = dtau_dadq_ana - dtau2_qa;

        temp_q_SO = get_tens_diff_norm(daba2_dq_ana, daba2_dq, model.nv);
        // temp_v_SO = get_tens_diff_norm(dtau2_dv_ana, dtau2_dqd, model.nv);
        // temp_qv_SO = get_tens_diff_norm(dtau2_dqv_ana, dtau2_qv, model.nv);
        // temp_qa_SO = get_tens_diff_norm(dtau_dadq_ana, dtau2_qa, model.nv);

        std::cout << "daba2_dq_ana = " << daba2_dq_ana << std::endl;
        std::cout << "daba2_dq = " << daba2_dq << std::endl;

        std::cout << "---------------------------------------------------------------"
                    "-------------------"
                    << std::endl;
        std::cout << "Difference in the SO partial w.r.t q for FD with Ana max val"
                    << (temptens1.abs()).maximum() << std::endl;
        std::cout << "Difference in the SO partial w.r.t q for FD with Ana norm"
                    << temp_q_SO << std::endl;

        // std::cout << "---------------------------------------------------------------"
        //             "-------------------"
        //             << std::endl;
        // std::cout << "Difference in the SO partial w.r.t v for FD with Ana max val"
        //             << (temptens2.abs()).maximum() << std::endl;
        // std::cout << "Difference in the SO partial w.r.t v for FD with Ana norm"
        //             << temp_v_SO << std::endl;
        // std::cout << "---------------------------------------------------------------"
        //             "-------------------"
        //             << std::endl;
        // std::cout << "Difference in the SO partial w.r.t q,v for FD with Ana max val"
        //             << (temptens3.abs()).maximum() << std::endl;
        // std::cout << "Difference in the SO partial w.r.t q,v for FD with Ana norm"
        //             << temp_qv_SO << std::endl;
        // std::cout << "---------------------------------------------------------------"
        //             "-------------------"
        //             << std::endl;
        // std::cout << "Difference in the SO partial w.r.t a,q for FD with Ana max val"
        //             << (temptens4.abs()).maximum() << std::endl;
        // std::cout << "Difference in the SO partial w.r.t q,v for FD with Ana norm"
        //             << temp_qa_SO << std::endl;


    return 0;
}

