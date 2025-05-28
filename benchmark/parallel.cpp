/* Date created on- 5/27/25
Testing the ID SO derivs in parallel

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
#include <fstream>
#include "pinocchio/utils/timer.hpp"
#include <string>
#include <iostream>
#include <ctime>
#include "pinocchio/utils/tensor_utils.hpp"
#include <casadi/casadi.hpp>
#include "pinocchio/autodiff/casadi.hpp"
#include "pinocchio/utils/tensor_utils.hpp"
#include <omp.h>
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
        int NBT_SO = 5000; // 50000 initially, then 1000*100

    #else
        int NBT = 1;
        std::cout << "(the time score in debug mode is not relevant) " << std::endl;
    #endif

    string str_robotname;

    str_robotname = "atlas";           // atlas

    char tmp[256];
    getcwd(tmp, 256);


    Model model;
    bool with_ff = true;

    string str_file_ext;
    string robot_name = "";
    string str_urdf;

    robot_name = str_robotname;
    std ::string filename = "../models/" + robot_name + std::string(".urdf");

    with_ff = true; // True for hyQ and atlas, talos_full_v2
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
    std::cout << "NBT_SO = " << NBT_SO << std::endl;
    std::cout << "OMP max threads: " << omp_get_max_threads() << std::endl;

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

    PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) qs(NBT_SO);
    PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) qdots(NBT_SO);
    PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) v_zero(NBT_SO);
    PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) taus(NBT_SO);
    PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) qddots(NBT_SO);
    PINOCCHIO_ALIGNED_STD_VECTOR(MatrixXd) tau_mat(NBT_SO);

    // randomizing input data here

    for (size_t i = 0; i < NBT_SO; ++i) {
        qs[i] = randomConfiguration(model, -qmax, qmax);
        qdots[i] = Eigen::VectorXd::Random(model.nv);
        qddots[i] = Eigen::VectorXd::Zero(model.nv);
        taus[i] = Eigen::VectorXd::Random(model.nv);
        v_zero[i] = Eigen::VectorXd::Zero(model.nv);
    }
    
    std::vector<Data> datas(NBT_SO, Data(model));

    std::vector<Eigen::Tensor<double, 3>> dtau2_dq_ana(NBT_SO);
    std::vector<Eigen::Tensor<double, 3>> dtau2_dv_ana(NBT_SO);
    std::vector<Eigen::Tensor<double, 3>> dtau2_dqv_ana(NBT_SO);
    std::vector<Eigen::Tensor<double, 3>> M_FO(NBT_SO);

    for (int i = 0; i < NBT_SO; ++i) {
        dtau2_dq_ana[i]  = Eigen::Tensor<double, 3>(model.nv, model.nv, model.nv);
        dtau2_dv_ana[i]  = Eigen::Tensor<double, 3>(model.nv, model.nv, model.nv);
        dtau2_dqv_ana[i] = Eigen::Tensor<double, 3>(model.nv, model.nv, model.nv);
        M_FO[i]          = Eigen::Tensor<double, 3>(model.nv, model.nv, model.nv);

        dtau2_dq_ana[i].setZero();
        dtau2_dv_ana[i].setZero();
        dtau2_dqv_ana[i].setZero();
        M_FO[i].setZero();
    }
        
    timer.tic();
    #pragma omp parallel for
    for (int i = 0; i < NBT_SO; ++i)
    {
        ComputeRNEASecondOrderDerivatives(model, datas[i], qs[i], qdots[i], qddots[i], dtau2_dq_ana[i], dtau2_dv_ana[i],
         dtau2_dqv_ana[i], M_FO[i]);

    }
    std::cout << "Parallel RNEA SO derivs time: " << timer.toc() / NBT_SO << " us/sample" << std::endl;
    
    timer.tic();
    for (int i = 0; i < NBT_SO; ++i)
    {
        ComputeRNEASecondOrderDerivatives(model, data, qs[i], qdots[i], qddots[i], dtau2_dq_ana[i], dtau2_dv_ana[i],
            dtau2_dqv_ana[i], M_FO[i]);
    }
    std::cout << "Serial RNEA SO derivs time: " << timer.toc() / NBT_SO << " us/sample" << std::endl;
     


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