

/* Date created on- 7/7/22
Modified by- Shubham Singh, sing281@utexas.edu

This version compares the CPU Runtime for

1. RNEA
2. RNEA FO analytical derivatives


*/
// Modifications

#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/rnea-derivatives-faster.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/codegen/code-generator-algo.hpp"
#include "pinocchio/codegen/cppadcg.hpp"
#include "pinocchio/container/aligned-vector.hpp"
#include "pinocchio/parsers/sample-models.hpp"
#include "pinocchio/parsers/urdf.hpp"

#include <fstream>

#include "pinocchio/utils/tensor_utils.hpp"
#include "pinocchio/utils/timer.hpp"
#include <ctime>
#include <iostream>
#include <string>

using namespace std;

bool replace(std::string &str, const std::string &from, const std::string &to);

int main(int argc, const char *argv[]) {
  using CppAD::AD;
  using CppAD::NearEqual;

  using namespace Eigen;
  using namespace pinocchio;

  PinocchioTicToc timer(PinocchioTicToc::US);

#ifdef NDEBUG
  // int NBT= 1; // 50000 initially
  int NBT = 10000;   // 50000 initially, then 1000*100
  int NBT_SO = 1000; // 1000 initially

#else
  int NBT = 1;
  std::cout << "(the time score in debug mode is not relevant) " << std::endl;
#endif

  string str_robotname[12];

  str_robotname[0] = "double_pendulum"; // double pendulum
  str_robotname[1] = "ur3_robot";       // UR3
  str_robotname[2] = "ur5_robot";       // UR5
  str_robotname[3] = "ur10_robot";      // UR10
  str_robotname[4] = "kuka_peg_lwr";    // kuka
  str_robotname[5] = "anymal";          // anymal
  str_robotname[6] = "hyq";             // hyq
  str_robotname[7] = "baxter_simple";   // baxter
  str_robotname[8] = "atlas_v1";        // atlas
  str_robotname[9] = "simple_humanoid"; // simple humanoid
  str_robotname[10] = "talos_full";     // TALOS

  char tmp[256];
  getcwd(tmp, 256);

  double time_vec[2]; // CHANGE

  for (int mm = 0; mm < 11; mm++) {

    Model model;

    string str_file_ext;
    string robot_name = "";
    string str_urdf;

    robot_name = str_robotname[mm];

    str_file_ext = tmp;
    replace(str_file_ext, "benchmark", "models");
    str_file_ext.append("/");
    str_urdf.append(str_file_ext);
    str_urdf.append(robot_name);
    str_urdf.append(".urdf");

    std ::string filename = str_urdf;

    bool with_ff = false; // true originally

    if ((mm > 4) && (mm < 11)) {
      with_ff = true; // True for anymal and atlas, true originally
    }

    if (mm == 7) {
      with_ff = false; // False for Baxter
    }

    if (filename == "HS")
      buildModels::humanoidRandom(model, true);
    else if (with_ff)
      pinocchio::urdf::buildModel(filename, JointModelFreeFlyer(), model);
    //      pinocchio::urdf::buildModel(filename,JointModelRX(),model);
    else
      pinocchio::urdf::buildModel(filename, model);

    cout << "------------------------------------------" << endl;
    cout << "Model is" << robot_name << endl;
    cout << "with ff = " << with_ff << "\n \n";
    std::cout << "nq = " << model.nq << std::endl;
    std::cout << "nv = " << model.nv << std::endl;

    //-- opening filename here
    ofstream file1;
    string filewrite = tmp;

    if (*argv[1] == 'c') {
      replace(filewrite, "pinocchio/benchmark",
              "Data/tree/avx/clang/mdof/"); // CHANGE
    } else if (*argv[1] == 'g') {
      replace(filewrite, "pinocchio/benchmark",
              "Data/tree/avx/gcc/mdof/"); // CHANGE
    } else {
    }

    filewrite.append(robot_name);
    filewrite.append(".txt");

    file1.open(filewrite);

    Data data(model);
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
      qdots_zero[i] = Eigen::VectorXd::Zero(model.nv);
      taus[i] = Eigen::VectorXd::Random(model.nv);
      qddots[i] = Eigen::VectorXd::Random(model.nv);
      tau_mat[i] =
          Eigen::MatrixXd::Zero(model.nv, 2 * model.nv); // new variable
    }

    //----------------------------------------------------//
    // RNEA -----------------------------------------------//
    //----------------------------------------------------//

    timer.tic();
    SMOOTH(NBT) {
      rnea(model, data, qs[_smooth], qdots[_smooth], qddots[_smooth]);
    }
    time_vec[0] = timer.toc() / NBT; // RNEA timing

    //----------------------------------------------------//
    // Compute RNEA derivatives faster--------------------//
    //----------------------------------------------------//
    MatrixXd drnea2_dq(MatrixXd::Zero(model.nv, model.nv));
    MatrixXd drnea2_dv(MatrixXd::Zero(model.nv, model.nv));
    MatrixXd drnea2_da(MatrixXd::Zero(model.nv, model.nv));

    timer.tic();
    SMOOTH(NBT) {
      computeRNEADerivativesFaster(model, data, qs[_smooth], qdots[_smooth],
                                   qddots[_smooth], drnea2_dq, drnea2_dv,
                                   drnea2_da);
    }
    time_vec[1] = timer.toc() / NBT; // RNEAF timing

    //------------------------------------------------//
    // Writing all the timings to the file
    //------------------------------------------------//

    for (int ii = 0; ii < 2; ii++) { // CHANGE
      file1 << time_vec[ii] << endl;
    }
    file1.close();
  }

  return 0;
}

bool replace(std::string &str, const std::string &from, const std::string &to) {
  size_t start_pos = str.find(from);
  if (start_pos == std::string::npos)
    return false;
  str.replace(start_pos, from.length(), to);
  return true;
}