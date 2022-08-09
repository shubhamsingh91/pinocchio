

#include "pinocchio/algorithm/M_FO_v1.hpp"
#include "pinocchio/algorithm/aba-derivatives.hpp"
#include "pinocchio/algorithm/aba.hpp"
#include "pinocchio/algorithm/cholesky.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/rnea-derivatives-faster.hpp"
#include "pinocchio/algorithm/rnea-derivatives.hpp"
#include "pinocchio/codegen/code-generator-algo.hpp"
#include "pinocchio/codegen/cppadcg.hpp"
#include "pinocchio/container/aligned-vector.hpp"
#include "pinocchio/parsers/sample-models.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/utils/tensor_utils.hpp"
#include "pinocchio/utils/timer.hpp"
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>

using namespace std;

bool replace(std::string &str, const std::string &from, const std::string &to);

int main(int argc, const char *argv[]) {
  using namespace Eigen;
  using namespace pinocchio;
  using CppAD::AD;
  using CppAD::NearEqual;

  PinocchioTicToc timer(PinocchioTicToc::US);

#ifdef NDEBUG
  int NBT = 10000; // 50000 initially
  int NBT_SO = 1000;

#else
  int NBT = 1;
  std::cout << "(the time score in debug mode is not relevant) " << std::endl;
#endif

  int bf_vec[] = {2}; // CHANGE THIS
                      //  int bf_vec[] = {1,2,5};  // CHANGE THIS

  // Regular Run cases
  // CHANGE THIS
  int n_vec[] = {2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,
                 14,  15,  16,  17,  18,  19,  20,  25,  30,  35,  40,  45,
                 50,  55,  60,  65,  70,  80,  90,  100, 120, 150, 180, 200,
                 220, 250, 280, 300, 320, 350, 380, 400, 420, 450, 480, 500};
  int n_vec_5[] = {5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,
                   17,  18,  19,  20,  25,  30,  35,  40,  45,  50,  55,  60,
                   65,  70,  80,  90,  100, 120, 150, 180, 200, 220, 250, 280,
                   300, 320, 350, 380, 400, 420, 450, 480, 500};

  char tmp[256];
  getcwd(tmp, 256);

  int bf; // branching factor

  // 3
  for (int pp = 0; pp < 1; pp++) {

    bf = bf_vec[pp];
    int len_n;

    if (bf == 5) {

      len_n = 45;

    } else {

      len_n = 48; // CHANGE THIS- 48 originally
    }

    std::cout << "-----------------------------------" << std::endl;
    std::cout << "bf = " << bf << std::endl;
    std::cout << "-----------------------------------" << std::endl;

    // 48
    for (int jj = 0; jj < len_n; jj++) {

      Model model;

      string str_urdf = "", str_file_ext, n_str, n_bf;
      string robot_name = "";

      int n_links;

      if (bf == 5) {
        n_links = n_vec_5[jj];
      } else {
        n_links = n_vec[jj];
      }

      if (n_links > 100) {
        NBT = 100;
        NBT_SO = 5;
      }
      if (n_links > 200) {
        NBT = 10;
        NBT_SO = 3;
      }
      if (n_links > 300) {
        NBT = 5;
        NBT_SO = 2;
      }

      cout << "n = " << n_links << endl;

      n_str = to_string(n_links);
      n_bf = to_string(bf);

      robot_name.append(n_str);
      robot_name.append("link_bf_");
      robot_name.append(n_bf);

      str_file_ext = tmp;
      replace(str_file_ext, "benchmark", "models");
      str_file_ext.append("/");
      str_urdf.append(str_file_ext);
      str_urdf.append(robot_name);
      str_urdf.append(".urdf");

      std ::string filename = str_urdf;

      ofstream file1;

      string filewrite = tmp;

      if (*argv[1] == 'c') {
        replace(filewrite, "pinocchio/benchmark",
                "Data/tree_v6/avx/clang/bf"); // CHANGE
      } else if (*argv[1] == 'g') {
        replace(filewrite, "pinocchio/benchmark",
                "Data/tree_v6/avx/gcc/bf"); // CHANGE
      } else {
      }
      filewrite.append(n_bf);
      filewrite.append("/");
      filewrite.append(robot_name);
      filewrite.append(".txt");

      file1.open(filewrite);

      bool with_ff = false; // true originally

      if (filename == "HS")
        buildModels::humanoidRandom(model, true);
      else if (with_ff)
        pinocchio::urdf::buildModel(filename, JointModelFreeFlyer(), model);
      //      pinocchio::urdf::buildModel(filename,JointModelRX(),model);
      else
        pinocchio::urdf::buildModel(filename, model);

      Data data(model);
      VectorXd qmax = Eigen::VectorXd::Ones(model.nq);

      PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) qs(NBT);
      PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) qdots(NBT);
      PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) taus(NBT);
      PINOCCHIO_ALIGNED_STD_VECTOR(VectorXd) qddots(NBT);
      PINOCCHIO_ALIGNED_STD_VECTOR(MatrixXd) tau_mat(NBT);

      for (size_t i = 0; i < NBT; ++i) {
        qs[i] = randomConfiguration(model, -qmax, qmax);
        qdots[i] = Eigen::VectorXd::Random(model.nv);
        taus[i] = Eigen::VectorXd::Random(model.nv);
        qddots[i] = Eigen::VectorXd::Random(model.nv);
        tau_mat[i] = Eigen::MatrixXd::Identity(model.nv, model.nv);
      }

      double time_vec[2]; // CHANGE

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
        file1 << time_vec[ii] << "\n" << endl;
      }
      file1.close();
    }
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