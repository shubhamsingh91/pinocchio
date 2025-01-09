#include <stdio.h>
#include "pinocchio/parsers/sample-models.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include <Eigen/Dense>
#include <gtest/gtest.h>
// #include <pinocchio/spatial/explog.hpp>

using namespace std;
using namespace pinocchio;

TEST(Test1, LogExp) {
    // Setup
    Eigen::Vector3d p(Eigen::Vector3d::Random());
    auto vec = log3(exp3(p)); // so3 to SO3 to so

    pinocchio::SE3 M = pinocchio::SE3::Random();
    pinocchio::SE3::Matrix3 M_res = exp3(log3(M.rotation())); // SO3 to so3 to SO3

    pinocchio::SE3 M2 = pinocchio::SE3::Random();
    pinocchio::SE3 M4 = M2.inverse();
    auto M5 = M2*M4;

    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();

    auto vec2 = log6(M2); // SE3 to se3
    auto M3 = exp6(vec2); // se3 to SE3

    // Verify
    EXPECT_NEAR((vec-p).norm(), 0.0, 1e-10);
    EXPECT_NEAR((M_res-M.rotation()).norm(), 0.0, 1e-10);
    EXPECT_NEAR((M2.translation()-M3.translation()).norm(), 0.0, 1e-10);
    EXPECT_NEAR((M2.rotation()-M3.rotation()).norm(), 0.0, 1e-10);
    EXPECT_NEAR(M5.translation().norm(), 0.0, 1e-10);
    EXPECT_TRUE(M5.rotation().isApprox(I));

}


template <typename T, typename U>
void print(const T& var1, const U& var2) {
    std::cout << var1 << " = " << var2 << std::endl;
}

template <typename T>
void print(const T& var1) {
    std::cout << var1 << std::endl;
}

int main() {

typedef double Scalar;
enum {Options = 0};

// testing SE2 object and it's tangent vector
SpecialEuclideanOperationTpl<2,Scalar,Options> aSE2;
SpecialEuclideanOperationTpl<2,Scalar,Options>::ConfigVector_t pose_s,pose_g, pose_check; // dim = 4 (x,y,cos(theta), sin(theta))
SpecialEuclideanOperationTpl<2,Scalar,Options>::TangentVector_t delta_u; /// dim = 3 (dx,dy,dtheta)

pose_s(0) = 1.0; pose_s(1) = 1.0;
pose_s(2) = cos(M_PI/4.0); pose_s(3) = sin(M_PI/4.0);

pose_g(0) = 1.0; pose_g(1) = 1.0;
pose_g(2) = cos(-M_PI/2.0); pose_g(3) = sin(-M_PI/2.0);

aSE2.difference(pose_s,pose_g,delta_u);
std::cout << "delta_u = \n" << delta_u << std::endl;

aSE2.integrate(pose_s,delta_u,pose_check);
std::cout << "pose_check = \n" << pose_check << std::endl;

// special euclidean group in 3D
typedef SpecialEuclideanOperationTpl<3,Scalar,Options>  SE3;
SE3::ConfigVector_t qvec, qout ;
SE3::TangentVector_t vvec;
SE3 aSE3;

print(aSE3.name());
print(aSE3.nq());
print(aSE3.nv());

pinocchio::SE3::Quaternion quaternion;
quaternion = Eigen::Quaternion<Scalar>(1,0.1,0.2,0.3);
quaternion.normalize();
qvec.head<4>() = quaternion.coeffs();
qvec.tail<3>() << 0.4, 0.5, 0.6;
vvec << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6;
qvec = aSE3.random();
qvec = aSE3.neutral();

print("qvec", qvec);
print("vvec", vvec);

aSE3.integrate(qvec,vvec,qout);
print("qout", qout);

std::cout << "------SE3 object-----------" << std::endl;
pinocchio::SE3 M(pinocchio::SE3::Random());
print("M.rotation = ",M.rotation());
print("M.translation = ",M.translation());

 // Running google tests

 ::testing::InitGoogleTest();


 // testing some more SE3 objects in pinocchio
  pinocchio::SE3 M2(M.rotation(),Eigen::Vector3d::Random());
  print("M2 = ",M2);
  

  return RUN_ALL_TESTS();


}