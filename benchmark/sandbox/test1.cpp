#include <iostream>
#include <stdio.h>
#include "pinocchio/parsers/sample-models.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include <Eigen/Dense>


using namespace std;
using namespace pinocchio;

class fun: serialization::Serializable<fun>{
  
  private:
    int num1;
    
  public:
   fun(int n):num1{n}{cout << "calling fun constructor!" << endl;};
   void print_num1(){cout << "num1 = " << num1 << endl;}
   void set_num1(int n){num1=n;};

   ~fun(){cout << "calling fun destructor!" << endl;};

};

int main(){
 

  pinocchio::Model model;
  buildModels::humanoidRandom(model,true);
  Data data(model);
 
  Eigen::VectorXd q = pinocchio::neutral(model);
  Eigen::VectorXd v = Eigen::VectorXd::Random(model.nv);
  Eigen::VectorXd a = Eigen::VectorXd::Zero(model.nv);
 
  const Eigen::VectorXd & tau = pinocchio::rnea(model,data,q,v,a);
  std::cout << "tau = " << tau.transpose() << std::endl;

  // ...Testing some data structures...
  
  // cout << "model njoints = " << model.njoints << endl;


  // Testng serializable derived objects
  fun f1{56};
  f1.set_num1(12);
  f1.print_num1();

  serialization::Serializable<fun> f2;

  // testing JointCollectionDefaultTpl

  JointCollectionDefaultTpl<float,0> ::JointModelRX j1; 
  auto data1 = j1.createData();
  Eigen::VectorXd v1(3);

  j1.calc(data1,v1);
  cout << j1.classname() << endl;

 // testing SE3 class objects
  
  cout << "---------------------------------" << endl;

   SE3Tpl<double,0> s1;
  
   Eigen::Quaterniond quat(1.0,1.0,1.0,0.0);
   Eigen::Vector3d trans(1.0,2.0,3.0);

   SE3Tpl<double> s2(quat, trans);
  
   cout << "s2 rot = " << s2.rotation() << endl;
   cout << "s2 trans = " << s2.translation() << endl;

   cout << "s2 homo = " << s2.toHomogeneousMatrix() << endl;

    pinocchio::traits< SE3Tpl<double,0>>::ActionMatrixType action_matrix = s2;
    cout << "action_matrix-1 = " << action_matrix << endl;

    cout << "to action matrix-2 =" << s2.toActionMatrix() << endl;
    cout << "to action matrix-3 =" << s2.toActionMatrix_impl() << endl;

   Eigen::Vector3d vec1{1.0,2.0,1.0};
   cout << "action on vec1 = \n" << s2.actOnEigenObject(vec1) << endl;


  cout << "------------------------- Testing ForceTpl------------------------" << endl;
 
  // pinocchio::ForceBase<ForceDense<ForceTpl<double,0>>> f3; // causing error
  pinocchio::ForceDense<ForceTpl<double,0>> f3;
  f3.setRandom();
  cout << "f3 linear part = \n" << f3.linear() << endl;
    // cout << "f3 angular part =" << f3.angular() << endl;






  


return 0;


}