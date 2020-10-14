//
// Copyright (c) 2020 LAAS-CNRS, INRIA
//

#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/rnea-derivatives.hpp"
#include "pinocchio/algorithm/aba-derivatives.hpp"
#include "pinocchio/algorithm/kinematics-derivatives.hpp"
#include "pinocchio/algorithm/contact-dynamics.hpp"
#include "pinocchio/algorithm/contact-dynamics-derivatives.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/parsers/sample-models.hpp"
#include "pinocchio/spatial/classic-acceleration.hpp"

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/utility/binary.hpp>

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

using namespace Eigen;
using namespace pinocchio;

BOOST_AUTO_TEST_CASE(test_sparse_contact_dynamics_derivatives_no_contact)
{
  using namespace Eigen;
  using namespace pinocchio;

  Model model;
  buildModels::humanoidRandom(model,true);
  Data data(model), data_ref(model);
  
  model.lowerPositionLimit.head<3>().fill(-1.);
  model.upperPositionLimit.head<3>().fill( 1.);
  VectorXd q = randomConfiguration(model);
  
  VectorXd v = VectorXd::Random(model.nv);
  VectorXd tau = VectorXd::Random(model.nv);
  
  // Contact models and data
  const PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidContactModel) empty_contact_models;
  PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidContactData) empty_contact_data;

  const Eigen::DenseIndex constraint_dim = 0;
  const double mu0 = 0.;

  initContactDynamics(model,data,empty_contact_models);
  contactDynamics(model,data,q,v,tau,empty_contact_models,empty_contact_data,mu0);
  data.M.triangularView<Eigen::StrictlyLower>()
  = data.M.transpose().triangularView<Eigen::StrictlyLower>();
  computeContactDynamicsDerivatives(model,data,empty_contact_models,empty_contact_data,mu0);
  
  // Reference values
  computeABADerivatives(model, data_ref, q, v, tau);
  forwardKinematics(model, data_ref, q, v, data_ref.ddq);
  BOOST_CHECK(data_ref.ddq.isApprox(data.ddq));
  
  for(size_t k = 1; k < model.njoints; ++k)
  {
    BOOST_CHECK(data_ref.oMi[k].isApprox(data.oMi[k]));
    BOOST_CHECK(data_ref.ov[k].isApprox(data.ov[k]));
    BOOST_CHECK(data_ref.v[k].isApprox(data.v[k]));
    BOOST_CHECK(data_ref.a[k].isApprox(data.a[k]));
    BOOST_CHECK(data_ref.oa[k].isApprox(data.oa[k]));
  }
  
  BOOST_CHECK(data_ref.ddq.isApprox(data.ddq));
  BOOST_CHECK(data_ref.dVdq.isApprox(data.dVdq));
  BOOST_CHECK(data_ref.J.isApprox(data.J));
  BOOST_CHECK(data_ref.dAdq.isApprox(data.dAdq));
  BOOST_CHECK(data_ref.dAdv.isApprox(data.dAdv));
  BOOST_CHECK(data_ref.dFdq.isApprox(data.dFdq));
  BOOST_CHECK(data_ref.dFdv.isApprox(data.dFdv));

  BOOST_CHECK(data_ref.dtau_dq.isApprox(data.dtau_dq));
  BOOST_CHECK(data_ref.dtau_dv.isApprox(data.dtau_dv));
  
  BOOST_CHECK(data_ref.ddq_dq.isApprox(data.ddq_dq));
  BOOST_CHECK(data_ref.ddq_dv.isApprox(data.ddq_dv));
  BOOST_CHECK(data_ref.Minv.isApprox(data.ddq_dtau));
}

BOOST_AUTO_TEST_CASE(test_sparse_contact_dynamics_derivatives)
{
  using namespace Eigen;
  using namespace pinocchio;
  Model model;
  buildModels::humanoidRandom(model,true);
  Data data(model), data_ref(model);
  
  model.lowerPositionLimit.head<3>().fill(-1.);
  model.upperPositionLimit.head<3>().fill( 1.);
  VectorXd q = randomConfiguration(model);
  
  VectorXd v = VectorXd::Random(model.nv);
  VectorXd tau = VectorXd::Random(model.nv);
  
  const std::string RF = "rleg6_joint";
  //  const Model::JointIndex RF_id = model.getJointId(RF);
  const std::string LF = "lleg6_joint";
  //  const Model::JointIndex LF_id = model.getJointId(LF);
  
  // Contact models and data
  PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidContactModel) contact_models;
  PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidContactData) contact_data;

  RigidContactModel ci_LF(CONTACT_6D,model.getJointId(LF),LOCAL);
  RigidContactModel ci_RF(CONTACT_3D,model.getJointId(RF),LOCAL);

  contact_models.push_back(ci_LF); contact_data.push_back(RigidContactData(ci_LF));
  contact_models.push_back(ci_RF); contact_data.push_back(RigidContactData(ci_RF));

  Eigen::DenseIndex constraint_dim = 0;
  for(size_t k = 0; k < contact_models.size(); ++k)
    constraint_dim += contact_models[k].size();
  
  const double mu0 = 0.;

  initContactDynamics(model,data,contact_models);
  contactDynamics(model,data,q,v,tau,contact_models,contact_data,mu0);
  data.M.triangularView<Eigen::StrictlyLower>()
  = data.M.transpose().triangularView<Eigen::StrictlyLower>();
  computeContactDynamicsDerivatives(model,data,contact_models,contact_data,mu0);
  
  // Reference values
  crba(model, data_ref, q);
  data_ref.M.triangularView<Eigen::StrictlyLower>()
  = data_ref.M.transpose().triangularView<Eigen::StrictlyLower>();
  container::aligned_vector<Force> fext(model.njoints,Force::Zero());
  for(size_t k = 0; k < contact_models.size(); ++k)
  {
    const RigidContactModel & cmodel = contact_models[k];
    const RigidContactData & cdata = contact_data[k];
    fext[cmodel.joint1_id] = cdata.contact_force;
  }
  computeABADerivatives(model, data_ref, q, v, tau, fext);
  forwardKinematics(model,data_ref,q,v);
  
  BOOST_CHECK(data.ddq.isApprox(data_ref.ddq));
  BOOST_CHECK(data.dVdq.isApprox(data_ref.dVdq));
  BOOST_CHECK(data.J.isApprox(data_ref.J));
  BOOST_CHECK(data.dAdq.isApprox(data_ref.dAdq));
  BOOST_CHECK(data.dAdv.isApprox(data_ref.dAdv));
  BOOST_CHECK(data.dFdq.isApprox(data_ref.dFdq));
  BOOST_CHECK(data.dFdv.isApprox(data_ref.dFdv));
  
  MatrixXd
    vLF_partial_dq(MatrixXd::Zero(6, model.nv)),
    aLF_partial_dq(MatrixXd::Zero(6, model.nv)),
    aLF_partial_dv(MatrixXd::Zero(6, model.nv)),
    aLF_partial_da(MatrixXd::Zero(6, model.nv));

  MatrixXd
    vRF_partial_dq(MatrixXd::Zero(6, model.nv)),
    aRF_partial_dq(MatrixXd::Zero(6, model.nv)),
    aRF_partial_dv(MatrixXd::Zero(6, model.nv)),
    aRF_partial_da(MatrixXd::Zero(6, model.nv));
  
  getJointAccelerationDerivatives(model, data_ref,
                                  model.getJointId(LF),
                                  LOCAL,
                                  vLF_partial_dq,
                                  aLF_partial_dq,
                                  aLF_partial_dv,
                                  aLF_partial_da);
  getJointAccelerationDerivatives(model, data_ref,
                                  model.getJointId(RF),
                                  LOCAL,
                                  vRF_partial_dq,
                                  aRF_partial_dq,
                                  aRF_partial_dv,
                                  aRF_partial_da);

  MatrixXd Jc(6+3, model.nv);
  Jc << aLF_partial_da,
    aRF_partial_da.topRows<3>();
  
  MatrixXd K(model.nv+6+3, model.nv+6+3);
  K << data_ref.M, Jc.transpose(),
    Jc, MatrixXd::Zero(6+3,6+3);
  const MatrixXd Kinv = K.inverse();

  MatrixXd J_LF(6, model.nv), J_RF(6, model.nv);
  J_LF.setZero(); J_RF.setZero();
  getJointJacobian(model, data_ref, model.getJointId(LF), LOCAL, J_LF);
  getJointJacobian(model, data_ref, model.getJointId(RF), LOCAL, J_RF);
  MatrixXd osim((Jc * data_ref.M.inverse() * Jc.transpose()).inverse());
  BOOST_CHECK(data.osim.isApprox(osim));

  MatrixXd ac_partial_dq(6+3, model.nv);

  BOOST_CHECK(data.ov[model.getJointId(RF)].isApprox(data_ref.oMi[model.getJointId(RF)].act(data_ref.v[model.getJointId(RF)])));
  aRF_partial_dq.topRows<3>() +=
    cross(data_ref.v[model.getJointId(RF)].angular(),
          vRF_partial_dq.topRows<3>())
    - cross(data_ref.v[model.getJointId(RF)].linear(),
            vRF_partial_dq.bottomRows<3>());
  
  BOOST_CHECK(data.ov[model.getJointId(LF)].isApprox(data_ref.oMi[model.getJointId(LF)].act(data_ref.v[model.getJointId(LF)])));
  
  ac_partial_dq << aLF_partial_dq,
    aRF_partial_dq.topRows<3>();

  MatrixXd dac_dq = ac_partial_dq - Jc * data_ref.Minv*data_ref.dtau_dq;

  BOOST_CHECK(data.dac_dq.isApprox(dac_dq,1e-8));
  BOOST_CHECK(Kinv.bottomLeftCorner(6+3, model.nv).isApprox(osim*Jc*data_ref.Minv));

  MatrixXd df_dq = Kinv.bottomLeftCorner(6+3, model.nv)* data_ref.dtau_dq +
    Kinv.bottomRightCorner(6+3,6+3)*ac_partial_dq;
  
  MatrixXd ddq_dq = data_ref.Minv * (-data_ref.dtau_dq +  Jc.transpose() * df_dq);
  
  BOOST_CHECK(df_dq.isApprox(data.dlambda_dq));
  BOOST_CHECK(ddq_dq.isApprox(data.ddq_dq));
}

pinocchio::Motion computeAcceleration(const pinocchio::Model & model,
                                      const pinocchio::Data & data,
                                      const pinocchio::JointIndex & joint_id,
                                      const pinocchio::ReferenceFrame reference_frame,
                                      const pinocchio::SE3 & placement = pinocchio::SE3::Identity())
{
  PINOCCHIO_UNUSED_VARIABLE(model);
  using namespace pinocchio;
  Motion res(Motion::Zero());
  
  const Data::SE3 & oMi = data.oMi[joint_id];
  const Data::SE3 oMc = oMi * placement;
  
  const Data::SE3 & iMc = placement;
  const Motion ov = oMi.act(data.v[joint_id]);
  const Motion oa = oMi.act(data.a[joint_id]);
  
  switch (reference_frame)
  {
    case WORLD:
      classicAcceleration(ov,oa,res.linear());
      res.angular() = oa.angular();
      break;
    case LOCAL_WORLD_ALIGNED:
      res.linear() = oMc.rotation() * classicAcceleration(data.v[joint_id],data.a[joint_id],iMc);
      res.angular() = oMi.rotation() * data.a[joint_id].angular();
      break;
    case LOCAL:
      classicAcceleration(data.v[joint_id],data.a[joint_id],iMc,res.linear());
      res.angular() = iMc.rotation().transpose() * data.a[joint_id].angular();
      break;
    default:
      break;
  }
  
  return res;
}

pinocchio::Motion getContactAcceleration(const Model & model,
                                         const Data & data,
                                         const RigidContactModel & cmodel)
{
  return computeAcceleration(model,data,cmodel.joint1_id,cmodel.reference_frame,cmodel.joint1_placement);
}

BOOST_AUTO_TEST_CASE(test_contact_dynamics_derivatives_LOCAL_6D_fd)
{
  using namespace Eigen;
  using namespace pinocchio;

  Model model;
  buildModels::humanoidRandom(model,true);
  Data data(model), data_fd(model);

  model.lowerPositionLimit.head<3>().fill(-1.);
  model.upperPositionLimit.head<3>().fill( 1.);
  VectorXd q = randomConfiguration(model);

  VectorXd v = VectorXd::Random(model.nv);
  VectorXd tau = VectorXd::Random(model.nv);

  const std::string RF = "rleg6_joint";
  //  const Model::JointIndex RF_id = model.getJointId(RF);
  const std::string LF = "lleg6_joint";
  //  const Model::JointIndex LF_id = model.getJointId(LF);

  // Contact models and data
  PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidContactModel) contact_models;
  PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidContactData) contact_data;

  RigidContactModel ci_LF(CONTACT_6D,model.getJointId(LF),LOCAL);

  contact_models.push_back(ci_LF); contact_data.push_back(RigidContactData(ci_LF));

  Eigen::DenseIndex constraint_dim = 0;
  for(size_t k = 0; k < contact_models.size(); ++k)
    constraint_dim += contact_models[k].size();

  const double mu0 = 0.;

  initContactDynamics(model,data,contact_models);
  contactDynamics(model,data,q,v,tau,contact_models,contact_data,mu0);
  const Data::TangentVectorType a = data.ddq;
  data.M.triangularView<Eigen::StrictlyLower>() = data.M.transpose().triangularView<Eigen::StrictlyLower>();
  computeContactDynamicsDerivatives(model, data, contact_models, contact_data, mu0);

  //Data_fd
  initContactDynamics(model,data_fd,contact_models);

  MatrixXd ddq_partial_dq_fd(model.nv,model.nv); ddq_partial_dq_fd.setZero();
  MatrixXd ddq_partial_dv_fd(model.nv,model.nv); ddq_partial_dv_fd.setZero();
  MatrixXd ddq_partial_dtau_fd(model.nv,model.nv); ddq_partial_dtau_fd.setZero();

  MatrixXd lambda_partial_dtau_fd(constraint_dim,model.nv); lambda_partial_dtau_fd.setZero();
  MatrixXd lambda_partial_dq_fd(constraint_dim,model.nv); lambda_partial_dq_fd.setZero();
  MatrixXd lambda_partial_dv_fd(constraint_dim,model.nv); lambda_partial_dv_fd.setZero();

  const VectorXd ddq0 = contactDynamics(model,data_fd,q,v,tau,contact_models,contact_data,mu0);
  const VectorXd lambda0 = data_fd.lambda_c;
  VectorXd v_eps(VectorXd::Zero(model.nv));
  VectorXd q_plus(model.nq);
  VectorXd ddq_plus(model.nv);

  VectorXd lambda_plus(constraint_dim);

  const double alpha = 1e-8;
  
  // Check dac_dq
  const RigidContactModel & cmodel = contact_models[0];
  
  forwardKinematics(model,data,q,v,a);
  
  for(int k = 0; k < model.nv; ++k)
  {
    v_eps[k] += alpha;
    q_plus = integrate(model,q,v_eps);
    ddq_plus = contactDynamics(model,data_fd,q_plus,v,tau,contact_models,contact_data,mu0);
    ddq_partial_dq_fd.col(k) = (ddq_plus - ddq0)/alpha;
    lambda_partial_dq_fd.col(k) = (data_fd.lambda_c - lambda0)/alpha;
    v_eps[k] = 0.;
  }

  BOOST_CHECK(ddq_partial_dq_fd.isApprox(data.ddq_dq,sqrt(alpha)));
  BOOST_CHECK(lambda_partial_dq_fd.isApprox(data.dlambda_dq,sqrt(alpha)));

  VectorXd v_plus(v);
  for(int k = 0; k < model.nv; ++k)
  {
    v_plus[k] += alpha;
    ddq_plus = contactDynamics(model,data_fd,q,v_plus,tau,contact_models,contact_data,mu0);
    ddq_partial_dv_fd.col(k) = (ddq_plus - ddq0)/alpha;
    lambda_partial_dv_fd.col(k) = (data_fd.lambda_c - lambda0)/alpha;
    v_plus[k] -= alpha;
  }
  
  BOOST_CHECK(ddq_partial_dv_fd.isApprox(data.ddq_dv,sqrt(alpha)));
  BOOST_CHECK(lambda_partial_dv_fd.isApprox(data.dlambda_dv,sqrt(alpha)));

  VectorXd tau_plus(tau);
  for(int k = 0; k < model.nv; ++k)
  {
    tau_plus[k] += alpha;
    ddq_plus = contactDynamics(model,data_fd,q,v,tau_plus,contact_models,contact_data,mu0);
    ddq_partial_dtau_fd.col(k) = (ddq_plus - ddq0)/alpha;
    lambda_partial_dtau_fd.col(k) = (data_fd.lambda_c - lambda0)/alpha;
    tau_plus[k] -= alpha;
  }
  
  BOOST_CHECK(lambda_partial_dtau_fd.isApprox(data.dlambda_dtau,sqrt(alpha)));
  BOOST_CHECK(ddq_partial_dtau_fd.isApprox(data.ddq_dtau,sqrt(alpha)));
}

BOOST_AUTO_TEST_CASE(test_contact_dynamics_derivatives_LOCAL_3D_fd)
{
  using namespace Eigen;
  using namespace pinocchio;

  Model model;
  buildModels::humanoidRandom(model,true);
  Data data(model), data_fd(model);

  model.lowerPositionLimit.head<3>().fill(-1.);
  model.upperPositionLimit.head<3>().fill( 1.);
  VectorXd q = randomConfiguration(model);

  VectorXd v = VectorXd::Random(model.nv);
  VectorXd tau = VectorXd::Random(model.nv);

  const std::string RF = "rleg6_joint";
  //  const Model::JointIndex RF_id = model.getJointId(RF);
  const std::string LF = "lleg6_joint";
  //  const Model::JointIndex LF_id = model.getJointId(LF);

  // Contact models and data
  PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidContactModel) contact_models;
  PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidContactData) contact_data;

  RigidContactModel ci_RF(CONTACT_3D,model.getJointId(RF),LOCAL);

  contact_models.push_back(ci_RF); contact_data.push_back(RigidContactData(ci_RF));

  Eigen::DenseIndex constraint_dim = 0;
  for(size_t k = 0; k < contact_models.size(); ++k)
    constraint_dim += contact_models[k].size();

  const double mu0 = 0.;

  initContactDynamics(model,data,contact_models);
  contactDynamics(model,data,q,v,tau,contact_models,contact_data,mu0);
  const Data::TangentVectorType a = data.ddq;
  data.M.triangularView<Eigen::StrictlyLower>() = data.M.transpose().triangularView<Eigen::StrictlyLower>();
  computeContactDynamicsDerivatives(model, data, contact_models, contact_data, mu0);

  //Data_fd
  initContactDynamics(model,data_fd,contact_models);

  MatrixXd ddq_partial_dq_fd(model.nv,model.nv); ddq_partial_dq_fd.setZero();
  MatrixXd ddq_partial_dv_fd(model.nv,model.nv); ddq_partial_dv_fd.setZero();
  MatrixXd ddq_partial_dtau_fd(model.nv,model.nv); ddq_partial_dtau_fd.setZero();

  MatrixXd lambda_partial_dtau_fd(constraint_dim,model.nv); lambda_partial_dtau_fd.setZero();
  MatrixXd lambda_partial_dq_fd(constraint_dim,model.nv); lambda_partial_dq_fd.setZero();
  MatrixXd lambda_partial_dv_fd(constraint_dim,model.nv); lambda_partial_dv_fd.setZero();

  const VectorXd ddq0 = contactDynamics(model,data_fd,q,v,tau,contact_models,contact_data,mu0);
  const VectorXd lambda0 = data_fd.lambda_c;
  VectorXd v_eps(VectorXd::Zero(model.nv));
  VectorXd q_plus(model.nq);
  VectorXd ddq_plus(model.nv);

  VectorXd lambda_plus(constraint_dim);

  const double alpha = 1e-8;
  
  // Check dac_dq
  const RigidContactModel & cmodel = contact_models[0];
  
  forwardKinematics(model,data,q,v,a);
  
  for(int k = 0; k < model.nv; ++k)
  {
    v_eps[k] += alpha;
    q_plus = integrate(model,q,v_eps);
    ddq_plus = contactDynamics(model,data_fd,q_plus,v,tau,contact_models,contact_data,mu0);
    ddq_partial_dq_fd.col(k) = (ddq_plus - ddq0)/alpha;
    lambda_partial_dq_fd.col(k) = (data_fd.lambda_c - lambda0)/alpha;
    v_eps[k] = 0.;
  }

  BOOST_CHECK(ddq_partial_dq_fd.isApprox(data.ddq_dq,sqrt(alpha)));
  BOOST_CHECK(lambda_partial_dq_fd.isApprox(data.dlambda_dq,sqrt(alpha)));

  VectorXd v_plus(v);
  for(int k = 0; k < model.nv; ++k)
  {
    v_plus[k] += alpha;
    ddq_plus = contactDynamics(model,data_fd,q,v_plus,tau,contact_models,contact_data,mu0);
    ddq_partial_dv_fd.col(k) = (ddq_plus - ddq0)/alpha;
    lambda_partial_dv_fd.col(k) = (data_fd.lambda_c - lambda0)/alpha;
    v_plus[k] -= alpha;
  }
  
  BOOST_CHECK(ddq_partial_dv_fd.isApprox(data.ddq_dv,sqrt(alpha)));
  BOOST_CHECK(lambda_partial_dv_fd.isApprox(data.dlambda_dv,sqrt(alpha)));

  VectorXd tau_plus(tau);
  for(int k = 0; k < model.nv; ++k)
  {
    tau_plus[k] += alpha;
    ddq_plus = contactDynamics(model,data_fd,q,v,tau_plus,contact_models,contact_data,mu0);
    ddq_partial_dtau_fd.col(k) = (ddq_plus - ddq0)/alpha;
    lambda_partial_dtau_fd.col(k) = (data_fd.lambda_c - lambda0)/alpha;
    tau_plus[k] -= alpha;
  }
  
  BOOST_CHECK(lambda_partial_dtau_fd.isApprox(data.dlambda_dtau,sqrt(alpha)));
  BOOST_CHECK(ddq_partial_dtau_fd.isApprox(data.ddq_dtau,sqrt(alpha)));
}


BOOST_AUTO_TEST_CASE ( test_contact_dynamics_derivatives_LOCAL_WORLD_ALIGNED_6D_fd )
{
  using namespace Eigen;
  using namespace pinocchio;

  Model model;
  buildModels::humanoidRandom(model,true);
  Data data(model), data_fd(model);

  model.lowerPositionLimit.head<3>().fill(-1.);
  model.upperPositionLimit.head<3>().fill( 1.);
  VectorXd q = randomConfiguration(model);

  VectorXd v = VectorXd::Random(model.nv);
  VectorXd tau = VectorXd::Random(model.nv);

  const std::string RF = "rleg6_joint";
  //  const Model::JointIndex RF_id = model.getJointId(RF);
  const std::string LF = "lleg6_joint";
  //  const Model::JointIndex LF_id = model.getJointId(LF);

  // Contact models and data
  PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidContactModel) contact_models;
  PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidContactData) contact_data;

  RigidContactModel ci_LF(CONTACT_6D,model.getJointId(LF),LOCAL_WORLD_ALIGNED);

  contact_models.push_back(ci_LF); contact_data.push_back(RigidContactData(ci_LF));

  Eigen::DenseIndex constraint_dim = 0;
  for(size_t k = 0; k < contact_models.size(); ++k)
    constraint_dim += contact_models[k].size();

  const double mu0 = 0.;

  initContactDynamics(model,data,contact_models);
  contactDynamics(model,data,q,v,tau,contact_models,contact_data,mu0);

  data.M.triangularView<Eigen::StrictlyLower>() = data.M.transpose().triangularView<Eigen::StrictlyLower>();
  computeContactDynamicsDerivatives(model, data, contact_models, contact_data, mu0);

  //Data_fd
  initContactDynamics(model,data_fd,contact_models);

  MatrixXd ddq_partial_dq_fd(model.nv,model.nv); ddq_partial_dq_fd.setZero();
  MatrixXd ddq_partial_dv_fd(model.nv,model.nv); ddq_partial_dv_fd.setZero();
  MatrixXd ddq_partial_dtau_fd(model.nv,model.nv); ddq_partial_dtau_fd.setZero();

  MatrixXd lambda_partial_dtau_fd(constraint_dim,model.nv); lambda_partial_dtau_fd.setZero();
  MatrixXd lambda_partial_dq_fd(constraint_dim,model.nv); lambda_partial_dq_fd.setZero();
  MatrixXd lambda_partial_dv_fd(constraint_dim,model.nv); lambda_partial_dv_fd.setZero();

  const VectorXd ddq0 = contactDynamics(model,data_fd,q,v,tau,contact_models,contact_data,mu0);
  const VectorXd lambda0 = data_fd.lambda_c;
  VectorXd v_eps(VectorXd::Zero(model.nv));
  VectorXd q_plus(model.nq);
  VectorXd ddq_plus(model.nv);

  VectorXd lambda_plus(constraint_dim);

  const double alpha = 1e-8;
  for(int k = 0; k < model.nv; ++k)
  {
    v_eps[k] += alpha;
    q_plus = integrate(model,q,v_eps);
    ddq_plus = contactDynamics(model,data_fd,q_plus,v,tau,contact_models,contact_data,mu0);
    ddq_partial_dq_fd.col(k) = (ddq_plus - ddq0)/alpha;
    lambda_partial_dq_fd.col(k) = (data_fd.lambda_c - lambda0)/alpha;
    v_eps[k] = 0.;
  }

  BOOST_CHECK(ddq_partial_dq_fd.isApprox(data.ddq_dq,sqrt(alpha)));
  BOOST_CHECK(lambda_partial_dq_fd.isApprox(data.dlambda_dq,sqrt(alpha)));

  VectorXd v_plus(v);
  for(int k = 0; k < model.nv; ++k)
  {
    v_plus[k] += alpha;
    ddq_plus = contactDynamics(model,data_fd,q,v_plus,tau,contact_models,contact_data,mu0);
    ddq_partial_dv_fd.col(k) = (ddq_plus - ddq0)/alpha;
    lambda_partial_dv_fd.col(k) = (data_fd.lambda_c - lambda0)/alpha;
    v_plus[k] -= alpha;
  }
  
  BOOST_CHECK(ddq_partial_dv_fd.isApprox(data.ddq_dv,sqrt(alpha)));
  BOOST_CHECK(lambda_partial_dv_fd.isApprox(data.dlambda_dv,sqrt(alpha)));

  VectorXd tau_plus(tau);
  for(int k = 0; k < model.nv; ++k)
  {
    tau_plus[k] += alpha;
    ddq_plus = contactDynamics(model,data_fd,q,v,tau_plus,contact_models,contact_data,mu0);
    ddq_partial_dtau_fd.col(k) = (ddq_plus - ddq0)/alpha;
    lambda_partial_dtau_fd.col(k) = (data_fd.lambda_c - lambda0)/alpha;
    tau_plus[k] -= alpha;
  }
  
  BOOST_CHECK(lambda_partial_dtau_fd.isApprox(data.dlambda_dtau,sqrt(alpha)));
  BOOST_CHECK(ddq_partial_dtau_fd.isApprox(data.ddq_dtau,sqrt(alpha)));
}

BOOST_AUTO_TEST_CASE ( test_contact_dynamics_derivatives_LOCAL_WORLD_ALIGNED_3D_fd )
{
  using namespace Eigen;
  using namespace pinocchio;

  Model model;
  buildModels::humanoidRandom(model,true);
  Data data(model), data_fd(model);

  model.lowerPositionLimit.head<3>().fill(-1.);
  model.upperPositionLimit.head<3>().fill( 1.);
  VectorXd q = randomConfiguration(model);

  VectorXd v = VectorXd::Random(model.nv);
  VectorXd tau = VectorXd::Random(model.nv);

  const std::string RF = "rleg6_joint";
  //  const Model::JointIndex RF_id = model.getJointId(RF);
  const std::string LF = "lleg6_joint";
  //  const Model::JointIndex LF_id = model.getJointId(LF);

  // Contact models and data
  PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidContactModel) contact_models;
  PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidContactData) contact_data;

  RigidContactModel ci_RF(CONTACT_3D,model.getJointId(RF),LOCAL_WORLD_ALIGNED);

  contact_models.push_back(ci_RF); contact_data.push_back(RigidContactData(ci_RF));

  Eigen::DenseIndex constraint_dim = 0;
  for(size_t k = 0; k < contact_models.size(); ++k)
    constraint_dim += contact_models[k].size();

  const double mu0 = 0.;

  initContactDynamics(model,data,contact_models);
  contactDynamics(model,data,q,v,tau,contact_models,contact_data,mu0);

  data.M.triangularView<Eigen::StrictlyLower>() = data.M.transpose().triangularView<Eigen::StrictlyLower>();
  computeContactDynamicsDerivatives(model, data, contact_models, contact_data, mu0);

  //Data_fd
  initContactDynamics(model,data_fd,contact_models);

  MatrixXd ddq_partial_dq_fd(model.nv,model.nv); ddq_partial_dq_fd.setZero();
  MatrixXd ddq_partial_dv_fd(model.nv,model.nv); ddq_partial_dv_fd.setZero();
  MatrixXd ddq_partial_dtau_fd(model.nv,model.nv); ddq_partial_dtau_fd.setZero();

  MatrixXd lambda_partial_dtau_fd(constraint_dim,model.nv); lambda_partial_dtau_fd.setZero();
  MatrixXd lambda_partial_dq_fd(constraint_dim,model.nv); lambda_partial_dq_fd.setZero();
  MatrixXd lambda_partial_dv_fd(constraint_dim,model.nv); lambda_partial_dv_fd.setZero();

  const VectorXd ddq0 = contactDynamics(model,data_fd,q,v,tau,contact_models,contact_data,mu0);
  const VectorXd lambda0 = data_fd.lambda_c;
  VectorXd v_eps(VectorXd::Zero(model.nv));
  VectorXd q_plus(model.nq);
  VectorXd ddq_plus(model.nv);

  VectorXd lambda_plus(constraint_dim);

  const double alpha = 1e-8;
  for(int k = 0; k < model.nv; ++k)
  {
    v_eps[k] += alpha;
    q_plus = integrate(model,q,v_eps);
    ddq_plus = contactDynamics(model,data_fd,q_plus,v,tau,contact_models,contact_data,mu0);
    ddq_partial_dq_fd.col(k) = (ddq_plus - ddq0)/alpha;
    lambda_partial_dq_fd.col(k) = (data_fd.lambda_c - lambda0)/alpha;
    v_eps[k] = 0.;
  }

  BOOST_CHECK(ddq_partial_dq_fd.isApprox(data.ddq_dq,sqrt(alpha)));
  BOOST_CHECK(lambda_partial_dq_fd.isApprox(data.dlambda_dq,sqrt(alpha)));

  VectorXd v_plus(v);
  for(int k = 0; k < model.nv; ++k)
  {
    v_plus[k] += alpha;
    ddq_plus = contactDynamics(model,data_fd,q,v_plus,tau,contact_models,contact_data,mu0);
    ddq_partial_dv_fd.col(k) = (ddq_plus - ddq0)/alpha;
    lambda_partial_dv_fd.col(k) = (data_fd.lambda_c - lambda0)/alpha;
    v_plus[k] -= alpha;
  }
  
  BOOST_CHECK(ddq_partial_dv_fd.isApprox(data.ddq_dv,sqrt(alpha)));
  BOOST_CHECK(lambda_partial_dv_fd.isApprox(data.dlambda_dv,sqrt(alpha)));

  VectorXd tau_plus(tau);
  for(int k = 0; k < model.nv; ++k)
  {
    tau_plus[k] += alpha;
    ddq_plus = contactDynamics(model,data_fd,q,v,tau_plus,contact_models,contact_data,mu0);
    ddq_partial_dtau_fd.col(k) = (ddq_plus - ddq0)/alpha;
    lambda_partial_dtau_fd.col(k) = (data_fd.lambda_c - lambda0)/alpha;
    tau_plus[k] -= alpha;
  }
  
  BOOST_CHECK(lambda_partial_dtau_fd.isApprox(data.dlambda_dtau,sqrt(alpha)));
  BOOST_CHECK(ddq_partial_dtau_fd.isApprox(data.ddq_dtau,sqrt(alpha)));
}


BOOST_AUTO_TEST_CASE(test_contact_dynamics_derivatives_dirty_data)
{
  // Verify that a dirty data doesn't affect the results of the contact dynamics derivs
  using namespace Eigen;
  using namespace pinocchio;

  Model model;
  buildModels::humanoidRandom(model,true);
  Data data_dirty(model);

  model.lowerPositionLimit.head<3>().fill(-1.);
  model.upperPositionLimit.head<3>().fill( 1.);
  VectorXd q = randomConfiguration(model);

  VectorXd v = VectorXd::Random(model.nv);
  VectorXd tau = VectorXd::Random(model.nv);

  const std::string RF = "rleg6_joint";
  //  const Model::JointIndex RF_id = model.getJointId(RF);
  const std::string LF = "lleg6_joint";
  //  const Model::JointIndex LF_id = model.getJointId(LF);

  // Contact models and data
  PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidContactModel) contact_models;
  PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidContactData) contact_data;

  RigidContactModel ci_LF(CONTACT_6D,model.getJointId(LF),LOCAL);
  RigidContactModel ci_RF(CONTACT_3D,model.getJointId(RF),LOCAL);

  contact_models.push_back(ci_LF); contact_data.push_back(RigidContactData(ci_LF));
  contact_models.push_back(ci_RF); contact_data.push_back(RigidContactData(ci_RF));

  Eigen::DenseIndex constraint_dim = 0;
  for(size_t k = 0; k < contact_models.size(); ++k)
    constraint_dim += contact_models[k].size();

  const double mu0 = 0.;

  initContactDynamics(model,data_dirty,contact_models);
  contactDynamics(model,data_dirty,q,v,tau,contact_models,contact_data,mu0);
  computeContactDynamicsDerivatives(model, data_dirty, contact_models, contact_data, mu0);

  // Reuse the same data with new configurations
  q = randomConfiguration(model);
  v = VectorXd::Random(model.nv);
  tau = VectorXd::Random(model.nv);
  contactDynamics(model,data_dirty,q,v,tau,contact_models,contact_data,mu0);
  computeContactDynamicsDerivatives(model, data_dirty, contact_models, contact_data, mu0);

  //Test with fresh data
  Data data_fresh(model);
  initContactDynamics(model,data_fresh,contact_models);
  contactDynamics(model,data_fresh,q,v,tau,contact_models,contact_data,mu0);
  computeContactDynamicsDerivatives(model, data_fresh, contact_models, contact_data, mu0);
  const double alpha = 1e-12;

  BOOST_CHECK(data_dirty.ddq_dq.isApprox(data_fresh.ddq_dq,sqrt(alpha)));
  BOOST_CHECK(data_dirty.ddq_dv.isApprox(data_fresh.ddq_dv,sqrt(alpha)));
  BOOST_CHECK(data_dirty.ddq_dtau.isApprox(data_fresh.ddq_dtau,sqrt(alpha)));
  BOOST_CHECK(data_dirty.dlambda_dq.isApprox(data_fresh.dlambda_dq,sqrt(alpha)));
  BOOST_CHECK(data_dirty.dlambda_dv.isApprox(data_fresh.dlambda_dv,sqrt(alpha)));
  BOOST_CHECK(data_dirty.dlambda_dtau.isApprox(data_fresh.dlambda_dtau,sqrt(alpha)));
}

BOOST_AUTO_TEST_SUITE_END ()
