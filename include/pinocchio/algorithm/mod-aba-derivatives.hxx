//
// Copyright (c) 2017-2020 CNRS INRIA
//

#ifndef __pinocchio_mod_aba_derivatives_hxx__
#define __pinocchio_mod_aba_derivatives_hxx__

#include "pinocchio/multibody/visitor.hpp"
#include "pinocchio/algorithm/check.hpp"
#include "pinocchio/algorithm/aba.hpp"
#include "pinocchio/algorithm/mod-aba-derivatives.hpp"
#include "pinocchio/algorithm/mod-rnea-derivatives.hpp"

namespace pinocchio
{
  

  template<typename Scalar, int Options, template<typename,int> class JointCollectionTpl, typename ConfigVectorType, typename TangentVectorType1, typename TangentVectorType2,
  typename TangentVectorType3, typename VectorType1, typename VectorType2, typename VectorType3>
  inline void
  computeModABADerivatives(const ModelTpl<Scalar,Options,JointCollectionTpl> & model,
                         DataTpl<Scalar,Options,JointCollectionTpl> & data,
                         const Eigen::MatrixBase<ConfigVectorType> & q,
                         const Eigen::MatrixBase<TangentVectorType1> & v,
                         const Eigen::MatrixBase<TangentVectorType2> & tau,
                         const Eigen::MatrixBase<TangentVectorType3> & lambda,
                         const Eigen::MatrixBase<VectorType1> & aba_partial_dq_mod,
                         const Eigen::MatrixBase<VectorType2> & aba_partial_dv_mod,
                         const Eigen::MatrixBase<VectorType3> & aba_partial_dtau_mod)
  {
    PINOCCHIO_CHECK_ARGUMENT_SIZE(q.size(), model.nq, "The joint configuration vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(v.size(), model.nv, "The joint velocity vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(tau.size(), model.nv, "The joint torque vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(lambda.size(), model.nv, "The input vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(aba_partial_dq_mod.rows(), model.nv);
    PINOCCHIO_CHECK_ARGUMENT_SIZE(aba_partial_dv_mod.rows(), model.nv);
    PINOCCHIO_CHECK_ARGUMENT_SIZE(aba_partial_dtau_mod.rows(), model.nv);
    assert(model.check(data) && "data is not consistent with model.");

    VectorType1 & aba_partial_dq_mod_ = PINOCCHIO_EIGEN_CONST_CAST(VectorType1,aba_partial_dq_mod);
    VectorType2 & aba_partial_dv_mod_ = PINOCCHIO_EIGEN_CONST_CAST(VectorType2,aba_partial_dv_mod);
    VectorType3 & aba_partial_dtau_mod_ = PINOCCHIO_EIGEN_CONST_CAST(VectorType3,aba_partial_dtau_mod);
    
    typedef ModelTpl<Scalar,Options,JointCollectionTpl> Model;
    typedef typename Model::JointIndex JointIndex;

    pinocchio::Motion original_grav = model.gravity;
    Model & mutable_model = const_cast<Model &>(model);
    mutable_model.gravity = MotionTpl<double>::Zero();

     Eigen::Matrix<Scalar, Eigen::Dynamic, 1>  mu = aba(mutable_model,data, q, v * 0.0,lambda);

    mutable_model.gravity = original_grav;

     Eigen::Matrix<Scalar, Eigen::Dynamic, 1>  ddq = aba(model, data, q, v, tau);
    
    computeModRNEADerivatives(model, data, q, v, ddq, -mu);

    aba_partial_dq_mod_ = data.dtau_dq_mod;
    aba_partial_dv_mod_ = data.dtau_dv_mod;
    aba_partial_dtau_mod_ = mu;
                              
  }
  
  template<typename Scalar, int Options, template<typename,int> class JointCollectionTpl, typename ConfigVectorType, typename TangentVectorType1, typename TangentVectorType2,
  typename TangentVectorType3, typename VectorType1, typename VectorType2, typename VectorType3>
  inline void
  computeModABADerivatives(const ModelTpl<Scalar,Options,JointCollectionTpl> & model,
                         DataTpl<Scalar,Options,JointCollectionTpl> & data,
                         const Eigen::MatrixBase<ConfigVectorType> & q,
                         const Eigen::MatrixBase<TangentVectorType1> & v,
                         const Eigen::MatrixBase<TangentVectorType2> & tau,
                         const Eigen::MatrixBase<TangentVectorType3> & lambda,
                         const container::aligned_vector< ForceTpl<Scalar,Options> > & fext,
                         const Eigen::MatrixBase<VectorType1> & aba_partial_dq_mod,
                         const Eigen::MatrixBase<VectorType2> & aba_partial_dv_mod,
                         const Eigen::MatrixBase<VectorType3> & aba_partial_dtau_mod)
  {
    PINOCCHIO_CHECK_ARGUMENT_SIZE(q.size(), model.nq, "The joint configuration vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(v.size(), model.nv, "The joint velocity vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(tau.size(), model.nv, "The joint torque vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(lambda.size(), model.nv, "The input vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(fext.size(), (size_t)model.njoints, "The size of the external forces is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(aba_partial_dq_mod.rows(), model.nv);
    PINOCCHIO_CHECK_ARGUMENT_SIZE(aba_partial_dv_mod.rows(), model.nv);
    PINOCCHIO_CHECK_ARGUMENT_SIZE(aba_partial_dtau_mod.rows(), model.nv);
    assert(model.check(data) && "data is not consistent with model.");

    VectorType1 & aba_partial_dq_mod_ = PINOCCHIO_EIGEN_CONST_CAST(VectorType1,aba_partial_dq_mod);
    VectorType2 & aba_partial_dv_mod_ = PINOCCHIO_EIGEN_CONST_CAST(VectorType2,aba_partial_dv_mod);
    VectorType3 & aba_partial_dtau_mod_ = PINOCCHIO_EIGEN_CONST_CAST(VectorType3,aba_partial_dtau_mod);
    
    typedef ModelTpl<Scalar,Options,JointCollectionTpl> Model;    

    pinocchio::Motion original_grav = model.gravity;
    Model & mutable_model = const_cast<Model &>(model);
    mutable_model.gravity = MotionTpl<double>::Zero();

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> mu = aba(mutable_model, data, q, v * 0.0, lambda);

    mutable_model.gravity = original_grav;

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1>  ddq = aba(model, data, q, v, tau, fext);
    
    computeModRNEADerivatives(model, data, q, v, ddq, -mu, fext);

    aba_partial_dq_mod_ = data.dtau_dq_mod;
    aba_partial_dv_mod_ = data.dtau_dv_mod;
    aba_partial_dtau_mod_ = mu;
  }
  

} // namespace pinocchio

#endif // ifndef __pinocchio_mod_aba_derivatives_hxx__
