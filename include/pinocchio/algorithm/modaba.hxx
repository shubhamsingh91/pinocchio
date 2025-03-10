//
// Copyright (c) 2016-2020 CNRS, INRIA
//

#ifndef __pinocchio_modaba_hxx__
#define __pinocchio_modaba_hxx__

#include "pinocchio/spatial/act-on-set.hpp"
#include "pinocchio/multibody/visitor.hpp"
#include "pinocchio/algorithm/check.hpp"
#include "pinocchio/algorithm/aba.hpp"

/// @cond DEV

namespace pinocchio
{

  template<typename Scalar, int Options, template<typename,int> class JointCollectionTpl, typename ConfigVectorType, typename TangentVectorType1, typename TangentVectorType2, typename TangentVectorType3>
  inline Scalar
  modaba(const ModelTpl<Scalar,Options,JointCollectionTpl> & model,
      DataTpl<Scalar,Options,JointCollectionTpl> & data,
      const Eigen::MatrixBase<ConfigVectorType> & q,
      const Eigen::MatrixBase<TangentVectorType1> & v,
      const Eigen::MatrixBase<TangentVectorType2> & tau,
      const Eigen::MatrixBase<TangentVectorType3> & mu)
  {
    assert(model.check(data) && "data is not consistent with model.");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(q.size(), model.nq, "The joint configuration vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(v.size(), model.nv, "The joint velocity vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(tau.size(), model.nv, "The joint torque vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(mu.size(), model.nv, "mu vector is not of right size");

    data.ddq = aba(model,data,q,v,tau);

    return mu.transpose() * data.ddq;
  }

  template<typename Scalar, int Options, template<typename,int> class JointCollectionTpl, typename ConfigVectorType, typename TangentVectorType1, typename TangentVectorType2, typename TangentVectorType3, typename ForceDerived>
  inline Scalar 
  modaba(const ModelTpl<Scalar,Options,JointCollectionTpl> & model,
      DataTpl<Scalar,Options,JointCollectionTpl> & data,
      const Eigen::MatrixBase<ConfigVectorType> & q,
      const Eigen::MatrixBase<TangentVectorType1> & v,
      const Eigen::MatrixBase<TangentVectorType2> & tau,
      const Eigen::MatrixBase<TangentVectorType3> & mu,
      const container::aligned_vector<ForceDerived> & fext)

  {
    assert(model.check(data) && "data is not consistent with model.");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(q.size(), model.nq, "The joint configuration vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(v.size(), model.nv, "The joint velocity vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(tau.size(), model.nv, "The joint torque vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(mu.size(), model.nv, "mu vector is not of right size");

    data.ddq = aba(model,data,q,v,tau,fext);

    return mu.transpose() * data.ddq;
  }



} // namespace pinocchio

/// @endcond

#endif // ifndef __pinocchio_modaba_hxx__
