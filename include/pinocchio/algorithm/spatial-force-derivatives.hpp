//
// Copyright (c) 2017-2019 CNRS INRIA
//

#ifndef __pinocchio_spatial_force_derivatives_hpp__
#define __pinocchio_spatial_force_derivatives_hpp__

#include "pinocchio/multibody/model.hpp"
#include "pinocchio/multibody/data.hpp"

#include "pinocchio/container/aligned-vector.hpp"

namespace pinocchio
{
  ///
  /// \brief Computes the partial derivatives of the cumulative spatial forces
  ///        with respect to the joint configuration, the joint velocity and the joint acceleration.
  ///
  /// \tparam JointCollection Collection of Joint types.
  /// \tparam ConfigVectorType Type of the joint configuration vector.
  /// \tparam TangentVectorType1 Type of the joint velocity vector.
  /// \tparam TangentVectorType2 Type of the joint acceleration vector.
  /// \tparam MatrixType1 Type of the matrix containing the partial derivative with respect to the joint configuration vector.
  /// \tparam MatrixType2 Type of the matrix containing the partial derivative with respect to the joint velocity vector.
  /// \tparam MatrixType3 Type of the matrix containing the partial derivative with respect to the joint acceleration vector.
  ///
  /// \param[in] model The model structure of the rigid body system.
  /// \param[in] data The data structure of the rigid body system.
  /// \param[in] q The joint configuration vector (dim model.nq).
  /// \param[in] v The joint velocity vector (dim model.nv).
  /// \param[in] a The joint acceleration vector (dim model.nv).
  /// \param[out] df_dq Partial derivative of the generalized torque vector with respect to the joint configuration.
  ///

  /// \sa pinocchio::rnea
  ///
  template<typename Scalar, int Options, template<typename,int> class JointCollectionTpl, typename ConfigVectorType, typename TangentVectorType1, typename TangentVectorType2>
  inline void
  computeSpatialForceDerivs(const ModelTpl<Scalar,Options,JointCollectionTpl> & model,
                         DataTpl<Scalar,Options,JointCollectionTpl> & data,
                         const Eigen::MatrixBase<ConfigVectorType> & q,
                         const Eigen::MatrixBase<TangentVectorType1> & v,
                         const Eigen::MatrixBase<TangentVectorType2> & a,
                         Eigen::Tensor<double,3> & df_dq);
  
  ///
} // namespace pinocchio 

#include "pinocchio/algorithm/spatial-force-derivatives.hxx"

#endif // ifndef __pinocchio_rnea_derivatives_hpp__
