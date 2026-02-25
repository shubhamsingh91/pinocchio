//
// Copyright (c) 2017-2019 CNRS INRIA
//

#ifndef __pinocchio_mod_rnea_second_order_derivatives_hpp__
#define __pinocchio_mod_rnea_second_order_derivatives_hpp__

#include "pinocchio/multibody/model.hpp"
#include "pinocchio/multibody/data.hpp"

#include "pinocchio/container/aligned-vector.hpp"

namespace pinocchio
{

   /// \brief Computes the second-order partial derivatives of the Recursive Newton Euler Algorithms
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
  /// \param[in] lambda Input vector (dim model.nv).
  /// \param[out] rnea_partial_dqq_mod Second-Order Partial derivative of the generalized torque vector with respect to the joint configuration.
  /// \param[out] rnea_partial_dvv_mod Second-Order Partial derivative of the generalized torque vector with respect to the joint velocity.
  /// \param[out] rnea_partial_dvq_mod Second-Order Partial derivative of the generalized torque vector with respect to the joint vel/config.
  ///
  /// \remarks rnea_partial_dqq_mod, rnea_partial_dvv_mod and rnea_partial_dvq_mod must be first initialized with zeros (rnea_partial_dqq_mod.setZero(),etc).
  ///         As for pinocchio::crba, only the upper triangular part of rnea_partial_dvq_mod is filled.
  ///
  /// \sa pinocchio::rnea
  ///
  // 4-matrix version (includes dqa = ∂M_mod/∂q)
  template<typename Scalar, int Options, template<typename,int> class JointCollectionTpl, typename ConfigVectorType, typename TangentVectorType1, typename TangentVectorType2,
  typename TangentVectorType3, typename MatrixType1, typename MatrixType2, typename MatrixType3, typename MatrixType4>
  inline void
  computeModRNEASecondOrderDerivatives(const ModelTpl<Scalar,Options,JointCollectionTpl> & model,
                         DataTpl<Scalar,Options,JointCollectionTpl> & data,
                         const Eigen::MatrixBase<ConfigVectorType> & q,
                         const Eigen::MatrixBase<TangentVectorType1> & v,
                         const Eigen::MatrixBase<TangentVectorType2> & a,
                         const Eigen::MatrixBase<TangentVectorType3> & lambda,
                         const Eigen::MatrixBase<MatrixType1> & rnea_partial_dqq_mod,
                         const Eigen::MatrixBase<MatrixType2> & rnea_partial_dvv_mod,
                         const Eigen::MatrixBase<MatrixType3> & rnea_partial_dvq_mod,
                         const Eigen::MatrixBase<MatrixType4> & rnea_partial_dqa_mod);

  ///
  /// \brief Computes the derivatives of the Recursive Newton Euler Algorithms
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
  /// \param[in] lambda Input vector (dim model.nv).
  /// \param[in] fext External forces expressed in the local frame of the joints (dim model.njoints).
  /// \param[out] rnea_partial_dqq_mod Partial derivative of the generalized torque vector with respect to the joint configuration.
  /// \param[out] rnea_partial_dvv_mod Partial derivative of the generalized torque vector with respect to the joint velocity.
  /// \param[out] rnea_partial_dvq_mod Partial derivative of the generalized torque vector with respect to the joint acceleration.
  ///
  /// \remarks rnea_partial_dqq_mod, rnea_partial_dvv_mod and rnea_partial_dvq_mod must be first initialized with zeros (rnea_partial_dqq_mod.setZero(),etc).
  ///         As for pinocchio::crba, only the upper triangular part of rnea_partial_dvq_mod is filled.
  ///
  /// \sa pinocchio::rnea
  ///
  // 4-matrix version with fext
  template<typename Scalar, int Options, template<typename,int> class JointCollectionTpl, typename ConfigVectorType, typename TangentVectorType1, typename TangentVectorType2,
  typename TangentVectorType3, typename MatrixType1, typename MatrixType2, typename MatrixType3, typename MatrixType4>
  inline void
  computeModRNEASecondOrderDerivatives(const ModelTpl<Scalar,Options,JointCollectionTpl> & model,
                         DataTpl<Scalar,Options,JointCollectionTpl> & data,
                         const Eigen::MatrixBase<ConfigVectorType> & q,
                         const Eigen::MatrixBase<TangentVectorType1> & v,
                         const Eigen::MatrixBase<TangentVectorType2> & a,
                         const Eigen::MatrixBase<TangentVectorType3> & lambda,
                         const container::aligned_vector< ForceTpl<Scalar,Options> > & fext,
                         const Eigen::MatrixBase<MatrixType1> & rnea_partial_dqq_mod,
                         const Eigen::MatrixBase<MatrixType2> & rnea_partial_dvv_mod,
                         const Eigen::MatrixBase<MatrixType3> & rnea_partial_dvq_mod,
                         const Eigen::MatrixBase<MatrixType4> & rnea_partial_dqa_mod);

} // namespace pinocchio

#include "pinocchio/algorithm/mod-rnea-second-order-derivatives.hxx"

#endif // ifndef __pinocchio_mod_rnea_second_order_derivatives_hpp__
