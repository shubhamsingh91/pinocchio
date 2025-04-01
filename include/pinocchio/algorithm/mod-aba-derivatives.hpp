//
// Copyright (c) 2017-2019 CNRS INRIA
//

#ifndef __pinocchio_mod_aba_derivatives_hpp__
#define __pinocchio_mod_aba_derivatives_hpp__

#include "pinocchio/multibody/model.hpp"
#include "pinocchio/multibody/data.hpp"

#include "pinocchio/container/aligned-vector.hpp"

namespace pinocchio
{

   /// \brief Computes the partial derivatives of the ABA Algorithms
  ///        with respect to the joint configuration, the joint velocity and the joint torque.
  ///
  /// \tparam JointCollection Collection of Joint types.
  /// \tparam ConfigVectorType Type of the joint configuration vector.
  /// \tparam TangentVectorType1 Type of the joint velocity vector.
  /// \tparam TangentVectorType2 Type of the joint acceleration vector.
  /// \tparam TangentVectorType3 Type of the lambda vector.
  /// \tparam VectorType1 Type of the matrix containing the partial derivative with respect to the joint configuration vector.
  /// \tparam VectorType2 Type of the matrix containing the partial derivative with respect to the joint velocity vector.
  /// \tparam VectorType3 Type of the matrix containing the partial derivative with respect to the joint acceleration vector.
  ///
  /// \param[in] model The model structure of the rigid body system.
  /// \param[in] data The data structure of the rigid body system.
  /// \param[in] q The joint configuration vector (dim model.nq).
  /// \param[in] v The joint velocity vector (dim model.nv).
  /// \param[in] tau The joint acceleration vector (dim model.nv).
  /// \param[in] lambda Input vector (dim model.nv).
  /// \param[out] aba_partial_dq_mod Partial derivative of the generalized acceleration vector with respect to the joint configuration.
  /// \param[out] aba_partial_dv_mod Partial derivative of the generalized acceleration vector with respect to the joint velocity.
  /// \param[out] aba_partial_dtau_mod Partial derivative of the generalized acceleration vector with respect to the joint torque.
  ///
  /// \remarks aba_partial_dq_mod, aba_partial_dv_mod and aba_partial_dtau_mod must be first initialized with zeros (aba_partial_dq_mod.setZero(),etc).
  ///         As for pinocchio::crba, only the upper triangular part of aba_partial_dtau_mod is filled.
  ///
  ///
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
                         const Eigen::MatrixBase<VectorType3> & aba_partial_dtau_mod);
  
  ///
  /// \brief Computes the derivatives of the ABA
  ///        with respect to the joint configuration, the joint velocity and the joint acceleration.
  ///
  /// \tparam JointCollection Collection of Joint types.
  /// \tparam ConfigVectorType Type of the joint configuration vector.
  /// \tparam TangentVectorType1 Type of the joint velocity vector.
  /// \tparam TangentVectorType2 Type of the joint acceleration vector.
  /// \tparam TangentVectorType3 Type of the lambda vector.
  /// \tparam ExternalForcesType Type of the external forces vector.
  /// \tparam VectorType1 Type of the matrix containing the partial derivative with respect to the joint configuration vector.
  /// \tparam VectorType2 Type of the matrix containing the partial derivative with respect to the joint velocity vector.
  /// \tparam VectorType3 Type of the matrix containing the partial derivative with respect to the joint acceleration vector.
  ///
  /// \param[in] model The model structure of the rigid body system.
  /// \param[in] data The data structure of the rigid body system.
  /// \param[in] q The joint configuration vector (dim model.nq).
  /// \param[in] v The joint velocity vector (dim model.nv).
  /// \param[in] tau The joint acceleration vector (dim model.nv).
  /// \param[in] lambda Input vector (dim model.nv).
  /// \param[in] fext External forces expressed in the local frame of the joints (dim model.njoints).
  /// \param[out] aba_partial_dq_mod Partial derivative of the generalized torque vector with respect to the joint configuration.
  /// \param[out] aba_partial_dv_mod Partial derivative of the generalized torque vector with respect to the joint velocity.
  /// \param[out] aba_partial_dtau_mod Partial derivative of the generalized torque vector with respect to the joint acceleration.
  ///
  /// \remarks aba_partial_dq_mod, aba_partial_dv_mod and aba_partial_dtau_mod must be first initialized with zeros (aba_partial_dq_mod.setZero(),etc).
  ///         As for pinocchio::crba, only the upper triangular part of aba_partial_dtau_mod is filled.
  ///
  /// \sa pinocchio::rnea
  ///
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
                         const Eigen::MatrixBase<VectorType3> & aba_partial_dtau_mod);
  
  ///
  /// \brief Computes the derivatives of the Recursive Newton Euler Algorithms
  ///        with respect to the joint configuration, the joint velocity and the joint acceleration.
  ///
  /// \tparam JointCollection Collection of Joint types.
  /// \tparam ConfigVectorType Type of the joint configuration vector.
  /// \tparam TangentVectorType1 Type of the joint velocity vector.
  /// \tparam TangentVectorType2 Type of the joint acceleration vector.
  ///
  /// \param[in] model The model structure of the rigid body system.
  /// \param[in] data The data structure of the rigid body system.
  /// \param[in] q The joint configuration vector (dim model.nq).
  /// \param[in] v The joint velocity vector (dim model.nv).
  /// \param[in] tau The joint acceleration vector (dim model.nv).
  /// \param[in] lambda Input vector (dim model.nv).
  ///
  ///  
  template<typename Scalar, int Options, template<typename,int> class JointCollectionTpl, typename ConfigVectorType, typename TangentVectorType1, typename TangentVectorType2, typename TangentVectorType3>
  inline void
  computeModABADerivatives(const ModelTpl<Scalar,Options,JointCollectionTpl> & model,
                         DataTpl<Scalar,Options,JointCollectionTpl> & data,
                         const Eigen::MatrixBase<ConfigVectorType> & q,
                         const Eigen::MatrixBase<TangentVectorType1> & v,
                         const Eigen::MatrixBase<TangentVectorType2> & tau,
                         const Eigen::MatrixBase<TangentVectorType3> & lambda)
  {
    computeModABADerivatives(model,data,q.derived(),v.derived(),tau.derived(),lambda.derived(),
                           data.ddq_dq_mod, data.ddq_dv_mod, data.ddq_dtau_mod);
  }
  
  ///
  /// \brief Computes the derivatives of the Recursive Newton Euler Algorithms
  ///        with respect to the joint configuration, the joint velocity and the joint acceleration.
  ///
  /// \tparam JointCollection Collection of Joint types.
  /// \tparam ConfigVectorType Type of the joint configuration vector.
  /// \tparam TangentVectorType1 Type of the joint velocity vector.
  /// \tparam TangentVectorType2 Type of the joint acceleration vector.
  ///
  /// \param[in] model The model structure of the rigid body system.
  /// \param[in] data The data structure of the rigid body system.
  /// \param[in] q The joint configuration vector (dim model.nq).
  /// \param[in] v The joint velocity vector (dim model.nv).
  /// \param[in] tau The joint acceleration vector (dim model.nv).
  /// \param[in] lambda Input vector (dim model.nv).
  /// \param[in] fext External forces expressed in the local frame of the joints (dim model.njoints).
  ///
  /// \returns The results are stored in data.dtau_dq_mod, data.dtau_dv_mod and data.M_mod which respectively correspond
  ///          to the partial derivatives of the joint torque vector with respect to the joint configuration, velocity and acceleration.
  ///          As for pinocchio::crba, only the upper triangular part of data.M_mod is filled.
  ///
  /// \sa pinocchio::rnea, pinocchio::crba, pinocchio::cholesky::decompose
  ///
  template<typename Scalar, int Options, template<typename,int> class JointCollectionTpl, typename ConfigVectorType, typename TangentVectorType1, typename TangentVectorType2, typename TangentVectorType3>
  inline void
  computeModABADerivatives(const ModelTpl<Scalar,Options,JointCollectionTpl> & model,
                         DataTpl<Scalar,Options,JointCollectionTpl> & data,
                         const Eigen::MatrixBase<ConfigVectorType> & q,
                         const Eigen::MatrixBase<TangentVectorType1> & v,
                         const Eigen::MatrixBase<TangentVectorType2> & tau,
                         const Eigen::MatrixBase<TangentVectorType3> & lambda,
                         const container::aligned_vector< ForceTpl<Scalar,Options> > & fext)
  {
    computeModABADerivatives(model,data,q.derived(),v.derived(),tau.derived(),lambda.derived(),fext,
                           data.dtau_dq_mod, data.dtau_dv_mod, data.M_mod);
  }


} // namespace pinocchio 

#include "pinocchio/algorithm/mod-aba-derivatives.hxx"

#endif // ifndef __pinocchio_mod_aba_derivatives_hpp__
