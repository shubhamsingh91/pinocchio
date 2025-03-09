//
// Copyright (c) 2015-2020 CNRS INRIA
//

#ifndef __pinocchio_algorithm_modrnea_hpp__
#define __pinocchio_algorithm_modrnea_hpp__

#include "pinocchio/multibody/model.hpp"
#include "pinocchio/multibody/data.hpp"
  
namespace pinocchio
{
  ///
  /// \brief The Modified Recursive Newton-Euler algorithm. It computes the inverse dynamics, aka the joint torques according to the current state of the system and the desired joint accelerations.
  ///
  /// \tparam JointCollection Collection of Joint types.
  /// \tparam ConfigVectorType Type of the joint configuration vector.
  /// \tparam TangentVectorType1 Type of the joint velocity vector.
  /// \tparam TangentVectorType2 Type of the joint acceleration vector.
  /// \tparam TangentVectorType3 Type of the input vector.
  ///
  /// \param[in] model The model structure of the rigid body system.
  /// \param[in] data The data structure of the rigid body system.
  /// \param[in] q The joint configuration vector (dim model.nq).
  /// \param[in] v The joint velocity vector (dim model.nv).
  /// \param[in] a The joint acceleration vector (dim model.nv).
  /// \param[in] lambda Input vector (dim model.nv).
  ///  
  /// \return The contracted sum of the joint torques with lambda
  ///
  template<typename Scalar, int Options, template<typename,int> class JointCollectionTpl, typename ConfigVectorType, typename TangentVectorType1, typename TangentVectorType2, typename TangentVectorType3>
  inline const Scalar&
  modrnea(const ModelTpl<Scalar,Options,JointCollectionTpl> & model,
       DataTpl<Scalar,Options,JointCollectionTpl> & data,
       const Eigen::MatrixBase<ConfigVectorType> & q,
       const Eigen::MatrixBase<TangentVectorType1> & v,
       const Eigen::MatrixBase<TangentVectorType2> & a,
       const Eigen::MatrixBase<TangentVectorType3> & lambda);
  
//   ///
//   /// \brief The Recursive Newton-Euler algorithm. It computes the inverse dynamics, aka the joint torques according to the current state of the system, the desired joint accelerations and the external forces.
//   ///
//   /// \tparam JointCollection Collection of Joint types.
//   /// \tparam ConfigVectorType Type of the joint configuration vector.
//   /// \tparam TangentVectorType1 Type of the joint velocity vector.
//   /// \tparam TangentVectorType2 Type of the joint acceleration vector.
//   /// \tparam ForceDerived Type of the external forces.
//   ///
//   /// \param[in] model The model structure of the rigid body system.
//   /// \param[in] data The data structure of the rigid body system.
//   /// \param[in] q The joint configuration vector (dim model.nq).
//   /// \param[in] v The joint velocity vector (dim model.nv).
//   /// \param[in] a The joint acceleration vector (dim model.nv).
//   /// \param[in] fext Vector of external forces expressed in the local frame of the joints (dim model.njoints)
//   ///
//   /// \return The desired joint torques stored in data.tau.
//   ///
//   template<typename Scalar, int Options, template<typename,int> class JointCollectionTpl, typename ConfigVectorType, typename TangentVectorType1, typename TangentVectorType2, typename ForceDerived>
//   inline const typename DataTpl<Scalar,Options,JointCollectionTpl>::TangentVectorType &
//   rnea(const ModelTpl<Scalar,Options,JointCollectionTpl> & model,
//        DataTpl<Scalar,Options,JointCollectionTpl> & data,
//        const Eigen::MatrixBase<ConfigVectorType> & q,
//        const Eigen::MatrixBase<TangentVectorType1> & v,
//        const Eigen::MatrixBase<TangentVectorType2> & a,
//        const container::aligned_vector<ForceDerived> & fext);
  
  /// 
} // namespace pinocchio 

/* --- Details -------------------------------------------------------------------- */
#include "pinocchio/algorithm/modrnea.hxx"

#endif // ifndef __pinocchio_algorithm_modrnea_hpp__
