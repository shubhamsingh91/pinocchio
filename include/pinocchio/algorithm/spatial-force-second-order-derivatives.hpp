//
// Copyright (c) 2017-2019 CNRS INRIA

#ifndef __pinocchio_spatial_force_second_order_derivatives_hpp__
#define __pinocchio_spatial_force_second_order_derivatives_hpp__

#include "pinocchio/container/aligned-vector.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"

namespace pinocchio {
///
/// \brief Computes the Second-Order partial derivatives of the Cumulative Spatial force
/// w.r.t the joint configuration, the joint velocity and the
/// joint acceleration.
///
/// \tparam JointCollection Collection of Joint types.
/// \tparam ConfigVectorType Type of the joint configuration vector.
/// \tparam TangentVectorType1 Type of the joint velocity vector.
/// \tparam TangentVectorType2 Type of the joint acceleration vector.
/// \tparam std::vector<Tensor1> Type of the 3D-Tensor containing the SO partial
/// derivative with respect to the joint configuration vector. The elements of
/// cumulative spatial force vector for ith joint are along the 1st dim, and joint config along 2nd,3rd
/// dimensions.
/// \tparam std::vector<Tensor2> Type of the 3D-Tensor containing the
/// Second-Order partial derivative with respect to the joint velocity vector.
/// The elements of cumulative spatial force vector vector are along the 1st dim, and the velocity
/// along 2nd,3rd dimensions.
/// \tparam std::vector<Tensor3> Type of the 3D-Tensor
/// containing the cross Second-Order partial derivative with respect to the
/// joint configuration and velocty vector. The elements of cumulative spatial force vector are
/// along the 1st dim, and the config. vector along 3nd dimension, and velocity
/// along the second dimension.
///\tparam Tensor4 Type of the 3D-Tensor containing the cross Second-Order
/// partial derivative with respect to the joint configuration and acceleration
/// vector. The elements of cumulative spatial force vector are
/// along the 1st dim, and the acceleration vector along 2nd dimension, while
/// configuration along the third dimension.
///
/// \param[in] model The model structure of the rigid body system.
/// \param[in] data The data structure of the rigid body system.
/// \param[in] q The joint configuration vector (dim model.nq).
/// \param[in] v The joint velocity vector (dim model.nv).
/// \param[in] a The joint acceleration vector (dim model.nv).
/// \param[out] d2fc_dqdq Second-Order Partial derivative of the generalized
/// cumulative spatial force vector with respect to the joint configuration.
/// \param[out] d2fc_dvdv Second-Order Partial derivative of the generalized
/// cumulative spatial force vector with respect to the joint velocity
/// \param[out] d2fc_dvdq Cross Second-Order Partial derivative of the
/// generalized cumulative spatial force vector with respect to the joint configuration and
/// velocity.
/// \param[out] d2fc_dadq Cross Second-Order Partial derivative of
/// the generalized cumulative spatial force vector with respect to the joint configuration and
/// accleration.
/// \remarks 

///
template <typename Scalar, int Options,
          template <typename, int> class JointCollectionTpl,
          typename ConfigVectorType, typename TangentVectorType1,
          typename TangentVectorType2, typename Tensor1,
          typename Tensor2, typename Tensor3, typename Tensor4>
inline void ComputeSpatialForceSecondOrderDerivatives(
    const ModelTpl<Scalar, Options, JointCollectionTpl> &model,
    DataTpl<Scalar, Options, JointCollectionTpl> &data,
    const Eigen::MatrixBase<ConfigVectorType> &q,
    const Eigen::MatrixBase<TangentVectorType1> &v,
    const Eigen::MatrixBase<TangentVectorType2> &a,
    const std::vector<Tensor1> &d2fc_dqdq, const std::vector<Tensor2> &d2fc_dvdv,
    const std::vector<Tensor3> &d2fc_dvdq, const std::vector<Tensor4> &d2fc_dadq);

///

} // namespace pinocchio

#include "pinocchio/algorithm/spatial-force-second-order-derivatives.hxx"

#endif // ifndef __pinocchio_algorithm_rnea_second_order_derivatives_hpp__
