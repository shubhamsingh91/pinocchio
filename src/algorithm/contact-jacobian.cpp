//
// Copyright (c) 2022 INRIA
//

#include "pinocchio/algorithm/contact-jacobian.hpp"

namespace pinocchio {
  template void getConstraintJacobian 
    <context::Scalar, context::Options, JointCollectionDefaultTpl, context::MatrixXs>
  (const context::Model &, const context::Data &, const context::RigidConstraintModel &, context::RigidConstraintData &, const Eigen::MatrixBase<context::MatrixXs> &);
 
  template void getConstraintsJacobian 
    <context::Scalar, context::Options, JointCollectionDefaultTpl, context::MatrixXs, typename context::RigidConstraintModelVector::allocator_type, typename context::RigidConstraintDataVector::allocator_type>
  (const context::Model &, const context::Data &, const context::RigidConstraintModelVector &, context::RigidConstraintDataVector &, const Eigen::MatrixBase<context::MatrixXs> &);

} // namespace pinocchio
