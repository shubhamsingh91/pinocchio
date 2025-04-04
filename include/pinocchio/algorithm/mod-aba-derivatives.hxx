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
  

//   template<typename Scalar, int Options, template<typename,int> class JointCollectionTpl, typename ConfigVectorType, typename TangentVectorType1, typename TangentVectorType2, typename TangentVectorType3>
//   struct ComputeModRNEADerivativesForwardStep
//   : public fusion::JointUnaryVisitorBase< ComputeModRNEADerivativesForwardStep<Scalar,Options,JointCollectionTpl,ConfigVectorType,TangentVectorType1,TangentVectorType2,TangentVectorType3> >
//   {
//     typedef ModelTpl<Scalar,Options,JointCollectionTpl> Model;
//     typedef DataTpl<Scalar,Options,JointCollectionTpl> Data;
    
//     typedef boost::fusion::vector<const Model &,
//                                   Data &,
//                                   const ConfigVectorType &,
//                                   const TangentVectorType1 &,
//                                   const TangentVectorType2 &,
//                                   const TangentVectorType3 &
//                                   > ArgsType;
    
//     template<typename JointModel>
//     static void algo(const JointModelBase<JointModel> & jmodel,
//                      JointDataBase<typename JointModel::JointDataDerived> & jdata,
//                      const Model & model,
//                      Data & data,
//                      const Eigen::MatrixBase<ConfigVectorType> & q,
//                      const Eigen::MatrixBase<TangentVectorType1> & v,
//                      const Eigen::MatrixBase<TangentVectorType2> & a,
//                      const Eigen::MatrixBase<TangentVectorType3> & lambda)
//     {
//         typedef typename Model::JointIndex JointIndex;
//         typedef typename Data::Motion Motion;
//         typedef typename Data::Inertia Inertia;
//         typedef typename Data::Coriolis Coriolis;

//         const JointIndex& i = jmodel.id();
//         const JointIndex& parent = model.parents[i];
//         Motion& ov = data.ov[i];
//         Motion& oa = data.oa[i];
//         Motion& ow = data.ow[i];
//         Motion& vJ = data.vJ[i];
//         Motion& wJ = data.wJ[i];

//         jmodel.calc(jdata.derived(), q.derived(), v.derived());

//         data.liMi[i] = model.jointPlacements[i] * jdata.M();

//         if (parent > 0) {
//             data.oMi[i] = data.oMi[parent] * data.liMi[i];
//             ov = data.ov[parent];
//             oa = data.oa[parent];
//             ow = data.ow[parent];
//         } else {
//             data.oMi[i] = data.liMi[i];
//             ov.setZero();
//             ow.setZero();
//             oa = -model.gravity;
//         }

//         typedef typename SizeDepType<JointModel::NV>::template ColsReturn<typename Data::Matrix6x>::Type ColsBlock;
//         ColsBlock J_cols = jmodel.jointCols(data.J);
//         ColsBlock dJ_cols = jmodel.jointCols(data.dJ);
//         ColsBlock ddJ_cols = jmodel.jointCols(data.ddJ);
//         ColsBlock vdJ_cols = jmodel.jointCols(data.vdJ);

//         // J and vJ
//         J_cols.noalias() = data.oMi[i].act(jdata.S());
//         vJ = data.oMi[i].act(jdata.v());
//         wJ = data.oMi[i].act(jdata.S() * jmodel.jointVelocitySelector(lambda));

//         // dJ
//         motionSet::motionAction(ov, J_cols, dJ_cols);

//         // ddJ
//         motionSet::motionAction(oa, J_cols, ddJ_cols);
//         motionSet::motionAction<ADDTO>(ov, dJ_cols, ddJ_cols);

//         // vdJ
//         motionSet::motionAction(vJ, J_cols, vdJ_cols);
//         vdJ_cols.noalias() += dJ_cols + dJ_cols;

//         // velocity and accelaration finishing
//         ov += vJ;
//         oa += (ov ^ vJ) + data.oMi[i].act(jdata.S() * jmodel.jointVelocitySelector(a) + jdata.c());
//         ow += wJ;

//         // Composite rigid body inertia
//         Inertia& oY = data.oYcrb[i];

//         oY = data.oMi[i].act(model.inertias[i]); // 0_IC_i
//         data.of[i] = oY * oa + oY.vxiv(ov); // 0_f_i
//         data.oh_lam[i]  = oY * ow; // 0_h_i

//         data.oBcrb[i] = Coriolis(oY, ov); // 0_BC_i
//         data.oz[i] = data.oBcrb[i].matrix().transpose() * data.ow[i].toVector(); // 0_z_i
//     }
    
//     template<typename ForceDerived, typename M6>
//     static void addForceCrossMatrix(const ForceDense<ForceDerived> & f,
//                                     const Eigen::MatrixBase<M6> & mout)
//     {
//       M6 & mout_ = PINOCCHIO_EIGEN_CONST_CAST(M6,mout);
//       addSkew(-f.linear(),mout_.template block<3,3>(ForceDerived::LINEAR,ForceDerived::ANGULAR));
//       addSkew(-f.linear(),mout_.template block<3,3>(ForceDerived::ANGULAR,ForceDerived::LINEAR));
//       addSkew(-f.angular(),mout_.template block<3,3>(ForceDerived::ANGULAR,ForceDerived::ANGULAR));
//     }
    
//   };
  
//   template<typename Scalar, int Options, template<typename,int> class JointCollectionTpl, typename VectorType1, typename VectorType2, typename VectorType3>
//   struct ComputeModRNEADerivativesBackwardStep
//   : public fusion::JointUnaryVisitorBase<ComputeModRNEADerivativesBackwardStep<Scalar,Options,JointCollectionTpl,VectorType1,VectorType2,VectorType3> >
//   {
//     typedef ModelTpl<Scalar,Options,JointCollectionTpl> Model;
//     typedef DataTpl<Scalar,Options,JointCollectionTpl> Data;
    
//     typedef boost::fusion::vector<const Model &,
//                                   Data &,
//                                   const VectorType1 &,
//                                   const VectorType2 &,
//                                   const VectorType3 &
//                                   > ArgsType;
    
//     template<typename JointModel>
//     static void algo(const JointModelBase<JointModel> & jmodel,
//                      const Model & model,
//                      Data & data,
//                      const Eigen::MatrixBase<VectorType1> & aba_partial_dq_mod,
//                      const Eigen::MatrixBase<VectorType2> & aba_partial_dv_mod,
//                      const Eigen::MatrixBase<VectorType3> & aba_partial_dtau_mod)
//     {
//       typedef typename Model::JointIndex JointIndex;
        
//         const JointIndex& i = jmodel.id();
//         const JointIndex& parent = model.parents[i];

//         VectorType1& rnea_partial_dq_mod_ = PINOCCHIO_EIGEN_CONST_CAST(VectorType1,aba_partial_dq_mod);
//         VectorType2& rnea_partial_dv_mod_ = PINOCCHIO_EIGEN_CONST_CAST(VectorType2,aba_partial_dv_mod);
//         VectorType3& rnea_partial_da_mod_ = PINOCCHIO_EIGEN_CONST_CAST(VectorType3,aba_partial_dtau_mod);

//         typedef typename SizeDepType<JointModel::NV>::template ColsReturn<typename Data::Matrix6x>::Type ColsBlock;

//         ColsBlock J_cols = jmodel.jointCols(data.J);
//         ColsBlock dJ_cols = jmodel.jointCols(data.dJ);
//         ColsBlock ddJ_cols = jmodel.jointCols(data.ddJ);
//         ColsBlock vdJ_cols = jmodel.jointCols(data.vdJ);

//         ColsBlock tmp3 = jmodel.jointCols(data.Ftmp3); // tmp3 is for this joint only, Ftmp3 is for the full body

//         const Eigen::Index joint_idx = (Eigen::Index)jmodel.idx_v();
//         const Eigen::Index joint_dofs = (Eigen::Index)jmodel.nv();

//         motionSet::act(J_cols, data.of[i], tmp3); // S{i} x* f{i}

//         rnea_partial_dq_mod_.segment(joint_idx, joint_dofs).noalias()
//           = tmp3.transpose() * data.ow[parent].toVector() + 
//              dJ_cols.transpose() * data.oz[i].toVector() + 
//              ddJ_cols.transpose() * data.oh_lam[i].toVector();

//         rnea_partial_dv_mod_.segment(joint_idx, joint_dofs).noalias()
//           = vdJ_cols.transpose() * data.oh_lam[i].toVector()
//           + J_cols.transpose() * data.oz[i].toVector();

//         rnea_partial_da_mod_.segment(joint_idx, joint_dofs).noalias()
//           = J_cols.transpose() * data.oh_lam[i].toVector();  

//         if (parent > 0) {
//             data.oz[parent] += data.oz[i];
//             data.oh_lam[parent] += data.oh_lam[i];
//             data.of[parent] += data.of[i];
//         }
    

//     }
    
//     template<typename Min, typename Mout>
//     static void lhsInertiaMult(const typename Data::Inertia & Y,
//                                const Eigen::MatrixBase<Min> & J,
//                                const Eigen::MatrixBase<Mout> & F)
//     {
//       Mout & F_ = PINOCCHIO_EIGEN_CONST_CAST(Mout,F);
//       motionSet::inertiaAction(Y,J.derived().transpose(),F_.transpose());
//     }
//   };
  
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
    
    typedef ModelTpl<Scalar,Options,JointCollectionTpl> Model;
    typedef typename Model::JointIndex JointIndex;

    Motion original_gravity = model.gravity;
    Model & mutable_model = const_cast<Model &>(model); // mutable_model is a reference to the original model
    mutable_model.gravity = MotionTpl<Scalar, Options>::Zero();

    Eigen::VectorXd ddq = aba(mutable_model, data, q.derived(), v.derived(), tau.derived());
    
    mutable_model.gravity = original_gravity;

    auto mu = aba(model, data, q.derived(), v.derived()*0.0, tau.derived()*0.0);

    // computeModRNEADerivatives(model, data, q.derived(), v.derived(), data.ddq.derived(), (-mu).eval());

    // aba_partial_dq_mod = data.dtau_dq_mod;
    // aba_partial_dv_mod = data.dtau_dv_mod;
    // aba_partial_dtau_mod = mu.derived();
                              
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
    
    typedef ModelTpl<Scalar,Options,JointCollectionTpl> Model;
    typedef typename Model::JointIndex JointIndex;
    
    
    // typedef ComputeModRNEADerivativesForwardStep<Scalar,Options,JointCollectionTpl,ConfigVectorType,TangentVectorType1,TangentVectorType2,TangentVectorType3> Pass1;
    // for(JointIndex i=1; i<(JointIndex) model.njoints; ++i)
    // {
    //   Pass1::run(model.joints[i],data.joints[i],
    //              typename Pass1::ArgsType(model,data,q.derived(),v.derived(),a.derived(),lambda.derived()));
    //   data.of[i] -= data.oMi[i].act(fext[i]);
    // }
    
    // typedef ComputeModRNEADerivativesBackwardStep<Scalar,Options,JointCollectionTpl,VectorType1,VectorType2,VectorType3> Pass2;
    // for(JointIndex i=(JointIndex)(model.njoints-1); i>0; --i)
    // {
    //   Pass2::run(model.joints[i],
    //              typename Pass2::ArgsType(model,data,
    //                                       PINOCCHIO_EIGEN_CONST_CAST(VectorType1,aba_partial_dq_mod),
    //                                       PINOCCHIO_EIGEN_CONST_CAST(VectorType2,aba_partial_dv_mod),
    //                                       PINOCCHIO_EIGEN_CONST_CAST(VectorType3,aba_partial_dtau_mod)));
    // }
  }
  

} // namespace pinocchio

#endif // ifndef __pinocchio_mod_aba_derivatives_hxx__
