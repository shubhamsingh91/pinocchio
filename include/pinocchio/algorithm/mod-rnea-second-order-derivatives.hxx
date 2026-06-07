//
// Copyright (c) 2017-2020 CNRS INRIA
//

#ifndef __pinocchio_mod_rnea_second_order_derivatives_hxx__
#define __pinocchio_mod_rnea_second_order_derivatives_hxx__

#include "pinocchio/multibody/visitor.hpp"
#include "pinocchio/algorithm/check.hpp"
#include "pinocchio/algorithm/mod-rnea-second-order-derivatives.hpp"

namespace pinocchio
{
  

  template<typename Scalar, int Options, template<typename,int> class JointCollectionTpl, typename ConfigVectorType, typename TangentVectorType1, typename TangentVectorType2, typename TangentVectorType3>
  struct computeModRNEASecondOrderDerivativesForwardStep
  : public fusion::JointUnaryVisitorBase< computeModRNEASecondOrderDerivativesForwardStep<Scalar,Options,JointCollectionTpl,ConfigVectorType,TangentVectorType1,TangentVectorType2,TangentVectorType3> >
  {
    typedef ModelTpl<Scalar,Options,JointCollectionTpl> Model;
    typedef DataTpl<Scalar,Options,JointCollectionTpl> Data;
    
    typedef boost::fusion::vector<const Model &,
                                  Data &,
                                  const ConfigVectorType &,
                                  const TangentVectorType1 &,
                                  const TangentVectorType2 &,
                                  const TangentVectorType3 &
                                  > ArgsType;
    
    template<typename JointModel>
    static void algo(const JointModelBase<JointModel> & jmodel,
                     JointDataBase<typename JointModel::JointDataDerived> & jdata,
                     const Model & model,
                     Data & data,
                     const Eigen::MatrixBase<ConfigVectorType> & q,
                     const Eigen::MatrixBase<TangentVectorType1> & v,
                     const Eigen::MatrixBase<TangentVectorType2> & a,
                     const Eigen::MatrixBase<TangentVectorType3> & lambda)
    {
        typedef typename Model::JointIndex JointIndex;
        typedef typename Data::Motion Motion;
        typedef typename Data::Inertia Inertia;
        typedef typename Data::Coriolis Coriolis;

        const JointIndex& i = jmodel.id();
        const JointIndex& parent = model.parents[i];
        Motion& ov = data.ov[i];
        Motion& oa = data.oa[i];
        Motion& ow = data.ow[i];
        Motion& vJ = data.vJ[i];
        Motion& wJ = data.wJ[i];

        jmodel.calc(jdata.derived(), q.derived(), v.derived());

        data.liMi[i] = model.jointPlacements[i] * jdata.M();

        if (parent > 0) {
            data.oMi[i] = data.oMi[parent] * data.liMi[i];
            ov = data.ov[parent];
            oa = data.oa[parent];
            ow = data.ow[parent];
        } else {
            data.oMi[i] = data.liMi[i];
            ov.setZero();
            ow.setZero();
            oa = -model.gravity;
        }

        typedef typename SizeDepType<JointModel::NV>::template ColsReturn<typename Data::Matrix6x>::Type ColsBlock;
        ColsBlock J_cols = jmodel.jointCols(data.J); 
        ColsBlock dJ_cols = jmodel.jointCols(data.dJ);
        ColsBlock Om_cols = jmodel.jointCols(data.Om);
        ColsBlock ddJ_cols = jmodel.jointCols(data.ddJ);
        ColsBlock vdJ_cols = jmodel.jointCols(data.vdJ);

        // J and vJ
        J_cols.noalias() = data.oMi[i].act(jdata.S());
        vJ = data.oMi[i].act(jdata.v());
        wJ = data.oMi[i].act(jdata.S() * jmodel.jointVelocitySelector(lambda));

        // dJ
        motionSet::motionAction(ov, J_cols, dJ_cols); // Yd

        // ddJ
        motionSet::motionAction(oa, J_cols, ddJ_cols);
        motionSet::motionAction<ADDTO>(ov, dJ_cols, ddJ_cols); // Ydd

        // vdJ
        motionSet::motionAction(vJ, J_cols, vdJ_cols); //Ud
        vdJ_cols.noalias() += dJ_cols + dJ_cols;

        // Om
        motionSet::motionAction(ow, J_cols, Om_cols); // Yd but when using lambda

        // velocity and accelaration finishing
        ov += vJ;
        oa += (ov ^ vJ) + data.oMi[i].act(jdata.S() * jmodel.jointVelocitySelector(a) + jdata.c());
        ow += wJ;

        // Composite rigid body inertia
        Inertia& oY = data.oYcrb[i];

        oY = data.oMi[i].act(model.inertias[i]); // 0_IC_i
        data.of[i] = oY * oa + oY.vxiv(ov);      // 0_f_i
        data.oh_lam[i]  = oY * ow;               // 0_h_i

        data.oBcrb[i] = Coriolis(oY, ov); // 0_BC_i
        data.oz[i] = data.oBcrb[i].matrix().transpose() * data.ow[i].toVector(); // 0_z_i

        data.oDc[i] = Coriolis(oY, -ow);                                         // o_Dc_i when using lambda
    }
    
    
    template<typename ForceDerived, typename M6>
    static void addForceCrossMatrix(const ForceDense<ForceDerived> & f,
                                    const Eigen::MatrixBase<M6> & mout)
    {
      M6 & mout_ = PINOCCHIO_EIGEN_CONST_CAST(M6,mout);
      addSkew(-f.linear(),mout_.template block<3,3>(ForceDerived::LINEAR,ForceDerived::ANGULAR));
      addSkew(-f.linear(),mout_.template block<3,3>(ForceDerived::ANGULAR,ForceDerived::LINEAR));
      addSkew(-f.angular(),mout_.template block<3,3>(ForceDerived::ANGULAR,ForceDerived::ANGULAR));
    }
    
  };
  
  template<typename Scalar, int Options, template<typename,int> class JointCollectionTpl, typename MatrixType1, typename MatrixType2, typename MatrixType3, typename MatrixType4>
  struct computeModRNEASecondOrderDerivativesBackwardStep
  : public fusion::JointUnaryVisitorBase<computeModRNEASecondOrderDerivativesBackwardStep<Scalar,Options,JointCollectionTpl,MatrixType1,MatrixType2,MatrixType3,MatrixType4> >
  {
    typedef ModelTpl<Scalar,Options,JointCollectionTpl> Model;
    typedef DataTpl<Scalar,Options,JointCollectionTpl> Data;

    typedef boost::fusion::vector<const Model &,
                                  Data &,
                                  const MatrixType1 &,
                                  const MatrixType2 &,
                                  const MatrixType3 &,
                                  const MatrixType4 &
                                  > ArgsType;

    template<typename JointModel>
    static void algo(const JointModelBase<JointModel> & jmodel,
                     const Model & model,
                     Data & data,
                     const Eigen::MatrixBase<MatrixType1> & rnea_partial_dqdq_mod,
                     const Eigen::MatrixBase<MatrixType2> & rnea_partial_dvdv_mod,
                     const Eigen::MatrixBase<MatrixType3> & rnea_partial_dvdq_mod,
                     const Eigen::MatrixBase<MatrixType4> & rnea_partial_dqa_mod)
    {
      typedef typename Model::JointIndex JointIndex;

        const JointIndex& i = jmodel.id();
        const JointIndex& parent = model.parents[i];

        MatrixType1& rnea_partial_dqdq_mod_ = PINOCCHIO_EIGEN_CONST_CAST(MatrixType1,rnea_partial_dqdq_mod);
        MatrixType2& rnea_partial_dvdv_mod_ = PINOCCHIO_EIGEN_CONST_CAST(MatrixType2,rnea_partial_dvdv_mod);
        MatrixType3& rnea_partial_dvdq_mod_ = PINOCCHIO_EIGEN_CONST_CAST(MatrixType3,rnea_partial_dvdq_mod);
        MatrixType4& rnea_partial_dqa_mod_ = PINOCCHIO_EIGEN_CONST_CAST(MatrixType4,rnea_partial_dqa_mod);

        typedef typename SizeDepType<JointModel::NV>::template ColsReturn<typename Data::Matrix6x>::Type ColsBlock;

        ColsBlock J_cols = jmodel.jointCols(data.J); // size of this is 6 x nv_i where nv_i is the dof of joint i
        ColsBlock dJ_cols = jmodel.jointCols(data.dJ);
        ColsBlock ddJ_cols = jmodel.jointCols(data.ddJ);
        ColsBlock vdJ_cols = jmodel.jointCols(data.vdJ);

        ColsBlock tmp1 = jmodel.jointCols(data.Ftmp1);
        ColsBlock tmp2 = jmodel.jointCols(data.Ftmp2);
        ColsBlock tmp3 = jmodel.jointCols(data.Ftmp3); // tmp3 is for this joint only, Ftmp3 is for the full body
        ColsBlock tmp4 = jmodel.jointCols(data.Ftmp4);
        ColsBlock tmp5 = jmodel.jointCols(data.Ftmp5);// F5(:,i)
        //   hphi{i} = icrf(h{i})*S(:,ii); % Si x* hi
        // Note: hphi should be 6 x nv_i, we use tmp1 as temporary storage
        motionSet::act(J_cols, data.oh_lam[i], tmp1); // S{i} x* h{i}


        const Eigen::Index joint_idx = (Eigen::Index)jmodel.idx_v(); // starting index of the joint i
        const Eigen::Index joint_dofs = (Eigen::Index)jmodel.nv(); // no of dofs of joint i
        const Eigen::Index subtree_dofs = (Eigen::Index)data.nvSubtree[i];
        const Eigen::Index successor_idx = joint_idx + joint_dofs; // successor joint starting index
        const Eigen::Index successor_dofs = subtree_dofs - joint_dofs; // all successor joints dofs

        motionSet::act(J_cols, data.of[i], tmp3); // fphi{i} = icrf(f{i})*S(:,ii)

        // F5(:,ii) = 2 * Dc{i}' * S(:,ii) in MATLAB (where Dc has factor 1/2 from factorFunctions)
        // In C++, Coriolis constructor already includes the proper factor, so no multiplication by 2 needed
        motionSet::coriolisTransposeAction(data.oDc[i], J_cols, tmp5);

        // ===================== dmod_dvv computation =====================
        rnea_partial_dvdv_mod_.block(joint_idx, joint_idx, joint_dofs, subtree_dofs).noalias()
          = J_cols.transpose() * data.Ftmp5.middleCols(joint_idx, subtree_dofs); // dmod_dvv (ii,jj)  = S(:,ii).' *F5(:,jj);

        //   dmod_dvv (ii,ii)  = dmod_dvv(ii,ii)   + hphi{i}.' *S(:,ii);
        rnea_partial_dvdv_mod_.block(joint_idx, joint_idx, joint_dofs, joint_dofs).noalias()
          += tmp1.transpose() * J_cols; // dmod_dvv (ii,ii)  = dmod_dvv(ii,ii)   + hphi{i}.' *S(:,ii);

        if (successor_dofs > 0) {
            rnea_partial_dvdv_mod_.block(successor_idx, joint_idx, successor_dofs, joint_dofs).noalias()
                = rnea_partial_dvdv_mod_.block(joint_idx, successor_idx, joint_dofs, successor_dofs).transpose();
        }

        // ===================== dmod_dqq computation =====================
        // Get Om_cols for this joint
        ColsBlock Om_cols = jmodel.jointCols(data.Om);
        ColsBlock tmp6 = jmodel.jointCols(data.Ftmp6);  // For F6
        ColsBlock tmp7 = jmodel.jointCols(data.Ftmp7);  // For F7

        // F2(:,ii) = Bc{i} * S(:,ii) + Ic{i} * Ud(:,ii)
        // Note: MATLAB has "2 * Bc" but Bc = 1/2*(...), so 2*Bc cancels to full Coriolis
        // C++ Coriolis implementation matches 2*Bc directly (no 1/2 factor)
        motionSet::coriolisAction(data.oBcrb[i], J_cols, tmp2);
        motionSet::inertiaAction<ADDTO>(data.oYcrb[i], vdJ_cols, tmp2);

        // F3(:,ii) = Bc{i} * Yd(:,ii) + Ic{i} * Ydd(:,ii) + fphi{i}
        // tmp4 used for F3 (stored in Ftmp4)
        motionSet::coriolisAction(data.oBcrb[i], dJ_cols, tmp4);
        motionSet::inertiaAction<ADDTO>(data.oYcrb[i], ddJ_cols, tmp4);
        tmp4 += tmp3; // add fphi (which is in tmp3)

        // F6(:,ii) = 2*Dc{i}*Yd(:,ii) + 2*Bc{i}'*Om(:,ii) + 2*icrf(h{i})*Yd(:,ii) + zphi{i}
        //          where zphi = 2*icrf(z)*S  and  Dc_cpp=2*Dc_matlab, Bc_cpp=2*Bc_matlab, z_cpp=2*z_matlab
        // The factor of 2 is absorbed by the Coriolis class for Dc, Bc, and z terms,
        // but h = Ic*w has no such factor, so we must explicitly multiply by 2.
        // Store in Ftmp6
        motionSet::coriolisAction(data.oDc[i], dJ_cols, tmp6);                     // Dc_cpp * Yd = 2*Dc_matlab * Yd
        motionSet::coriolisTransposeAction<ADDTO>(data.oBcrb[i], Om_cols, tmp6);   // + Bc_cpp' * Om = 2*Bc_matlab' * Om
        motionSet::act<ADDTO>(dJ_cols, data.oh_lam[i], tmp6);                      // + icrf(h) * Yd (first copy)
        motionSet::act<ADDTO>(dJ_cols, data.oh_lam[i], tmp6);                      // + icrf(h) * Yd (second copy, total = 2*icrf(h)*Yd)
        // zphi = icrf(z_cpp) * S = icrf(2*z_matlab) * S = 2*icrf(z_matlab) * S
        typename Data::Force z_force(data.oz[i]);
        motionSet::act<ADDTO>(J_cols, z_force, tmp6);                              // + icrf(z_cpp) * S

        // F7(:,ii) = Ic{i} * Om(:,ii) + hphi{i}
        // Store in Ftmp7
        motionSet::inertiaAction(data.oYcrb[i], Om_cols, tmp7);
        tmp7 += tmp1;  // + hphi (which is in tmp1)

        // dmod_dqq (jj,ii) = F3(:,jj).'*Om(:,ii) + F6(:,jj).'*Yd(:,ii) + F7(:,jj).'*Ydd(:,ii)
        // In MATLAB: jj = subtree_vinds{i} (this joint + descendants)
        //            ii = vinds{i} (this joint only)
        // F3(:,jj), F6(:,jj), F7(:,jj) were computed in previous iterations for descendant joints
        // and just now for this joint (stored in tmp4, tmp6, tmp7 -> Ftmp4, Ftmp6, Ftmp7)

        rnea_partial_dqdq_mod_.block(joint_idx, joint_idx, subtree_dofs, joint_dofs).noalias()
          = data.Ftmp4.middleCols(joint_idx, subtree_dofs).transpose() * Om_cols
          + data.Ftmp6.middleCols(joint_idx, subtree_dofs).transpose() * dJ_cols
          + data.Ftmp7.middleCols(joint_idx, subtree_dofs).transpose() * ddJ_cols;

        // Symmetry: dmod_dqq(ii, kk) = dmod_dqq(kk, ii)'
        if (successor_dofs > 0) {
            rnea_partial_dqdq_mod_.block(joint_idx, successor_idx, joint_dofs, successor_dofs).noalias()
                = rnea_partial_dqdq_mod_.block(successor_idx, joint_idx, successor_dofs, joint_dofs).transpose();
        }

        // ===================== dmod_dqv computation =====================
        // dmod_dqv (ii,jj)  = Om(:,ii).'*F2(:,jj)  + Yd(:,ii).'*F5(:,jj);
        rnea_partial_dvdq_mod_.block(joint_idx, joint_idx, joint_dofs, subtree_dofs).noalias()
          = Om_cols.transpose() * data.Ftmp2.middleCols(joint_idx, subtree_dofs)
          + dJ_cols.transpose() * data.Ftmp5.middleCols(joint_idx, subtree_dofs);

        // dmod_dqv (kk,ii)  = F6(:,kk).'*S(:,ii)   + F7(:,kk).'*Ud(:,ii);
        // kk = successor indices
        if (successor_dofs > 0) {
            rnea_partial_dvdq_mod_.block(successor_idx, joint_idx, successor_dofs, joint_dofs).noalias()
              = data.Ftmp6.middleCols(successor_idx, successor_dofs).transpose() * J_cols
              + data.Ftmp7.middleCols(successor_idx, successor_dofs).transpose() * vdJ_cols;
        }

        // ===================== dmod_dqa computation =====================
        // dmod_dqa(m,j) = ∂²(λτ)/(∂q_m ∂a_j) = ∂M_mod_j/∂q_m
        // Convention: row = q direction, col = a direction (same as dqv)
        // F8(:,ii) = Ic_i^{comp} * S(:,ii)
        ColsBlock tmp8 = jmodel.jointCols(data.Ftmp8);
        motionSet::inertiaAction(data.oYcrb[i], J_cols, tmp8);

        // Case 3: m in subtree(j=i), dqa(m,j) = S_j^T * F7(:,m)
        // Block (jj, ii): rows = subtree q-indices, cols = joint i a-indices
        rnea_partial_dqa_mod_.block(joint_idx, joint_idx, subtree_dofs, joint_dofs).noalias()
          = data.Ftmp7.middleCols(joint_idx, subtree_dofs).transpose() * J_cols;

        // Diagonal correction for multi-DOF joints (j=m same joint):
        // F7(:,i)^T * S_i includes hphi^T * S = (icrf(h)*S)^T * S, but the ∂S_i/∂q_i
        // contribution cancels it. Since icrf(h) is antisymmetric, hphi^T*S ≠ S^T*hphi.
        // For single-DOF joints hphi^T*S=0 (v×*f)^T*v=0), but for multi-DOF (free-flyer)
        // the off-diagonal elements are nonzero. We subtract hphi^T * S (= tmp1^T * J_cols).
        rnea_partial_dqa_mod_.block(joint_idx, joint_idx, joint_dofs, joint_dofs).noalias()
          -= tmp1.transpose() * J_cols;  // subtract hphi^T * S

        // Case 2: j in subtree(m=i), dqa(m,j) = S_j^T * Ic_j * Om_m
        // Block (ii, kk): rows = joint i q-indices, cols = successor a-indices
        if (successor_dofs > 0) {
            rnea_partial_dqa_mod_.block(joint_idx, successor_idx, joint_dofs, successor_dofs).noalias()
              = Om_cols.transpose() * data.Ftmp8.middleCols(successor_idx, successor_dofs);
        }

        if (parent > 0) {
            data.oz[parent] += data.oz[i];
            data.oh_lam[parent] += data.oh_lam[i];
            data.of[parent] += data.of[i];
            data.oYcrb[parent] += data.oYcrb[i];
            data.oBcrb[parent] += data.oBcrb[i];
            data.oDc[parent] += data.oDc[i];
        }
    

    }
    
    template<typename Min, typename Mout>
    static void lhsInertiaMult(const typename Data::Inertia & Y,
                               const Eigen::MatrixBase<Min> & J,
                               const Eigen::MatrixBase<Mout> & F)
    {
      Mout & F_ = PINOCCHIO_EIGEN_CONST_CAST(Mout,F);
      motionSet::inertiaAction(Y,J.derived().transpose(),F_.transpose());
    }
  };
  
  // 4-matrix version (with dqa output)
  template<typename Scalar, int Options, template<typename,int> class JointCollectionTpl, typename ConfigVectorType, typename TangentVectorType1, typename TangentVectorType2,
  typename TangentVectorType3, typename MatrixType1, typename MatrixType2, typename MatrixType3, typename MatrixType4>
  inline void
  computeModRNEASecondOrderDerivatives(const ModelTpl<Scalar,Options,JointCollectionTpl> & model,
                         DataTpl<Scalar,Options,JointCollectionTpl> & data,
                         const Eigen::MatrixBase<ConfigVectorType> & q,
                         const Eigen::MatrixBase<TangentVectorType1> & v,
                         const Eigen::MatrixBase<TangentVectorType2> & a,
                         const Eigen::MatrixBase<TangentVectorType3> & lambda,
                         const Eigen::MatrixBase<MatrixType1> & rnea_partial_dqdq_mod,
                         const Eigen::MatrixBase<MatrixType2> & rnea_partial_dvdv_mod,
                         const Eigen::MatrixBase<MatrixType3> & rnea_partial_dvdq_mod,
                         const Eigen::MatrixBase<MatrixType4> & rnea_partial_dqa_mod)
  {
    PINOCCHIO_CHECK_ARGUMENT_SIZE(q.size(), model.nq, "The joint configuration vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(v.size(), model.nv, "The joint velocity vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(a.size(), model.nv, "The joint acceleration vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(lambda.size(), model.nv, "The input vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(rnea_partial_dqdq_mod.rows(), model.nv);
    PINOCCHIO_CHECK_ARGUMENT_SIZE(rnea_partial_dqdq_mod.cols(), model.nv);
    PINOCCHIO_CHECK_ARGUMENT_SIZE(rnea_partial_dvdv_mod.rows(), model.nv);
    PINOCCHIO_CHECK_ARGUMENT_SIZE(rnea_partial_dvdv_mod.cols(), model.nv);
    PINOCCHIO_CHECK_ARGUMENT_SIZE(rnea_partial_dvdq_mod.rows(), model.nv);
    PINOCCHIO_CHECK_ARGUMENT_SIZE(rnea_partial_dvdq_mod.rows(), model.nv);
    PINOCCHIO_CHECK_ARGUMENT_SIZE(rnea_partial_dqa_mod.rows(), model.nv);
    PINOCCHIO_CHECK_ARGUMENT_SIZE(rnea_partial_dqa_mod.cols(), model.nv);
    assert(model.check(data) && "data is not consistent with model.");

    typedef ModelTpl<Scalar,Options,JointCollectionTpl> Model;
    typedef typename Model::JointIndex JointIndex;

    data.oa_gf[0] = -model.gravity;

    typedef computeModRNEASecondOrderDerivativesForwardStep<Scalar,Options,JointCollectionTpl,ConfigVectorType,TangentVectorType1,TangentVectorType2,TangentVectorType3> Pass1;
    for(JointIndex i=1; i<(JointIndex) model.njoints; ++i)
    {
      Pass1::run(model.joints[i],data.joints[i],
                 typename Pass1::ArgsType(model,data,q.derived(),v.derived(),a.derived(),lambda.derived()));
    }

    typedef computeModRNEASecondOrderDerivativesBackwardStep<Scalar,Options,JointCollectionTpl,MatrixType1,MatrixType2,MatrixType3,MatrixType4> Pass2;
    for(JointIndex i=(JointIndex)(model.njoints-1); i>0; --i)
    {
      Pass2::run(model.joints[i],
                 typename Pass2::ArgsType(model,data,
                                          PINOCCHIO_EIGEN_CONST_CAST(MatrixType1,rnea_partial_dqdq_mod),
                                          PINOCCHIO_EIGEN_CONST_CAST(MatrixType2,rnea_partial_dvdv_mod),
                                          PINOCCHIO_EIGEN_CONST_CAST(MatrixType3,rnea_partial_dvdq_mod),
                                          PINOCCHIO_EIGEN_CONST_CAST(MatrixType4,rnea_partial_dqa_mod)));
    }
  }

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
                         const Eigen::MatrixBase<MatrixType1> & rnea_partial_dqdq_mod,
                         const Eigen::MatrixBase<MatrixType2> & rnea_partial_dvdv_mod,
                         const Eigen::MatrixBase<MatrixType3> & rnea_partial_dvdq_mod,
                         const Eigen::MatrixBase<MatrixType4> & rnea_partial_dqa_mod)
  {
    PINOCCHIO_CHECK_ARGUMENT_SIZE(q.size(), model.nq, "The joint configuration vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(v.size(), model.nv, "The joint velocity vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(a.size(), model.nv, "The joint acceleration vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(lambda.size(), model.nv, "The input vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(fext.size(), (size_t)model.njoints, "The size of the external forces is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(rnea_partial_dqdq_mod.rows(), model.nv);
    PINOCCHIO_CHECK_ARGUMENT_SIZE(rnea_partial_dqdq_mod.cols(), model.nv);
    PINOCCHIO_CHECK_ARGUMENT_SIZE(rnea_partial_dvdv_mod.rows(), model.nv);
    PINOCCHIO_CHECK_ARGUMENT_SIZE(rnea_partial_dvdv_mod.cols(), model.nv);
    PINOCCHIO_CHECK_ARGUMENT_SIZE(rnea_partial_dvdq_mod.rows(), model.nv);
    PINOCCHIO_CHECK_ARGUMENT_SIZE(rnea_partial_dvdq_mod.cols(), model.nv);
    PINOCCHIO_CHECK_ARGUMENT_SIZE(rnea_partial_dqa_mod.rows(), model.nv);
    PINOCCHIO_CHECK_ARGUMENT_SIZE(rnea_partial_dqa_mod.cols(), model.nv);
    assert(model.check(data) && "data is not consistent with model.");

    typedef ModelTpl<Scalar,Options,JointCollectionTpl> Model;
    typedef typename Model::JointIndex JointIndex;

    data.oa_gf[0] = -model.gravity;

    typedef computeModRNEASecondOrderDerivativesForwardStep<Scalar,Options,JointCollectionTpl,ConfigVectorType,TangentVectorType1,TangentVectorType2,TangentVectorType3> Pass1;
    for(JointIndex i=1; i<(JointIndex) model.njoints; ++i)
    {
      Pass1::run(model.joints[i],data.joints[i],
                 typename Pass1::ArgsType(model,data,q.derived(),v.derived(),a.derived(),lambda.derived()));
      data.of[i] -= data.oMi[i].act(fext[i]);
    }

    typedef computeModRNEASecondOrderDerivativesBackwardStep<Scalar,Options,JointCollectionTpl,MatrixType1,MatrixType2,MatrixType3,MatrixType4> Pass2;
    for(JointIndex i=(JointIndex)(model.njoints-1); i>0; --i)
    {
      Pass2::run(model.joints[i],
                 typename Pass2::ArgsType(model,data,
                                          PINOCCHIO_EIGEN_CONST_CAST(MatrixType1,rnea_partial_dqdq_mod),
                                          PINOCCHIO_EIGEN_CONST_CAST(MatrixType2,rnea_partial_dvdv_mod),
                                          PINOCCHIO_EIGEN_CONST_CAST(MatrixType3,rnea_partial_dvdq_mod),
                                          PINOCCHIO_EIGEN_CONST_CAST(MatrixType4,rnea_partial_dqa_mod)));
    }
  }

} // namespace pinocchio

#endif // ifndef __pinocchio_mod_rnea_second_order_derivatives_hxx__
