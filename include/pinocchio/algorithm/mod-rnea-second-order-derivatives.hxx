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
  struct ComputeModRNEASecondOrderDerivativesForwardStep
  : public fusion::JointUnaryVisitorBase< ComputeModRNEASecondOrderDerivativesForwardStep<Scalar,Options,JointCollectionTpl,ConfigVectorType,TangentVectorType1,TangentVectorType2,TangentVectorType3> >
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
        ColsBlock ddJ_cols = jmodel.jointCols(data.ddJ);
        ColsBlock vdJ_cols = jmodel.jointCols(data.vdJ);

        // J and vJ
        J_cols.noalias() = data.oMi[i].act(jdata.S());
        vJ = data.oMi[i].act(jdata.v());
        wJ = data.oMi[i].act(jdata.S() * jmodel.jointVelocitySelector(lambda));

        // dJ
        motionSet::motionAction(ov, J_cols, dJ_cols);

        // ddJ
        motionSet::motionAction(oa, J_cols, ddJ_cols);
        motionSet::motionAction<ADDTO>(ov, dJ_cols, ddJ_cols);

        // vdJ
        motionSet::motionAction(vJ, J_cols, vdJ_cols);
        vdJ_cols.noalias() += dJ_cols + dJ_cols;

        // velocity and accelaration finishing
        ov += vJ;
        oa += (ov ^ vJ) + data.oMi[i].act(jdata.S() * jmodel.jointVelocitySelector(a) + jdata.c());
        ow += wJ;

        // Composite rigid body inertia
        Inertia& oY = data.oYcrb[i];

        oY = data.oMi[i].act(model.inertias[i]); // 0_IC_i
        data.of[i] = oY * oa + oY.vxiv(ov); // 0_f_i
        data.oh_lam[i]  = oY * ow; // 0_h_i

        data.oBcrb[i] = Coriolis(oY, ov); // 0_BC_i
        data.oz[i] = data.oBcrb[i].matrix().transpose() * data.ow[i].toVector(); // 0_z_i
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
  
  template<typename Scalar, int Options, template<typename,int> class JointCollectionTpl, typename MatrixType1, typename MatrixType2, typename MatrixType3>
  struct ComputeModRNEASecondOrderDerivativesBackwardStep
  : public fusion::JointUnaryVisitorBase<ComputeModRNEASecondOrderDerivativesBackwardStep<Scalar,Options,JointCollectionTpl,MatrixType1,MatrixType2,MatrixType3> >
  {
    typedef ModelTpl<Scalar,Options,JointCollectionTpl> Model;
    typedef DataTpl<Scalar,Options,JointCollectionTpl> Data;
    
    typedef boost::fusion::vector<const Model &,
                                  Data &,
                                  const MatrixType1 &,
                                  const MatrixType2 &,
                                  const MatrixType3 &
                                  > ArgsType;
    
    template<typename JointModel>
    static void algo(const JointModelBase<JointModel> & jmodel,
                     const Model & model,
                     Data & data,
                     const Eigen::MatrixBase<MatrixType1> & rnea_partial_dq_mod,
                     const Eigen::MatrixBase<MatrixType2> & rnea_partial_dv_mod,
                     const Eigen::MatrixBase<MatrixType3> & rnea_partial_da_mod)
    {
      typedef typename Model::JointIndex JointIndex;
        
        const JointIndex& i = jmodel.id();
        const JointIndex& parent = model.parents[i];

        MatrixType1& rnea_partial_dq_mod_ = PINOCCHIO_EIGEN_CONST_CAST(MatrixType1,rnea_partial_dq_mod);
        MatrixType2& rnea_partial_dv_mod_ = PINOCCHIO_EIGEN_CONST_CAST(MatrixType2,rnea_partial_dv_mod);
        MatrixType3& rnea_partial_da_mod_ = PINOCCHIO_EIGEN_CONST_CAST(MatrixType3,rnea_partial_da_mod);

        typedef typename SizeDepType<JointModel::NV>::template ColsReturn<typename Data::Matrix6x>::Type ColsBlock;

        ColsBlock J_cols = jmodel.jointCols(data.J);
        ColsBlock dJ_cols = jmodel.jointCols(data.dJ);
        ColsBlock ddJ_cols = jmodel.jointCols(data.ddJ);
        ColsBlock vdJ_cols = jmodel.jointCols(data.vdJ);

        ColsBlock tmp3 = jmodel.jointCols(data.Ftmp3); // tmp3 is for this joint only, Ftmp3 is for the full body

        const Eigen::Index joint_idx = (Eigen::Index)jmodel.idx_v();
        const Eigen::Index joint_dofs = (Eigen::Index)jmodel.nv();

        motionSet::act(J_cols, data.of[i], tmp3); // S{i} x* f{i}

        rnea_partial_dq_mod_.segment(joint_idx, joint_dofs).noalias()
          = tmp3.transpose() * data.ow[parent].toVector() + 
             dJ_cols.transpose() * data.oz[i].toVector() + 
             ddJ_cols.transpose() * data.oh_lam[i].toVector();

        rnea_partial_dv_mod_.segment(joint_idx, joint_dofs).noalias()
          = vdJ_cols.transpose() * data.oh_lam[i].toVector()
          + J_cols.transpose() * data.oz[i].toVector();

        rnea_partial_da_mod_.segment(joint_idx, joint_dofs).noalias()
          = J_cols.transpose() * data.oh_lam[i].toVector();  

        if (parent > 0) {
            data.oz[parent] += data.oz[i];
            data.oh_lam[parent] += data.oh_lam[i];
            data.of[parent] += data.of[i];
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
  
  template<typename Scalar, int Options, template<typename,int> class JointCollectionTpl, typename ConfigVectorType, typename TangentVectorType1, typename TangentVectorType2,
  typename TangentVectorType3, typename MatrixType1, typename MatrixType2, typename MatrixType3>
  inline void
  ComputeModRNEASecondOrderDerivatives(const ModelTpl<Scalar,Options,JointCollectionTpl> & model,
                         DataTpl<Scalar,Options,JointCollectionTpl> & data,
                         const Eigen::MatrixBase<ConfigVectorType> & q,
                         const Eigen::MatrixBase<TangentVectorType1> & v,
                         const Eigen::MatrixBase<TangentVectorType2> & a,
                         const Eigen::MatrixBase<TangentVectorType3> & lambda,
                         const Eigen::MatrixBase<MatrixType1> & rnea_partial_dqdq_mod,
                         const Eigen::MatrixBase<MatrixType2> & rnea_partial_dvdv_mod,
                         const Eigen::MatrixBase<MatrixType3> & rnea_partial_dvdq_mod)
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
    PINOCCHIO_CHECK_ARGUMENT_SIZE(rnea_partial_da_mod.rows(), model.nv);
    assert(model.check(data) && "data is not consistent with model.");
    
    typedef ModelTpl<Scalar,Options,JointCollectionTpl> Model;
    typedef typename Model::JointIndex JointIndex;
    
    data.oa_gf[0] = -model.gravity;
    
    typedef ComputeModRNEASecondOrderDerivativesForwardStep<Scalar,Options,JointCollectionTpl,ConfigVectorType,TangentVectorType1,TangentVectorType2,TangentVectorType3> Pass1;
    for(JointIndex i=1; i<(JointIndex) model.njoints; ++i)
    {
      Pass1::run(model.joints[i],data.joints[i],
                 typename Pass1::ArgsType(model,data,q.derived(),v.derived(),a.derived(),lambda.derived()));
    }
    
    typedef ComputeModRNEASecondOrderDerivativesBackwardStep<Scalar,Options,JointCollectionTpl,MatrixType1,MatrixType2,MatrixType3> Pass2;
    for(JointIndex i=(JointIndex)(model.njoints-1); i>0; --i)
    {
      Pass2::run(model.joints[i],
                 typename Pass2::ArgsType(model,data,
                                          PINOCCHIO_EIGEN_CONST_CAST(MatrixType1,rnea_partial_dqdq_mod),
                                          PINOCCHIO_EIGEN_CONST_CAST(MatrixType2,rnea_partial_dvdv_mod),
                                          PINOCCHIO_EIGEN_CONST_CAST(MatrixType3,rnea_partial_dvdq_mod)));
    }
  }
  
  template<typename Scalar, int Options, template<typename,int> class JointCollectionTpl, typename ConfigVectorType, typename TangentVectorType1, typename TangentVectorType2,
  typename TangentVectorType3, typename MatrixType1, typename MatrixType2, typename MatrixType3>
  inline void
  ComputeModRNEASecondOrderDerivatives(const ModelTpl<Scalar,Options,JointCollectionTpl> & model,
                         DataTpl<Scalar,Options,JointCollectionTpl> & data,
                         const Eigen::MatrixBase<ConfigVectorType> & q,
                         const Eigen::MatrixBase<TangentVectorType1> & v,
                         const Eigen::MatrixBase<TangentVectorType2> & a,
                         const Eigen::MatrixBase<TangentVectorType3> & lambda,
                         const container::aligned_vector< ForceTpl<Scalar,Options> > & fext,
                         const Eigen::MatrixBase<MatrixType1> & rnea_partial_dqdq_mod,
                         const Eigen::MatrixBase<MatrixType2> & rnea_partial_dvdv_mod,
                         const Eigen::MatrixBase<MatrixType3> & rnea_partial_dvdq_mod)
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
    assert(model.check(data) && "data is not consistent with model.");
    
    typedef ModelTpl<Scalar,Options,JointCollectionTpl> Model;
    typedef typename Model::JointIndex JointIndex;
    
    data.oa_gf[0] = -model.gravity;
    
    typedef ComputeModRNEASecondOrderDerivativesForwardStep<Scalar,Options,JointCollectionTpl,ConfigVectorType,TangentVectorType1,TangentVectorType2,TangentVectorType3> Pass1;
    for(JointIndex i=1; i<(JointIndex) model.njoints; ++i)
    {
      Pass1::run(model.joints[i],data.joints[i],
                 typename Pass1::ArgsType(model,data,q.derived(),v.derived(),a.derived(),lambda.derived()));
      data.of[i] -= data.oMi[i].act(fext[i]);
    }
    
    typedef ComputeModRNEASecondOrderDerivativesBackwardStep<Scalar,Options,JointCollectionTpl,MatrixType1,MatrixType2,MatrixType3> Pass2;
    for(JointIndex i=(JointIndex)(model.njoints-1); i>0; --i)
    {
      Pass2::run(model.joints[i],
                 typename Pass2::ArgsType(model,data,
                                          PINOCCHIO_EIGEN_CONST_CAST(MatrixType1,rnea_partial_dqdq_mod),
                                          PINOCCHIO_EIGEN_CONST_CAST(MatrixType2,rnea_partial_dvdv_mod),
                                          PINOCCHIO_EIGEN_CONST_CAST(MatrixType3,rnea_partial_dvdq_mod)));
    }
  }
  

} // namespace pinocchio

#endif // ifndef __pinocchio_mod_rnea_second_order_derivatives_hxx__
