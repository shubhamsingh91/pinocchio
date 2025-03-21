//
// Copyright (c) 2017-2020 CNRS INRIA
//

#ifndef __pinocchio_mod_rnea_derivatives_hxx__
#define __pinocchio_mod_rnea_derivatives_hxx__

#include "pinocchio/multibody/visitor.hpp"
#include "pinocchio/algorithm/check.hpp"
#include "pinocchio/algorithm/mod-rnea-derivatives.hpp"

namespace pinocchio
{
  

  template<typename Scalar, int Options, template<typename,int> class JointCollectionTpl, typename ConfigVectorType, typename TangentVectorType1, typename TangentVectorType2, typename TangentVectorType3>
  struct ComputeModRNEADerivativesForwardStep
  : public fusion::JointUnaryVisitorBase< ComputeModRNEADerivativesForwardStep<Scalar,Options,JointCollectionTpl,ConfigVectorType,TangentVectorType1,TangentVectorType2,TangentVectorType3> >
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

      const JointIndex & i = jmodel.id();
      const JointIndex & parent = model.parents[i];
      Motion & ov = data.ov[i];
      Motion & oa = data.oa[i];
      Motion & oa_gf = data.oa_gf[i];
      
      jmodel.calc(jdata.derived(),q.derived(),v.derived());
      
      data.liMi[i] = model.jointPlacements[i]*jdata.M();
      
      data.v[i] = jdata.v();
      
      if(parent > 0)
      {
        data.oMi[i] = data.oMi[parent] * data.liMi[i];
        data.v[i] += data.liMi[i].actInv(data.v[parent]);
      }
      else
        data.oMi[i] = data.liMi[i];
      
      data.a[i] = jdata.S() * jmodel.jointVelocitySelector(a) + jdata.c() + (data.v[i] ^ jdata.v());
      if(parent > 0)
      {
        data.a[i] += data.liMi[i].actInv(data.a[parent]);
      }
      
      data.oYcrb[i] = data.oinertias[i] = data.oMi[i].act(model.inertias[i]);
      ov = data.oMi[i].act(data.v[i]);
      oa = data.oMi[i].act(data.a[i]);
      oa_gf = oa - model.gravity; // add gravity contribution
      
      data.oh[i] = data.oYcrb[i] * ov;
      data.of[i] = data.oYcrb[i] * oa_gf + ov.cross(data.oh[i]);
      data.f[i] = data.oMi[i].actInv(data.of[i]);
    
      typedef typename SizeDepType<JointModel::NV>::template ColsReturn<typename Data::Matrix6x>::Type ColsBlock;
      ColsBlock J_cols = jmodel.jointCols(data.J);
      ColsBlock dJ_cols = jmodel.jointCols(data.dJ);
      ColsBlock dVdq_cols = jmodel.jointCols(data.dVdq);
      ColsBlock dAdq_cols = jmodel.jointCols(data.dAdq);
      ColsBlock dAdv_cols = jmodel.jointCols(data.dAdv);

      J_cols = data.oMi[i].act(jdata.S());
      motionSet::motionAction(ov,J_cols,dJ_cols);
      motionSet::motionAction(data.oa_gf[parent],J_cols,dAdq_cols);
      dAdv_cols = dJ_cols;
      if(parent > 0)
      {
        motionSet::motionAction(data.ov[parent],J_cols,dVdq_cols);
        motionSet::motionAction<ADDTO>(data.ov[parent],dVdq_cols,dAdq_cols);
        dAdv_cols.noalias() += dVdq_cols;
      }
      else
      {
        dVdq_cols.setZero();
      }

      // computes variation of inertias
      data.doYcrb[i] = data.oYcrb[i].variation(ov);
      
      addForceCrossMatrix(data.oh[i],data.doYcrb[i]);
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
  
  template<typename Scalar, int Options, template<typename,int> class JointCollectionTpl, typename VectorType1, typename VectorType2, typename VectorType3>
  struct ComputeModRNEADerivativesBackwardStep
  : public fusion::JointUnaryVisitorBase<ComputeModRNEADerivativesBackwardStep<Scalar,Options,JointCollectionTpl,VectorType1,VectorType2,VectorType3> >
  {
    typedef ModelTpl<Scalar,Options,JointCollectionTpl> Model;
    typedef DataTpl<Scalar,Options,JointCollectionTpl> Data;
    
    typedef boost::fusion::vector<const Model &,
                                  Data &,
                                  const VectorType1 &,
                                  const VectorType2 &,
                                  const VectorType3 &
                                  > ArgsType;
    
    template<typename JointModel>
    static void algo(const JointModelBase<JointModel> & jmodel,
                     const Model & model,
                     Data & data,
                     const Eigen::MatrixBase<VectorType1> & rnea_partial_dq_mod,
                     const Eigen::MatrixBase<VectorType2> & rnea_partial_dv_mod,
                     const Eigen::MatrixBase<VectorType3> & rnea_partial_da_mod)
    {
      typedef typename Model::JointIndex JointIndex;
      
      // const JointIndex & i = jmodel.id();
      // const JointIndex & parent = model.parents[i];
      // typename Data::RowMatrix6 & M6tmpR = data.M6tmpR;
      // typename Data::RowMatrix6 & M6tmpR2 = data.M6tmpR2;

      // typedef typename SizeDepType<JointModel::NV>::template ColsReturn<typename Data::Matrix6x>::Type ColsBlock;
      
      // ColsBlock J_cols = jmodel.jointCols(data.J);
      // ColsBlock dVdq_cols = jmodel.jointCols(data.dVdq);
      // ColsBlock dAdq_cols = jmodel.jointCols(data.dAdq);
      // ColsBlock dAdv_cols = jmodel.jointCols(data.dAdv);
      // ColsBlock dFdq_cols = jmodel.jointCols(data.dFdq);
      // ColsBlock dFdv_cols = jmodel.jointCols(data.dFdv);
      // ColsBlock dFda_cols = jmodel.jointCols(data.dFda);
      
      // VectorType1 & rnea_partial_dq_mod_ = PINOCCHIO_EIGEN_CONST_CAST(VectorType1,rnea_partial_dq_mod);
      // VectorType2 & rnea_partial_dv_ = PINOCCHIO_EIGEN_CONST_CAST(VectorType2,rnea_partial_dv_mod);
      // VectorType3 & rnea_partial_da_ = PINOCCHIO_EIGEN_CONST_CAST(VectorType3,rnea_partial_da_mod);

      // pinocchio::SE3 X_0_m = data.oMi[i];
      // typename Data::Matrix6 X_0_mT_mat = X_0_m.toActionMatrix().transpose();

      // // tau
      // jmodel.jointVelocitySelector(data.tau).noalias() = J_cols.transpose()*data.of[i].toVector();
      
      // // dtau/da similar to data.M
      // motionSet::inertiaAction(data.oYcrb[i],J_cols,dFda_cols);
      // rnea_partial_da_.block(jmodel.idx_v(),jmodel.idx_v(),jmodel.nv(),data.nvSubtree[i]).noalias()
      // = J_cols.transpose()*data.dFda.middleCols(jmodel.idx_v(),data.nvSubtree[i]);
      
      // // dtau/dv
      // dFdv_cols.noalias() = data.doYcrb[i] * J_cols;
      // motionSet::inertiaAction<ADDTO>(data.oYcrb[i],dAdv_cols,dFdv_cols);

      // rnea_partial_dv_.block(jmodel.idx_v(),jmodel.idx_v(),jmodel.nv(),data.nvSubtree[i]).noalias()
      // = J_cols.transpose()*data.dFdv.middleCols(jmodel.idx_v(),data.nvSubtree[i]);
      
      // // dtau/dq
      // if(parent>0)
      // {
      //   dFdq_cols.noalias() = data.doYcrb[i] * dVdq_cols;
      //   motionSet::inertiaAction<ADDTO>(data.oYcrb[i],dAdq_cols,dFdq_cols);
      // }
      // else
      //   motionSet::inertiaAction(data.oYcrb[i],dAdq_cols,dFdq_cols);

      // rnea_partial_dq_mod_.block(jmodel.idx_v(),jmodel.idx_v(),jmodel.nv(),data.nvSubtree[i]).noalias()
      // = J_cols.transpose()*data.dFdq.middleCols(jmodel.idx_v(),data.nvSubtree[i]);
      
      // motionSet::act<ADDTO>(J_cols,data.of[i],dFdq_cols);
      
      // if(parent > 0)
      // {
      //   lhsInertiaMult(data.oYcrb[i],J_cols.transpose(),M6tmpR.topRows(jmodel.nv()));
      //   M6tmpR2.topRows(jmodel.nv()).noalias() = J_cols.transpose() * data.doYcrb[i];
      //   for(int j = data.parents_fromRow[(typename Model::Index)jmodel.idx_v()];j >= 0; j = data.parents_fromRow[(typename Model::Index)j])
      //   {
      //     rnea_partial_dq_.middleRows(jmodel.idx_v(),jmodel.nv()).col(j).noalias()
      //     = M6tmpR.topRows(jmodel.nv()) * data.dAdq.col(j)
      //     + M6tmpR2.topRows(jmodel.nv()) * data.dVdq.col(j);
      //   }
      //   for(int j = data.parents_fromRow[(typename Model::Index)jmodel.idx_v()];j >= 0; j = data.parents_fromRow[(typename Model::Index)j])
      //   {
      //     rnea_partial_dv_.middleRows(jmodel.idx_v(),jmodel.nv()).col(j).noalias()
      //     = M6tmpR.topRows(jmodel.nv()) * data.dAdv.col(j)
      //     + M6tmpR2.topRows(jmodel.nv()) * data.J.col(j);
      //   }
      // }
      // if(parent>0)
      // {
      //   data.oYcrb[parent] += data.oYcrb[i];
      //   data.doYcrb[parent] += data.doYcrb[i];
      //   data.of[parent] += data.of[i];
      //   data.f[parent] += data.liMi[i].act(data.f[i]);
      // }      
      // // Restore the status of dAdq_cols (remove gravity)
      // PINOCCHIO_CHECK_INPUT_ARGUMENT(isZero(model.gravity.angular()),
      //                                "The gravity must be a pure force vector, no angular part");
      // for(Eigen::DenseIndex k =0; k < jmodel.nv(); ++k)
      // {
      //   MotionRef<typename ColsBlock::ColXpr> m_in(J_cols.col(k));
      //   MotionRef<typename ColsBlock::ColXpr> m_out(dAdq_cols.col(k));
      //   m_out.linear() += model.gravity.linear().cross(m_in.angular());
      // }
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
  typename TangentVectorType3, typename VectorType1, typename VectorType2, typename VectorType3>
  inline void
  computeModRNEADerivatives(const ModelTpl<Scalar,Options,JointCollectionTpl> & model,
                         DataTpl<Scalar,Options,JointCollectionTpl> & data,
                         const Eigen::MatrixBase<ConfigVectorType> & q,
                         const Eigen::MatrixBase<TangentVectorType1> & v,
                         const Eigen::MatrixBase<TangentVectorType2> & a,
                         const Eigen::MatrixBase<TangentVectorType3> & lambda,
                         const Eigen::MatrixBase<VectorType1> & rnea_partial_dq_mod,
                         const Eigen::MatrixBase<VectorType2> & rnea_partial_dv_mod,
                         const Eigen::MatrixBase<VectorType3> & rnea_partial_da_mod)
  {
    PINOCCHIO_CHECK_ARGUMENT_SIZE(q.size(), model.nq, "The joint configuration vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(v.size(), model.nv, "The joint velocity vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(a.size(), model.nv, "The joint acceleration vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(lambda.size(), model.nv, "The input vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(rnea_partial_dq_mod.rows(), model.nv);
    PINOCCHIO_CHECK_ARGUMENT_SIZE(rnea_partial_dv_mod.rows(), model.nv);
    PINOCCHIO_CHECK_ARGUMENT_SIZE(rnea_partial_da_mod.rows(), model.nv);
    assert(model.check(data) && "data is not consistent with model.");
    
    typedef ModelTpl<Scalar,Options,JointCollectionTpl> Model;
    typedef typename Model::JointIndex JointIndex;
    
    data.oa_gf[0] = -model.gravity;
    
    typedef ComputeModRNEADerivativesForwardStep<Scalar,Options,JointCollectionTpl,ConfigVectorType,TangentVectorType1,TangentVectorType2,TangentVectorType3> Pass1;
    for(JointIndex i=1; i<(JointIndex) model.njoints; ++i)
    {
      Pass1::run(model.joints[i],data.joints[i],
                 typename Pass1::ArgsType(model,data,q.derived(),v.derived(),a.derived(),lambda.derived()));
    }
    
    typedef ComputeModRNEADerivativesBackwardStep<Scalar,Options,JointCollectionTpl,VectorType1,VectorType2,VectorType3> Pass2;
    for(JointIndex i=(JointIndex)(model.njoints-1); i>0; --i)
    {
      Pass2::run(model.joints[i],
                 typename Pass2::ArgsType(model,data,
                                          PINOCCHIO_EIGEN_CONST_CAST(VectorType1,rnea_partial_dq_mod),
                                          PINOCCHIO_EIGEN_CONST_CAST(VectorType2,rnea_partial_dv_mod),
                                          PINOCCHIO_EIGEN_CONST_CAST(VectorType3,rnea_partial_da_mod)));
    }
  }
  
  template<typename Scalar, int Options, template<typename,int> class JointCollectionTpl, typename ConfigVectorType, typename TangentVectorType1, typename TangentVectorType2,
  typename TangentVectorType3, typename VectorType1, typename VectorType2, typename VectorType3>
  inline void
  computeModRNEADerivatives(const ModelTpl<Scalar,Options,JointCollectionTpl> & model,
                         DataTpl<Scalar,Options,JointCollectionTpl> & data,
                         const Eigen::MatrixBase<ConfigVectorType> & q,
                         const Eigen::MatrixBase<TangentVectorType1> & v,
                         const Eigen::MatrixBase<TangentVectorType2> & a,
                         const Eigen::MatrixBase<TangentVectorType3> & lambda,
                         const container::aligned_vector< ForceTpl<Scalar,Options> > & fext,
                         const Eigen::MatrixBase<VectorType1> & rnea_partial_dq_mod,
                         const Eigen::MatrixBase<VectorType2> & rnea_partial_dv_mod,
                         const Eigen::MatrixBase<VectorType3> & rnea_partial_da_mod)
  {
    PINOCCHIO_CHECK_ARGUMENT_SIZE(q.size(), model.nq, "The joint configuration vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(v.size(), model.nv, "The joint velocity vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(a.size(), model.nv, "The joint acceleration vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(lambda.size(), model.nv, "The input vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(fext.size(), (size_t)model.njoints, "The size of the external forces is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(rnea_partial_dq_mod.rows(), model.nv);
    PINOCCHIO_CHECK_ARGUMENT_SIZE(rnea_partial_dv_mod.rows(), model.nv);
    PINOCCHIO_CHECK_ARGUMENT_SIZE(rnea_partial_da_mod.rows(), model.nv);
    assert(model.check(data) && "data is not consistent with model.");
    
    typedef ModelTpl<Scalar,Options,JointCollectionTpl> Model;
    typedef typename Model::JointIndex JointIndex;
    
    data.oa_gf[0] = -model.gravity;
    
    typedef ComputeModRNEADerivativesForwardStep<Scalar,Options,JointCollectionTpl,ConfigVectorType,TangentVectorType1,TangentVectorType2,TangentVectorType3> Pass1;
    for(JointIndex i=1; i<(JointIndex) model.njoints; ++i)
    {
      Pass1::run(model.joints[i],data.joints[i],
                 typename Pass1::ArgsType(model,data,q.derived(),v.derived(),a.derived(),lambda.derived()));
      data.of[i] -= data.oMi[i].act(fext[i]);
    }
    
    typedef ComputeModRNEADerivativesBackwardStep<Scalar,Options,JointCollectionTpl,VectorType1,VectorType2,VectorType3> Pass2;
    for(JointIndex i=(JointIndex)(model.njoints-1); i>0; --i)
    {
      Pass2::run(model.joints[i],
                 typename Pass2::ArgsType(model,data,
                                          PINOCCHIO_EIGEN_CONST_CAST(VectorType1,rnea_partial_dq_mod),
                                          PINOCCHIO_EIGEN_CONST_CAST(VectorType2,rnea_partial_dv_mod),
                                          PINOCCHIO_EIGEN_CONST_CAST(VectorType3,rnea_partial_da_mod)));
    }
  }
  

} // namespace pinocchio

#endif // ifndef __pinocchio_mod_rnea_derivatives_hxx__
