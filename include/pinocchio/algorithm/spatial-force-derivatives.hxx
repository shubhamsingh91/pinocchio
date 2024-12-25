//
// Copyright (c) 2017-2020 CNRS INRIA
//

#ifndef __pinocchio_spatial_force_derivatives_hxx__
#define __pinocchio_spatial_force_derivatives_hxx__

#include "pinocchio/multibody/visitor.hpp"
#include "pinocchio/algorithm/check.hpp"
#include "pinocchio/utils/tensor_utils.hpp"

#include <stdio.h>
#include <vector>

namespace pinocchio {

template <typename Scalar, int Options, template <typename, int> class JointCollectionTpl, typename ConfigVectorType,
    typename TangentVectorType1, typename TangentVectorType2>
struct computeSpatialForceDerivsForwardStep
: public fusion::JointUnaryVisitorBase<computeSpatialForceDerivsForwardStep<Scalar, Options, JointCollectionTpl,
      ConfigVectorType, TangentVectorType1, TangentVectorType2>>
{
    typedef ModelTpl<Scalar, Options, JointCollectionTpl> Model;
    typedef DataTpl<Scalar, Options, JointCollectionTpl> Data;

    typedef boost::fusion::vector<const Model&, Data&, const ConfigVectorType&, const TangentVectorType1&,
        const TangentVectorType2&>
        ArgsType;

    template <typename JointModel>
    static void algo(const JointModelBase<JointModel>& jmodel,
        JointDataBase<typename JointModel::JointDataDerived>& jdata, const Model& model, Data& data,
        const Eigen::MatrixBase<ConfigVectorType>& q, const Eigen::MatrixBase<TangentVectorType1>& v,
        const Eigen::MatrixBase<TangentVectorType2>& a)
    {
        typedef typename Model::JointIndex JointIndex;
        typedef typename Data::Motion Motion;
        typedef typename Data::Inertia Inertia;
        typedef typename Data::Coriolis Coriolis;

        const JointIndex& i = jmodel.id();
        const JointIndex& parent = model.parents[i];
        Motion& ov = data.ov[i];
        Motion& oa = data.oa[i];
        Motion& vJ = data.vJ[i];

        jmodel.calc(jdata.derived(), q.derived(), v.derived());

        data.liMi[i] = model.jointPlacements[i] * jdata.M();

        if (parent > 0) {
            data.oMi[i] = data.oMi[parent] * data.liMi[i];
            ov = data.ov[parent];
            oa = data.oa[parent];
        } else {
            data.oMi[i] = data.liMi[i];
            ov.setZero();
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

        // dJ
        motionSet::motionAction(ov, J_cols, dJ_cols); // psi_dot = v(p(i)) x S_i

        // ddJ
        motionSet::motionAction(oa, J_cols, ddJ_cols);
        motionSet::motionAction<ADDTO>(ov, dJ_cols, ddJ_cols); // psi_dotdot = a(p(i)) x S_i + v(p(i)) x psi_dot

        // vdJ
        motionSet::motionAction(vJ, J_cols, vdJ_cols); // vdj = vJ x S_i + 2 psi_dot
        vdJ_cols.noalias() += dJ_cols + dJ_cols;

        // velocity and accelaration finishing
        ov += vJ;
        oa += (ov ^ vJ) + data.oMi[i].act(jdata.S() * jmodel.jointVelocitySelector(a) + jdata.c());

        // joint frame variables here

        data.a[i] = data.oMi[i].actInv(oa + model.gravity); // a in joint frame
        data.v[i] = data.oMi[i].actInv(ov);                 // v in joint frame

        // Composite rigid body inertia
        Inertia& oY = data.oYcrb[i];

        oY = data.oMi[i].act(model.inertias[i]);
        data.of[i] = oY * oa + oY.vxiv(ov);

        data.oBcrb[i] = Coriolis(oY, ov);
    }
};

template <typename Scalar, int Options, template <typename, int> class JointCollectionTpl>
struct computeSpatialForceDerivsBackwardStep
: public fusion::JointUnaryVisitorBase<computeSpatialForceDerivsBackwardStep<Scalar, Options, JointCollectionTpl>>
{
    typedef ModelTpl<Scalar, Options, JointCollectionTpl> Model;
    typedef DataTpl<Scalar, Options, JointCollectionTpl> Data;

    typedef boost::fusion::vector<const Model&, Data&, Eigen::Tensor<double,3>&>
        ArgsType;

    template <typename JointModel>
    static void algo(const JointModelBase<JointModel>& jmodel, const Model& model, Data& data,
        Eigen::Tensor<double,3> & df_dq)
    {
        typedef typename Model::JointIndex JointIndex;
        typedef typename Data::Force Force;

        const JointIndex& i = jmodel.id();
        std::cout << "Joint Index i : " << i << std::endl;
        const JointIndex& parent = model.parents[i];

        typedef typename SizeDepType<JointModel::NV>::template ColsReturn<typename Data::Matrix6x>::Type ColsBlock;
        typedef typename Data::Matrix6 Matrix6;
        typedef typename Data::Vector6c Vector6c;

        ColsBlock J_cols = jmodel.jointCols(data.J);    // S_i
        ColsBlock dJ_cols = jmodel.jointCols(data.dJ);  // psi_dot_i
        ColsBlock ddJ_cols = jmodel.jointCols(data.ddJ);// psi_dotdot_i
        ColsBlock vdJ_cols = jmodel.jointCols(data.vdJ);

        ColsBlock tmp1 = jmodel.jointCols(data.Ftmp1);
        ColsBlock tmp2 = jmodel.jointCols(data.Ftmp2);
        ColsBlock tmp3 = jmodel.jointCols(data.Ftmp3); // tmp3 is for this joint only, Ftmp3 is for the full body
        ColsBlock tmp4 = jmodel.jointCols(data.Ftmp4);

        Matrix6 mat6tmp1;
        mat6tmp1.setZero();
        Vector6c vec6tmp1, vec6tmp2, vec6tmp3, vec6tmp4;

        const Eigen::Index joint_idx = (Eigen::Index)jmodel.idx_v();
        const Eigen::Index joint_dofs = (Eigen::Index)jmodel.nv();
        const Eigen::Index subtree_dofs = (Eigen::Index)data.nvSubtree[i];
        const Eigen::Index successor_idx = joint_idx + joint_dofs;
        const Eigen::Index successor_dofs = subtree_dofs - joint_dofs;

        JointIndex j = i;

         while (j > 0) {

          std::cout << "Joint Index j : " << j << std::endl;
          auto J_j = data.J.col(j);  // S_j
          auto psidot_j = data.dJ.col(j); // psi_dot_j
          auto psiddot_j = data.ddJ.col(j); // psi_dotdot_j
          auto Ic_j = data.oYcrb[j]; // Ic_j
          auto Bc_j = data.oBcrb[j]; // Bc_j

          motionSet::inertiaAction(data.oYcrb[i], psiddot_j, vec6tmp1); // IiC * psi_dotdot_j
          motionSet::coriolisAction(data.oBcrb[i], psidot_j, vec6tmp2); // Bic * psi_dot_j  
          Force ofi = data.of[i]; // f_i


          addForceCrossMatrix(ofi, mat6tmp1); //  f_i
           vec6tmp3 = mat6tmp1* J_j;  

           vec6tmp4 =  vec6tmp1 + vec6tmp2 + vec6tmp3; // IiC * psi_dotdot_j + Bic * psi_dot_j + cmf_bar(f_i)*S_j
           tens_assign6_col(df_dq,vec6tmp4,j,i); // dfci_dqj
           
        
            j = model.parents[j];
          }

        
        if (parent > 0) {
            data.oYcrb[parent] += data.oYcrb[i];
            data.oBcrb[parent] += data.oBcrb[i];
            data.of[parent] += data.of[i];
        }
    }

  template <typename ForceDerived, typename M6>
  static void addForceCrossMatrix(const ForceDense<ForceDerived> &f,
                                  const Eigen::MatrixBase<M6> &mout) {
    M6 &mout_ = PINOCCHIO_EIGEN_CONST_CAST(M6, mout);
    addSkew(-f.linear(), mout_.template block<3, 3>(ForceDerived::LINEAR,
                                                    ForceDerived::ANGULAR));
    addSkew(-f.linear(), mout_.template block<3, 3>(ForceDerived::ANGULAR,
                                                    ForceDerived::LINEAR));
    addSkew(-f.angular(), mout_.template block<3, 3>(ForceDerived::ANGULAR,
                                                     ForceDerived::ANGULAR));
  }

};


template <typename Scalar, int Options, template <typename, int> class JointCollectionTpl, typename ConfigVectorType,
    typename TangentVectorType1, typename TangentVectorType2>
inline void computeSpatialForceDerivs(const ModelTpl<Scalar, Options, JointCollectionTpl>& model,
    DataTpl<Scalar, Options, JointCollectionTpl>& data, const Eigen::MatrixBase<ConfigVectorType>& q,
    const Eigen::MatrixBase<TangentVectorType1>& v, const Eigen::MatrixBase<TangentVectorType2>& a,
    Eigen::Tensor<double,3>& df_dq)
{
    PINOCCHIO_CHECK_ARGUMENT_SIZE(q.size(), model.nq, "The joint configuration vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(v.size(), model.nv, "The joint velocity vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(a.size(), model.nv, "The joint acceleration vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(df_dq.dimension(0), 6);
    PINOCCHIO_CHECK_ARGUMENT_SIZE(df_dq.dimension(1), model.nv);
    PINOCCHIO_CHECK_ARGUMENT_SIZE(df_dq.dimension(2), model.nv);

    assert(model.check(data) && "data is not consistent with model.");
    assert(model.nq == model.nv);

    typedef ModelTpl<Scalar, Options, JointCollectionTpl> Model;
    typedef typename Model::JointIndex JointIndex;

    typedef computeSpatialForceDerivsForwardStep<Scalar, Options, JointCollectionTpl, ConfigVectorType,
        TangentVectorType1, TangentVectorType2>
        Pass1;
    for (JointIndex i = 1; i < (JointIndex)model.njoints; ++i) { // goes from 1 to 29
        Pass1::run(model.joints[i], data.joints[i],
            typename Pass1::ArgsType(model, data, q.derived(), v.derived(), a.derived()));
    }

    typedef computeSpatialForceDerivsBackwardStep<Scalar, Options, JointCollectionTpl>
        Pass2;
    for (JointIndex i = (JointIndex)(model.njoints - 1); i > 0; --i) { // i from 29 to 1
        Pass2::run(model.joints[i],
            typename Pass2::ArgsType(model, data, df_dq));
    }
}

} // namespace pinocchio

#endif // ifndef __pinocchio_spatial_force_derivatives_hxx__
