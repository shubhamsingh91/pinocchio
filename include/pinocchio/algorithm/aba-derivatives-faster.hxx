//
// Copyright (c) 2018-2020 CNRS INRIA
//

#ifndef __pinocchio_algorithm_aba_derivatives_faster_hxx__
#define __pinocchio_algorithm_aba_derivatives_faster_hxx__

#include "pinocchio/multibody/visitor.hpp"
#include "pinocchio/algorithm/check.hpp"
#include "pinocchio/algorithm/aba.hpp"
#include <iostream>

using std::cout;
using std::endl;

namespace pinocchio {

template <typename Scalar, int Options, template <typename, int> class JointCollectionTpl, typename ConfigVectorType,
    typename TangentVectorType>
struct ComputeABADerivativesFasterForwardStep1
: public fusion::JointUnaryVisitorBase<
      ComputeABADerivativesFasterForwardStep1<Scalar, Options, JointCollectionTpl, ConfigVectorType, TangentVectorType>>
{
    typedef ModelTpl<Scalar, Options, JointCollectionTpl> Model;
    typedef DataTpl<Scalar, Options, JointCollectionTpl> Data;

    typedef boost::fusion::vector<const Model&, Data&, const ConfigVectorType&, const TangentVectorType&> ArgsType;

    template <typename JointModel>
    static void algo(const JointModelBase<JointModel>& jmodel,
        JointDataBase<typename JointModel::JointDataDerived>& jdata, const Model& model, Data& data,
        const Eigen::MatrixBase<ConfigVectorType>& q, const Eigen::MatrixBase<TangentVectorType>& v)
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
        motionSet::motionAction(ov, J_cols, dJ_cols);

        // ddJ
        motionSet::motionAction(oa, J_cols, ddJ_cols);
        motionSet::motionAction<ADDTO>(ov, dJ_cols, ddJ_cols);

        // vdJ
        motionSet::motionAction(vJ, J_cols, vdJ_cols);
        vdJ_cols.noalias() += dJ_cols + dJ_cols;

        // velocity and accelaration finishing
        ov += vJ;
        oa += (ov ^ vJ) + data.oMi[i].act(jdata.S() * jmodel.jointVelocitySelector(data.ddq) + jdata.c());

        // joint frame variables here

        data.a[i] = data.oMi[i].actInv(oa + model.gravity); // a in joint frame
        data.v[i] = data.oMi[i].actInv(ov);                 // v in joint frame

        // Composite rigid body inertia
        Inertia& oY = data.oYcrb[i];

        oY = data.oMi[i].act(model.inertias[i]);
        data.of[i] = oY * oa + oY.vxiv(ov);

        data.oBcrb[i] = Coriolis(oY, ov);
        data.Yaba[i] = model.inertias[i].matrix();
    }
};

template <typename Scalar, int Options, template <typename, int> class JointCollectionTpl, typename MatrixType>
struct ComputeABADerivativesFasterBackwardStep1
: public fusion::JointUnaryVisitorBase<
      ComputeABADerivativesFasterBackwardStep1<Scalar, Options, JointCollectionTpl, MatrixType>>
{
    typedef ModelTpl<Scalar, Options, JointCollectionTpl> Model;
    typedef DataTpl<Scalar, Options, JointCollectionTpl> Data;

    typedef boost::fusion::vector<const Model&, Data&, MatrixType&> ArgsType;

    template <typename JointModel>
    static void algo(const JointModelBase<JointModel>& jmodel,
        JointDataBase<typename JointModel::JointDataDerived>& jdata, const Model& model, Data& data,
        const Eigen::MatrixBase<MatrixType>& Minv)
    {
        typedef typename Model::JointIndex JointIndex;
        typedef typename Data::Inertia Inertia;

        MatrixType& Minv_ = PINOCCHIO_EIGEN_CONST_CAST(MatrixType, Minv);

        const JointIndex& i = jmodel.id();
        const JointIndex& parent = model.parents[i];
        const JointIndex& j_idx = jmodel.idx_v();

        const Eigen::Index joint_dofs = (Eigen::Index)jmodel.nv();
        const Eigen::Index subtree_dofs = (Eigen::Index)data.nvSubtree[i];
        const Eigen::Index successor_idx = j_idx + joint_dofs;
        const Eigen::Index successor_dofs = subtree_dofs - joint_dofs;

        typename Inertia::Matrix6& Ia = data.Yaba[i];
        typename Data::Matrix6x& Fcrb = data.Fcrb[0];
        typename Data::Matrix6x& FcrbTmp = data.Fcrb.back();

        jmodel.calc_aba(jdata.derived(), Ia, parent > 0);

        typedef typename SizeDepType<JointModel::NV>::template ColsReturn<typename Data::Matrix6x>::Type ColsBlock;

        ColsBlock J_cols = jmodel.jointCols(data.J);
        ColsBlock dJ_cols = jmodel.jointCols(data.dJ);
        ColsBlock ddJ_cols = jmodel.jointCols(data.ddJ);
        ColsBlock vdJ_cols = jmodel.jointCols(data.vdJ);

        ColsBlock U_cols = jmodel.jointCols(data.IS);
        forceSet::se3Action(data.oMi[i], jdata.U(), U_cols); // expressed in the world frame

        Minv_.block(j_idx, j_idx, jmodel.nv(), jmodel.nv()) = jdata.Dinv();
        const int nv_children = data.nvSubtree[i] - jmodel.nv();
        if (nv_children > 0) {
            ColsBlock SDinv_cols = jmodel.jointCols(data.SDinv);
            SDinv_cols.noalias() = J_cols * jdata.Dinv();

            Minv_.block(j_idx, j_idx + jmodel.nv(), jmodel.nv(), nv_children).noalias()
                = -SDinv_cols.transpose() * Fcrb.middleCols(j_idx + jmodel.nv(), nv_children);

            if (parent > 0) {
                FcrbTmp.leftCols(data.nvSubtree[i]).noalias()
                    = U_cols * Minv_.block(j_idx, j_idx, jmodel.nv(), data.nvSubtree[i]);
                Fcrb.middleCols(j_idx, data.nvSubtree[i]) += FcrbTmp.leftCols(data.nvSubtree[i]);
            }
        } else {
            Fcrb.middleCols(j_idx, data.nvSubtree[i]).noalias()
                = U_cols * Minv_.block(j_idx, j_idx, jmodel.nv(), data.nvSubtree[i]);
        }

        ColsBlock tmp1 = jmodel.jointCols(data.Ftmp1);
        ColsBlock tmp2 = jmodel.jointCols(data.Ftmp2);
        ColsBlock tmp3 = jmodel.jointCols(data.Ftmp3);
        ColsBlock tmp4 = jmodel.jointCols(data.Ftmp4);

        typename Data::MatrixXs& rnea_partial_dq = data.dtau_dq;
        typename Data::MatrixXs& rnea_partial_dv = data.dtau_dv;

        jmodel.jointVelocitySelector(data.tau).noalias() = J_cols.transpose() * data.of[i].toVector();

        motionSet::inertiaAction(data.oYcrb[i], J_cols, tmp1);

        motionSet::coriolisAction(data.oBcrb[i], J_cols, tmp2);
        motionSet::inertiaAction<ADDTO>(data.oYcrb[i], vdJ_cols, tmp2);

        motionSet::coriolisAction(data.oBcrb[i], dJ_cols, tmp3);
        motionSet::inertiaAction<ADDTO>(data.oYcrb[i], ddJ_cols, tmp3);

        motionSet::act<ADDTO>(J_cols, data.of[i], tmp3);
        motionSet::coriolisTransposeAction(data.oBcrb[i], J_cols, tmp4);

        if (successor_dofs > 0) {

            rnea_partial_dq.block(j_idx, successor_idx, joint_dofs, successor_dofs).noalias()
                = J_cols.transpose() * data.Ftmp3.middleCols(successor_idx, successor_dofs);

            rnea_partial_dv.block(j_idx, successor_idx, joint_dofs, successor_dofs).noalias()
                = J_cols.transpose() * data.Ftmp2.middleCols(successor_idx, successor_dofs);
        }

        rnea_partial_dq.block(j_idx, j_idx, subtree_dofs, joint_dofs).noalias()
            = data.Ftmp1.middleCols(j_idx, subtree_dofs).transpose() * ddJ_cols
              + data.Ftmp4.middleCols(j_idx, subtree_dofs).transpose() * dJ_cols;

        rnea_partial_dv.block(j_idx, j_idx, subtree_dofs, joint_dofs).noalias()
            = data.Ftmp1.middleCols(j_idx, subtree_dofs).transpose() * vdJ_cols
              + data.Ftmp4.middleCols(j_idx, subtree_dofs).transpose() * J_cols;

        if (parent > 0) {
            data.Yaba[parent] += internal::SE3actOn<Scalar>::run(data.liMi[i], Ia);
            data.oYcrb[parent] += data.oYcrb[i];
            data.oBcrb[parent] += data.oBcrb[i];
            data.of[parent] += data.of[i];
        }
    }
};

template <typename Scalar, int Options, template <typename, int> class JointCollectionTpl, typename MatrixType>
struct ComputeABADerivativesFasterForwardStep2
: public fusion::JointUnaryVisitorBase<
      ComputeABADerivativesFasterForwardStep2<Scalar, Options, JointCollectionTpl, MatrixType>>
{
    typedef ModelTpl<Scalar, Options, JointCollectionTpl> Model;
    typedef DataTpl<Scalar, Options, JointCollectionTpl> Data;

    typedef boost::fusion::vector<const Model&, Data&, MatrixType&> ArgsType;

    template <typename JointModel>
    static void algo(const JointModelBase<JointModel>& jmodel,
        JointDataBase<typename JointModel::JointDataDerived>& jdata, const Model& model, Data& data,
        const MatrixType& Minv)
    {
        typedef typename Model::JointIndex JointIndex;
        MatrixType& Minv_ = PINOCCHIO_EIGEN_CONST_CAST(MatrixType, Minv);

        const JointIndex& i = jmodel.id();
        const JointIndex& parent = model.parents[i];
        typename Data::Matrix6x& FcrbTmp = data.Fcrb.back();
        const JointIndex& j_idx = jmodel.idx_v();

        typedef typename SizeDepType<JointModel::NV>::template ColsReturn<typename Data::Matrix6x>::Type ColsBlock;
        ColsBlock UDinv_cols = jmodel.jointCols(data.UDinv);
        forceSet::se3Action(data.oMi[i], jdata.UDinv(), UDinv_cols); // expressed in the world frame
        ColsBlock J_cols = jmodel.jointCols(data.J);

        if (parent > 0) {
            FcrbTmp.topRows(jmodel.nv()).rightCols(model.nv - j_idx).noalias()
                = UDinv_cols.transpose() * data.Fcrb[parent].rightCols(model.nv - j_idx);
            Minv_.middleRows(j_idx, jmodel.nv()).rightCols(model.nv - j_idx)
                -= FcrbTmp.topRows(jmodel.nv()).rightCols(model.nv - j_idx);
        }

        data.Fcrb[i].rightCols(model.nv - j_idx).noalias()
            = J_cols * Minv_.middleRows(j_idx, jmodel.nv()).rightCols(model.nv - j_idx);
        if (parent > 0)
            data.Fcrb[i].rightCols(model.nv - j_idx) += data.Fcrb[parent].rightCols(model.nv - j_idx);
    }
};
template <typename Scalar, int Options, template <typename, int> class JointCollectionTpl, typename ConfigVectorType,
    typename TangentVectorType1, typename TangentVectorType2, typename MatrixType1, typename MatrixType2,
    typename MatrixType3>
inline void computeABADerivativesFaster(const ModelTpl<Scalar, Options, JointCollectionTpl>& model,
    DataTpl<Scalar, Options, JointCollectionTpl>& data, const Eigen::MatrixBase<ConfigVectorType>& q,
    const Eigen::MatrixBase<TangentVectorType1>& v, const Eigen::MatrixBase<TangentVectorType2>& tau,
    const Eigen::MatrixBase<MatrixType1>& aba_partial_dq, const Eigen::MatrixBase<MatrixType2>& aba_partial_dv,
    const Eigen::MatrixBase<MatrixType3>& aba_partial_dtau)
{
    PINOCCHIO_CHECK_ARGUMENT_SIZE(q.size(), model.nq, "The joint configuration vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(v.size(), model.nv, "The joint velocity vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(tau.size(), model.nv, "The joint torque vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(aba_partial_dq.cols(), model.nv);
    PINOCCHIO_CHECK_ARGUMENT_SIZE(aba_partial_dq.rows(), model.nv);
    PINOCCHIO_CHECK_ARGUMENT_SIZE(aba_partial_dv.cols(), model.nv);
    PINOCCHIO_CHECK_ARGUMENT_SIZE(aba_partial_dv.rows(), model.nv);
    PINOCCHIO_CHECK_ARGUMENT_SIZE(aba_partial_dtau.cols(), model.nv);
    PINOCCHIO_CHECK_ARGUMENT_SIZE(aba_partial_dtau.rows(), model.nv);
    assert(model.check(data) && "data is not consistent with model.");

    typedef typename ModelTpl<Scalar, Options, JointCollectionTpl>::JointIndex JointIndex;

    data.u = tau;
    data.ddq = aba(model, data, q.derived(), v.derived(), tau.derived());

    MatrixType3& Minv_ = PINOCCHIO_EIGEN_CONST_CAST(MatrixType3, aba_partial_dtau);
    Minv_.template triangularView<Eigen::Upper>().setZero();

    /// First, compute Minv and a, the joint acceleration vector
    typedef ComputeABADerivativesFasterForwardStep1<Scalar, Options, JointCollectionTpl, ConfigVectorType,
        TangentVectorType1>
        Pass1;
    for (JointIndex i = 1; i < (JointIndex)model.njoints; ++i) {
        Pass1::run(model.joints[i], data.joints[i], typename Pass1::ArgsType(model, data, q.derived(), v.derived()));
    }

    data.Fcrb[0].setZero();
    typedef ComputeABADerivativesFasterBackwardStep1<Scalar, Options, JointCollectionTpl, MatrixType3> Pass2;
    for (JointIndex i = (JointIndex)(model.njoints - 1); i > 0; --i) {
        Pass2::run(model.joints[i], data.joints[i], typename Pass2::ArgsType(model, data, Minv_));
    }

    typedef ComputeABADerivativesFasterForwardStep2<Scalar, Options, JointCollectionTpl, MatrixType3> Pass3;
    for (JointIndex i = 1; i < (JointIndex)model.njoints; ++i) {
        Pass3::run(model.joints[i], data.joints[i], typename Pass3::ArgsType(model, data, Minv_));
    }

    Minv_.template triangularView<Eigen::StrictlyLower>()
        = Minv_.transpose().template triangularView<Eigen::StrictlyLower>();

    PINOCCHIO_EIGEN_CONST_CAST(MatrixType1, aba_partial_dq).noalias() = -Minv_ * data.dtau_dq;
    PINOCCHIO_EIGEN_CONST_CAST(MatrixType2, aba_partial_dv).noalias() = -Minv_ * data.dtau_dv;
}

template <typename Scalar, int Options, template <typename, int> class JointCollectionTpl, typename ConfigVectorType,
    typename TangentVectorType1, typename TangentVectorType2, typename MatrixType1, typename MatrixType2,
    typename MatrixType3>
inline void computeABADerivativesFaster(const ModelTpl<Scalar, Options, JointCollectionTpl>& model,
    DataTpl<Scalar, Options, JointCollectionTpl>& data, const Eigen::MatrixBase<ConfigVectorType>& q,
    const Eigen::MatrixBase<TangentVectorType1>& v, const Eigen::MatrixBase<TangentVectorType2>& tau,
    const container::aligned_vector<ForceTpl<Scalar, Options>>& fext,
    const Eigen::MatrixBase<MatrixType1>& aba_partial_dq, const Eigen::MatrixBase<MatrixType2>& aba_partial_dv,
    const Eigen::MatrixBase<MatrixType3>& aba_partial_dtau)
{
    PINOCCHIO_CHECK_ARGUMENT_SIZE(q.size(), model.nq, "The joint configuration vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(v.size(), model.nv, "The joint velocity vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(tau.size(), model.nv, "The joint torque vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(
        fext.size(), (size_t)model.njoints, "The external forces vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(aba_partial_dq.cols(), model.nv);
    PINOCCHIO_CHECK_ARGUMENT_SIZE(aba_partial_dq.rows(), model.nv);
    PINOCCHIO_CHECK_ARGUMENT_SIZE(aba_partial_dv.cols(), model.nv);
    PINOCCHIO_CHECK_ARGUMENT_SIZE(aba_partial_dv.rows(), model.nv);
    PINOCCHIO_CHECK_ARGUMENT_SIZE(aba_partial_dtau.cols(), model.nv);
    PINOCCHIO_CHECK_ARGUMENT_SIZE(aba_partial_dtau.rows(), model.nv);
    assert(model.check(data) && "data is not consistent with model.");

    typedef typename ModelTpl<Scalar, Options, JointCollectionTpl>::JointIndex JointIndex;

    data.u = tau;
    data.ddq = aba(model, data, q, v, tau);

    MatrixType3& Minv_ = PINOCCHIO_EIGEN_CONST_CAST(MatrixType3, aba_partial_dtau);
    Minv_.template triangularView<Eigen::Upper>().setZero();

    /// First, compute Minv and a, the joint acceleration vector
    typedef ComputeABADerivativesFasterForwardStep1<Scalar, Options, JointCollectionTpl, ConfigVectorType,
        TangentVectorType1>
        Pass1;
    for (JointIndex i = 1; i < (JointIndex)model.njoints; ++i) {
        Pass1::run(model.joints[i], data.joints[i], typename Pass1::ArgsType(model, data, q.derived(), v.derived()));
        data.f[i] -= fext[i];
    }

    data.Fcrb[0].setZero();
    typedef ComputeABADerivativesFasterBackwardStep1<Scalar, Options, JointCollectionTpl, MatrixType3> Pass2;
    for (JointIndex i = (JointIndex)(model.njoints - 1); i > 0; --i) {
        Pass2::run(model.joints[i], data.joints[i], typename Pass2::ArgsType(model, data, Minv_));
    }

    typedef ComputeABADerivativesFasterForwardStep2<Scalar, Options, JointCollectionTpl, MatrixType3> Pass3;
    for (JointIndex i = 1; i < (JointIndex)model.njoints; ++i) {
        Pass3::run(model.joints[i], data.joints[i], typename Pass3::ArgsType(model, data, Minv_));
        data.of[i] -= data.oMi[i].act(fext[i]);
    }

    Minv_.template triangularView<Eigen::StrictlyLower>()
        = Minv_.transpose().template triangularView<Eigen::StrictlyLower>();

    PINOCCHIO_EIGEN_CONST_CAST(MatrixType1, aba_partial_dq).noalias() = -Minv_ * data.dtau_dq;
    PINOCCHIO_EIGEN_CONST_CAST(MatrixType2, aba_partial_dv).noalias() = -Minv_ * data.dtau_dv;
}

} // namespace pinocchio

#endif // ifndef __pinocchio_algorithm_aba_derivatives_hxx__
