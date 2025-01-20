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
        ColsBlock S_i = jmodel.jointCols(data.J); // data.J has all the phi (in ground frame) stacked in columns
        ColsBlock psid_i = jmodel.jointCols(data.psid);   // psid_i is the psi_dot in ground frame
        ColsBlock psidd_i = jmodel.jointCols(data.psidd); // psidd_i is the psi_dotdot in ground frame
        ColsBlock phidot_i = jmodel.jointCols(data.dJ);       // This here is phi_dot in ground frame

        S_i.noalias() = data.oMi[i].act(jdata.S()); // S_i is just the phi in ground frame for a joint
        vJ = data.oMi[i].act(jdata.v());
        motionSet::motionAction(ov, S_i, psid_i);            // This ov here is v(p(i)), psi_dot calcs
        motionSet::motionAction(oa, S_i, psidd_i);           // This oa here is a(p(i)) , psi_dotdot calcs
        motionSet::motionAction<ADDTO>(ov, psid_i, psidd_i); // This ov here is v(p(i)) , psi_dotdot calcs
        ov += vJ;
        oa += (ov ^ vJ) + data.oMi[i].act(jdata.S() * jmodel.jointVelocitySelector(a) + jdata.c());
        motionSet::motionAction(ov, S_i, phidot_i); // This here is phi_dot, here ov used is v(p(i)) + vJ
                                                      // Composite rigid body inertia
        Inertia& oY = data.oYcrb[i];

        oY = data.oMi[i].act(model.inertias[i]);
        data.oh[i] = oY * ov;

        data.of[i] = oY * oa + oY.vxiv(ov); // f_i in ground frame
        data.oBcrb[i] = Coriolis(oY, ov); // BC{i}

    }
};

template <typename Scalar, int Options, template <typename, int> class JointCollectionTpl>
struct computeSpatialForceDerivsBackwardStep
: public fusion::JointUnaryVisitorBase<computeSpatialForceDerivsBackwardStep<Scalar, Options, JointCollectionTpl>>
{
    typedef ModelTpl<Scalar, Options, JointCollectionTpl> Model;
    typedef DataTpl<Scalar, Options, JointCollectionTpl> Data;

    typedef boost::fusion::vector<const Model&, Data&, Eigen::Tensor<double,3>&, Eigen::Tensor<double,3>&,
     Eigen::Tensor<double,3>&>
        ArgsType;

    template <typename JointModel>
    static void algo(const JointModelBase<JointModel>& jmodel, const Model& model, Data& data,
        Eigen::Tensor<double,3> & df_dq, Eigen::Tensor<double,3> & df_dv, Eigen::Tensor<double,3> & df_da)
    {
        typedef typename Model::JointIndex JointIndex;
        typedef typename Data::Force Force;
        typedef typename Motion::ActionMatrixType ActionMatrixType;

        const JointIndex& i = jmodel.id();
        const JointIndex& parent = model.parents[i];

        typedef typename SizeDepType<JointModel::NV>::template ColsReturn<typename Data::Matrix6x>::Type ColsBlock;
        typedef typename Data::Matrix6 Matrix6;
        typedef typename Data::Vector6c Vector6c;

        Matrix6 mat6tmp1;
        mat6tmp1.setZero();
        Vector6c vec6tmp1, vec6tmp2, vec6tmp3, vec6tmp4;
        Vector6c S_j, psidot_j, psiddot_j, phidot_j;
        Vector6c S_i, psidot_i, psiddot_i, phidot_i;

        S_i = data.J.col(i-1);  // S_i
        psidot_i = data.psid.col(i-1); // psi_dot_i
        psiddot_i = data.psidd.col(i-1); // psi_dotdot_i
        phidot_i = data.dJ.col(i-1); // phi_dot_i
        Force ofci = data.of[i]; // fc_i
        Force ftmp1;

        JointIndex j = i;

         while (j > 0) {
            const MotionRef<typename Data::Matrix6x::ColXpr> S_jc = data.J.col(j-1);
            S_j = data.J.col(j-1);  // S_j
            psidot_j = data.psid.col(j-1); // psi_dot_j
            psiddot_j = data.psidd.col(j-1); // psi_dotdot_j
            phidot_j = data.dJ.col(j-1); // phi_dot_j
            auto Ic_j = data.oYcrb[j]; // Ic_j
            auto Bc_j = data.oBcrb[j]; // Bc_j
            const ActionMatrixType crfSt = S_jc.toDualActionMatrix();                             //(S{i}(:,p) )x* matrix

            mat6tmp1.setZero();
            motionSet::inertiaAction(data.oYcrb[i], psiddot_j, vec6tmp1); // IiC * psi_dotdot_j
            motionSet::coriolisAction(data.oBcrb[i], psidot_j, vec6tmp2); // Bic * psi_dot_j  

            addForceCrossMatrix(ofci, mat6tmp1); //  cmf_bar(fc_i)
            vec6tmp3 = mat6tmp1* S_j;   
            vec6tmp4 = -crfSt * ofci.toVector(); // - S_j x* 0_fc_i
            ftmp1.toVector() = vec6tmp1 + vec6tmp2 + vec6tmp3 + vec6tmp4;
            tens_assign6_col(df_dq,data.oMi[i].actInv(ftmp1).toVector(),j-1,i-1); // dfci_dqj

            // partials w.r.t v
            motionSet::inertiaAction(data.oYcrb[i], (psidot_j + phidot_j) , vec6tmp1); // 
            motionSet::coriolisAction(data.oBcrb[i], S_j, vec6tmp2); // Bic * psi_dot_j 
            ftmp1.toVector() = vec6tmp1 + vec6tmp2;
            tens_assign6_col(df_dv,data.oMi[i].actInv(ftmp1).toVector(),j-1,i-1); // dfci_dv

            // partials w.r.t a
            motionSet::inertiaAction(data.oYcrb[i], S_j , vec6tmp1); 
            ftmp1.toVector() = vec6tmp1;
            tens_assign6_col(df_da,data.oMi[i].actInv(ftmp1).toVector(),j-1,i-1); // dfci_da

            if (j !=i){ // for the case j>i

                motionSet::inertiaAction(data.oYcrb[i], psiddot_i, vec6tmp1); // 
                motionSet::coriolisAction(data.oBcrb[i], psidot_i, vec6tmp2); // 
                mat6tmp1.setZero();
                addForceCrossMatrix(ofci, mat6tmp1); // 
                vec6tmp3 = mat6tmp1* S_i;  
                ftmp1.toVector() = vec6tmp1 + vec6tmp2 + vec6tmp3;
                tens_assign6_col(df_dq,data.oMi[j].actInv(ftmp1).toVector(),i-1,j-1); // dfCj_dqi

                // partials w.r.t v
                motionSet::inertiaAction(data.oYcrb[i], (psidot_i + phidot_i) , vec6tmp1); // 
                motionSet::coriolisAction(data.oBcrb[i], S_i, vec6tmp2); // Bic * psi_dot_j 
                ftmp1.toVector() = vec6tmp1 + vec6tmp2;
                tens_assign6_col(df_dv,data.oMi[j].actInv(ftmp1).toVector(),i-1,j-1); // dfcj_dvi
  
                // partials w.r.t a
                motionSet::inertiaAction(data.oYcrb[i], S_i , vec6tmp1);
                ftmp1.toVector() = vec6tmp1;
                tens_assign6_col(df_da,data.oMi[j].actInv(ftmp1).toVector(),i-1,j-1); // dfcj_dai

            }
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
    Eigen::Tensor<double,3>& df_dq, Eigen::Tensor<double,3>& df_dv, Eigen::Tensor<double,3>& df_da)
{
    PINOCCHIO_CHECK_ARGUMENT_SIZE(q.size(), model.nq, "The joint configuration vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(v.size(), model.nv, "The joint velocity vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(a.size(), model.nv, "The joint acceleration vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(df_dq.dimension(0), 6);
    PINOCCHIO_CHECK_ARGUMENT_SIZE(df_dq.dimension(1), model.nv);
    PINOCCHIO_CHECK_ARGUMENT_SIZE(df_dq.dimension(2), model.nv);

    PINOCCHIO_CHECK_ARGUMENT_SIZE(df_dv.dimension(0), 6);
    PINOCCHIO_CHECK_ARGUMENT_SIZE(df_dv.dimension(1), model.nv);
    PINOCCHIO_CHECK_ARGUMENT_SIZE(df_dv.dimension(2), model.nv);

    PINOCCHIO_CHECK_ARGUMENT_SIZE(df_da.dimension(0), 6);
    PINOCCHIO_CHECK_ARGUMENT_SIZE(df_da.dimension(1), model.nv);
    PINOCCHIO_CHECK_ARGUMENT_SIZE(df_da.dimension(2), model.nv);

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
            typename Pass2::ArgsType(model, data, df_dq, df_dv, df_da));
    }
}

} // namespace pinocchio

#endif // ifndef __pinocchio_spatial_force_derivatives_hxx__
