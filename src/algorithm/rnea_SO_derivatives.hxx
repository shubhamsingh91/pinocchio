//
// Copyright (c) 2017-2020 CNRS INRIA
/*
/ RNEA SO derivatives algorithm
  Author- Shubham Singh singh281@utexas.edu
*/

#ifndef __pinocchio_rnea_SO_derivatives_hxx__
#define __pinocchio_rnea_SO_derivatives_hxx__

#include "pinocchio/algorithm/check.hpp"
#include "pinocchio/multibody/visitor.hpp"
#include "pinocchio/utils/tensor_utils.hpp"

#include <iostream>

using std::cout;
using std::endl;

namespace pinocchio {

template <typename Scalar, int Options,
          template <typename, int> class JointCollectionTpl,
          typename ConfigVectorType, typename TangentVectorType1,
          typename TangentVectorType2>
struct computeRNEA_SO_derivsForwardStep
    : public fusion::JointUnaryVisitorBase<computeRNEA_SO_derivsForwardStep<
          Scalar, Options, JointCollectionTpl, ConfigVectorType,
          TangentVectorType1, TangentVectorType2>> {
  typedef ModelTpl<Scalar, Options, JointCollectionTpl> Model;
  typedef DataTpl<Scalar, Options, JointCollectionTpl> Data;

  typedef boost::fusion::vector<const Model &, Data &, const ConfigVectorType &,
                                const TangentVectorType1 &,
                                const TangentVectorType2 &>
      ArgsType;

  template <typename JointModel>
  static void algo(const JointModelBase<JointModel> &jmodel,
                   JointDataBase<typename JointModel::JointDataDerived> &jdata,
                   const Model &model, Data &data,
                   const Eigen::MatrixBase<ConfigVectorType> &q,
                   const Eigen::MatrixBase<TangentVectorType1> &v,
                   const Eigen::MatrixBase<TangentVectorType2> &a) {
    typedef typename Model::JointIndex JointIndex;
    typedef typename Data::Motion Motion;
    typedef typename Data::Inertia Inertia;

    const JointIndex &i = jmodel.id();
    const JointIndex &parent = model.parents[i];
    Motion &ov = data.ov[i];
    Motion &oa = data.oa[i];
    Motion &vJ = data.vJ[i];

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

    typedef typename SizeDepType<JointModel::NV>::template ColsReturn<
        typename Data::Matrix6x>::Type ColsBlock;
    ColsBlock J_cols = jmodel.jointCols(data.J);
    ColsBlock dJ_cols = jmodel.jointCols(data.dJ);
    ColsBlock ddJ_cols = jmodel.jointCols(data.ddJ);
    ColsBlock phidJ_cols = jmodel.jointCols(data.phidJ);

    J_cols.noalias() = data.oMi[i].act(jdata.S());
    vJ = data.oMi[i].act(jdata.v());
    motionSet::motionAction(ov, J_cols, dJ_cols);
    motionSet::motionAction(oa, J_cols, ddJ_cols);
    motionSet::motionAction<ADDTO>(ov, dJ_cols, ddJ_cols);
    ov += vJ;
    oa += (ov ^ vJ) +
          data.oMi[i].act(jdata.S() * jmodel.jointVelocitySelector(a) +
                          jdata.c());
    motionSet::motionAction(ov, J_cols, phidJ_cols);
    Inertia &oY = data.oYcrb[i];

    oY = data.oMi[i].act(model.inertias[i]);
    data.of[i] = oY * oa + oY.vxiv(ov); // f_i in ground frame
    data.oBcrb[i] = Coriolis(oY, ov);   // B matrix in ground frame
  }
};

template <typename Scalar, int Options,
          template <typename, int> class JointCollectionTpl,
          typename tensortype1, typename tensortype2, typename tensortype3,
          typename tensortype4>
struct computeRNEA_SO_derivsBackwardStep
    : public fusion::JointUnaryVisitorBase<computeRNEA_SO_derivsBackwardStep<
          Scalar, Options, JointCollectionTpl, tensortype1, tensortype2,
          tensortype3, tensortype4>> {
  typedef ModelTpl<Scalar, Options, JointCollectionTpl> Model;
  typedef DataTpl<Scalar, Options, JointCollectionTpl> Data;

  typedef boost::fusion::vector<const Model &, Data &, const tensortype1 &,
                                const tensortype2 &, const tensortype3 &,
                                const tensortype4 &>
      ArgsType;

  template <typename JointModel>
  static void algo(const JointModelBase<JointModel> &jmodel, const Model &model,
                   Data &data, const tensortype1 &dtau_dq2,
                   const tensortype2 &dtau_dv2, const tensortype3 &dtau_dqdv,
                   const tensortype3 &M_FO) {
    typedef typename Model::JointIndex JointIndex;

    const JointIndex &i = jmodel.id();
    const JointIndex &parent = model.parents[i];
    JointIndex j, k;
    Eigen::Index joint_idx_j, joint_dofs_j, joint_idx_k, joint_dofs_k;

    typedef typename SizeDepType<JointModel::NV>::template ColsReturn<
        typename Data::Matrix6x>::Type ColsBlock;

    ColsBlock J_cols = jmodel.jointCols(data.J);
    ColsBlock dJ_cols = jmodel.jointCols(data.dJ);
    ColsBlock ddJ_cols = jmodel.jointCols(data.ddJ);
    ColsBlock phidJ_cols = jmodel.jointCols(data.phidJ);

    const Eigen::Index joint_idx = (Eigen::Index)jmodel.idx_v();
    const Eigen::Index joint_dofs =
        (Eigen::Index)jmodel.nv(); // no of joint DOFs

    Inertia &oYcrb = data.oYcrb[i];  // IC{i}
    Coriolis &oBcrb = data.oBcrb[i]; // BC{i}

    tensortype1 &dtau_dq2_ = const_cast<tensortype1 &>(dtau_dq2);
    tensortype2 &dtau_dv2_ = const_cast<tensortype2 &>(dtau_dv2);
    tensortype3 &dtau_dqdv_ = const_cast<tensortype3 &>(dtau_dqdv);
    tensortype4 &M_FO_ = const_cast<tensortype4 &>(M_FO);

    Motion &S_dm = data.S_dm[0];
    Motion &psid_dm = data.psid_dm[0];
    Motion &psidd_dm = data.psidd_dm[0];
    Force &F_var1 = data.f_var1[0];
    Motion &phid_dm = data.phid_dm[0];

    Eigen::Matrix<double, 6, 6> tempmat1, Jcols_j, dJ_cols_j, ddJ_cols_j,
        phidJ_cols_j, Jcols_k, dJ_cols_k, ddJ_cols_k, phidJ_cols_k;
    Eigen::Matrix<double, 1, 6> tempvec1, u1, u2, u11, u12;
    Eigen::Matrix<double, 6, 1> tempvec2, tempvec3, tempvec4, u3, u4, u5, u6,
        u7, u8, u9, u10, u13;
    double p1, p2, p3, p4, p5, p6;
    int ip, jq, kr;

    for (int p = 0; p < joint_dofs; p++) {
      ip = joint_idx + p;
      Coriolis &Bicphii = data.oBicphii[0];

      S_dm = J_cols.col(p);        // S{i}(:,p)
      psid_dm = dJ_cols.col(p);    // psi_dot for p DOF
      psidd_dm = ddJ_cols.col(p);  // psi_ddot for p DOF
      phid_dm = phidJ_cols.col(p); // phi_dot for p DOF

      data.oBicphii[0] = Coriolis(oYcrb, S_dm);      // Bic_phii matrix
      data.oBicpsidot[0] = Coriolis(oYcrb, psid_dm); // Bic_psii_dot matrix
      motionSet::inertiaAction(oYcrb, S_dm.toVector(),
                               data.ftemp1); // IC{i}S{i}(:,p)
      F_var1 = data.ftemp1;
      ForceCrossMatrix(F_var1, data.r0[0]); // cmf_bar(IC{i}S{i}(:,p))
      data.r1[0] = oYcrb.variation(S_dm);   // S{i}(p)x*IC{i} - IC{i} S{i}(p)x
      data.r2[0].noalias() = 2 * data.r0[0] - Bicphii.matrix_impl();
      data.r3[0].noalias() =
          (data.oBicpsidot[0]).matrix_impl() -
          ((S_dm.toActionMatrix_impl()).transpose()) * (oBcrb.matrix_impl()) -
          (oBcrb.matrix_impl()) * (S_dm.toActionMatrix_impl());
      motionSet::coriolisTransposeAction(oBcrb, S_dm.toVector(), data.ftemp1);
      F_var1 = data.ftemp1;
      ForceCrossMatrix(F_var1, data.r4[0]);
      tempmat1.noalias() = -((S_dm.toActionMatrix_impl()).transpose());
      motionSet::coriolisAction(oBcrb, psid_dm.toVector(), data.ftemp1);
      motionSet::inertiaAction<ADDTO>(oYcrb, psidd_dm.toVector(), data.ftemp1);
      data.ftemp1.noalias() += tempmat1 * data.of[i].toVector();
      F_var1 = data.ftemp1;
      ForceCrossMatrix(F_var1, data.r5[0]);
      // r6
      data.r6[0].noalias() = tempmat1 * oYcrb.matrix_impl() + data.r0[0];
      // r7
      motionSet::coriolisAction(oBcrb, S_dm.toVector(), data.ftemp1);
      motionSet::inertiaAction<ADDTO>(
          oYcrb, psid_dm.toVector() + phid_dm.toVector(), data.ftemp1);
      F_var1 = data.ftemp1;
      ForceCrossMatrix(F_var1, data.r7[0]);

      j = i;

      while (j > 0) {
        joint_idx_j = (Eigen::Index)(model.joints[j]).idx_v();
        joint_dofs_j = (Eigen::Index)(model.joints[j]).nv(); // no of joint DOFs
        Jcols_j = (model.joints[j]).jointCols(data.J);       //  S{j}
        dJ_cols_j = (model.joints[j]).jointCols(data.dJ);    //  psi_dot{j}
        ddJ_cols_j = (model.joints[j]).jointCols(data.ddJ);  //  psi_ddot{j}
        phidJ_cols_j = (model.joints[j]).jointCols(data.phidJ); //  phi_dot{j}

        for (int q = 0; q < joint_dofs_j; q++) {
          jq = joint_idx_j + q;
          S_dm = Jcols_j.col(q);                      // S{j}(:,q)
          psid_dm = dJ_cols_j.col(q);                 // psj_dot for q DOF
          psidd_dm = ddJ_cols_j.col(q);               //
          phid_dm = phidJ_cols_j.col(q);              //
          tempvec1 = ((S_dm.toVector()).transpose()); // (S{j}(:,q)).'
          tempvec2 = S_dm.toVector();                 // (S{j}(:,q))
          tempvec3 = psid_dm.toVector();              // psi_dot{j}(:,q)
          tempmat1 = (data.oBicphii[0]).matrix_impl();

          u1 = tempvec1 * data.r3[0];
          u2 = tempvec1 * data.r1[0];
          u3 = data.r3[0] * tempvec3 + data.r1[0] * psidd_dm.toVector() +
               data.r5[0] * tempvec2;
          u4 = data.r6[0] * tempvec2;
          u5 = data.r2[0] * tempvec3;
          motionSet::coriolisAction(data.oBicphii[0], tempvec3, u6);
          u6 += data.r7[0] * tempvec2;
          u7 = data.r3[0] * tempvec2 +
               data.r1[0] * (tempvec3 + phid_dm.toVector());
          u8 = data.r4[0] * tempvec2;
          u9 = data.r0[0] * tempvec2;
          motionSet::coriolisAction(data.oBicphii[0], tempvec2, u10);
          u11 = tempvec1 * tempmat1;
          u12 = tempvec3.transpose() * tempmat1;
          u13 = data.r1[0] * tempvec2;

          k = j;

          while (k > 0) {
            joint_idx_k = (Eigen::Index)(model.joints[k]).idx_v();
            joint_dofs_k =
                (Eigen::Index)(model.joints[k]).nv();      // no of joint DOFs
            Jcols_k = (model.joints[k]).jointCols(data.J); //  S{k}
            dJ_cols_k = (model.joints[k]).jointCols(data.dJ);   //  psi_dot{k}
            ddJ_cols_k = (model.joints[k]).jointCols(data.ddJ); //  psi_ddot{k}
            phidJ_cols_k =
                (model.joints[k]).jointCols(data.phidJ); //  phi_dot{k}

            for (int r = 0; r < joint_dofs_k; r++) {
              kr = joint_idx_k + r;
              S_dm = Jcols_k.col(r);                      // S{k}(:,r)
              psid_dm = dJ_cols_k.col(r);                 // psik_dot for r DOF
              psidd_dm = ddJ_cols_k.col(r);               //
              phid_dm = phidJ_cols_k.col(r);              //
              tempvec1 = ((S_dm.toVector()).transpose()); // (S{k}(:,r)).'
              tempvec2 = S_dm.toVector();                 // (S{k}(:,r))
              tempvec3 = psid_dm.toVector();              // psi_dot{k}(:,r)
              tempvec4 = phid_dm.toVector();
              p1 = u11 * tempvec3;
              p2 = (u9.transpose()) * psidd_dm.toVector();
              p2 += (-u12 + u8.transpose()) * tempvec3;

              dtau_dq2_(ip, jq, kr) = p2;
              dtau_dqdv_(ip, kr, jq) = -p1;

              if (j != i) {
                p3 = -u11 * tempvec2;
                p4 = tempvec1 * u13;
                dtau_dq2_(jq, kr, ip) = u1 * tempvec3;
                dtau_dq2_(jq, kr, ip) += u2 * psidd_dm.toVector();
                dtau_dq2_(jq, ip, kr) = dtau_dq2_(jq, kr, ip);
                dtau_dqdv_(jq, kr, ip) = p1;
                dtau_dqdv_(jq, ip, kr) = u1 * tempvec2;
                dtau_dqdv_(jq, ip, kr) += u2 * (tempvec3 + tempvec4);
                dtau_dv2_(jq, kr, ip) = -p3;
                dtau_dv2_(jq, ip, kr) = -p3;
                M_FO_(kr, jq, ip) = p4;
                M_FO_(jq, kr, ip) = p4;
              }

              if (k != j) {
                p3 = -u11 * tempvec2;
                p5 = tempvec1 * u9;
                dtau_dq2_(ip, kr, jq) = p2;
                dtau_dq2_(kr, ip, jq) = tempvec1 * u3;
                dtau_dv2_(ip, jq, kr) = p3;
                dtau_dv2_(ip, kr, jq) = p3;
                dtau_dqdv_(ip, jq, kr) = tempvec1 * (u5 + u8);
                dtau_dqdv_(ip, jq, kr) +=
                    u9.transpose() * (tempvec3 + tempvec4);
                dtau_dqdv_(kr, jq, ip) = tempvec1 * u6;
                M_FO_(kr, ip, jq) = p5;
                M_FO_(ip, kr, jq) = p5;
                if (j != i) {
                  p6 = tempvec1 * u10;
                  dtau_dq2_(kr, jq, ip) = dtau_dq2_(kr, ip, jq);
                  dtau_dv2_(kr, ip, jq) = p6;
                  dtau_dv2_(kr, jq, ip) = p6;
                  dtau_dqdv_(kr, ip, jq) = tempvec1 * u7;

                } else {
                  dtau_dv2_(kr, jq, ip) = tempvec1 * u4;
                }

              } else {
                dtau_dv2_(ip, jq, kr) = -u2 * tempvec2;
              }
            }

            k = model.parents[k];
          }
        }
        j = model.parents[j];
      }
    }

    if (parent > 0) {
      data.oYcrb[parent] += data.oYcrb[i];
      data.oBcrb[parent] += data.oBcrb[i];
      data.of[parent] += data.of[i];
    }
  }

  // Adding this function for cmf_bar operator
  template <typename ForceDerived, typename M6>
  static void ForceCrossMatrix(const ForceDense<ForceDerived> &f,
                               const Eigen::MatrixBase<M6> &mout) {
    M6 &mout_ = PINOCCHIO_EIGEN_CONST_CAST(M6, mout);
    mout_.template block<3, 3>(ForceDerived::LINEAR, ForceDerived::ANGULAR) =
        mout_.template block<3, 3>(ForceDerived::ANGULAR,
                                   ForceDerived::LINEAR) = skew(-f.linear());
    mout_.template block<3, 3>(ForceDerived::ANGULAR, ForceDerived::ANGULAR) =
        skew(-f.angular());
  }
};

template <typename Scalar, int Options,
          template <typename, int> class JointCollectionTpl,
          typename ConfigVectorType, typename TangentVectorType1,
          typename TangentVectorType2, typename tensortype1,
          typename tensortype2, typename tensortype3, typename tensortype4>
inline void computeRNEA_SO_derivs(
    const ModelTpl<Scalar, Options, JointCollectionTpl> &model,
    DataTpl<Scalar, Options, JointCollectionTpl> &data,
    const Eigen::MatrixBase<ConfigVectorType> &q,
    const Eigen::MatrixBase<TangentVectorType1> &v,
    const Eigen::MatrixBase<TangentVectorType2> &a, const tensortype1 &dtau_dq2,
    const tensortype2 &dtau_dv2, const tensortype3 &dtau_dqdv,
    const tensortype4 &M_FO) {
  // Extra safety here

  // PINOCCHIO_CHECK_ARGUMENT_SIZE(q.size(), model.nq, "The joint configuration
  // vector is not of right size"); PINOCCHIO_CHECK_ARGUMENT_SIZE(v.size(),
  // model.nv, "The joint velocity vector is not of right size");
  // PINOCCHIO_CHECK_ARGUMENT_SIZE(a.size(), model.nv, "The joint acceleration
  // vector is not of right size");
  // PINOCCHIO_CHECK_ARGUMENT_SIZE(rnea_partial_dq.cols(), model.nv);
  // PINOCCHIO_CHECK_ARGUMENT_SIZE(rnea_partial_dq.rows(), model.nv);
  // PINOCCHIO_CHECK_ARGUMENT_SIZE(rnea_partial_dv.cols(), model.nv);
  // PINOCCHIO_CHECK_ARGUMENT_SIZE(rnea_partial_dv.rows(), model.nv);
  // PINOCCHIO_CHECK_ARGUMENT_SIZE(rnea_partial_da.cols(), model.nv);
  // PINOCCHIO_CHECK_ARGUMENT_SIZE(rnea_partial_da.rows(), model.nv);
  // assert(model.check(data) && "data is not consistent with model.");

  typedef ModelTpl<Scalar, Options, JointCollectionTpl> Model;
  typedef typename Model::JointIndex JointIndex;

  typedef computeRNEA_SO_derivsForwardStep<Scalar, Options, JointCollectionTpl,
                                           ConfigVectorType, TangentVectorType1,
                                           TangentVectorType2>
      Pass1;
  for (JointIndex i = 1; i < (JointIndex)model.njoints; ++i) {
    Pass1::run(model.joints[i], data.joints[i],
               typename Pass1::ArgsType(model, data, q.derived(), v.derived(),
                                        a.derived()));
  }

  typedef computeRNEA_SO_derivsBackwardStep<Scalar, Options, JointCollectionTpl,
                                            tensortype1, tensortype2,
                                            tensortype3, tensortype4>
      Pass2;
  for (JointIndex i = (JointIndex)(model.njoints - 1); i > 0; --i) {
    Pass2::run(model.joints[i],
               typename Pass2::ArgsType(model, data,
                                        const_cast<tensortype1 &>(dtau_dq2),
                                        const_cast<tensortype2 &>(dtau_dv2),
                                        const_cast<tensortype3 &>(dtau_dqdv),
                                        const_cast<tensortype4 &>(M_FO)));
  }
}

} // namespace pinocchio

#endif //
