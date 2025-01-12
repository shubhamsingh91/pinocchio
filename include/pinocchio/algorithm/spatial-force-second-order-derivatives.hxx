//
// Copyright (c) 2017-2020 CNRS INRIA

#ifndef __pinocchio_algorithm_spatial_force_second_order_derivatives_hxx__
#define __pinocchio_algorithm_spatial_force_second_order_derivatives_hxx__

#include "pinocchio/algorithm/check.hpp"
#include "pinocchio/multibody/visitor.hpp"

namespace pinocchio {

template <typename Scalar, int Options,
          template <typename, int> class JointCollectionTpl,
          typename ConfigVectorType, typename TangentVectorType1,
          typename TangentVectorType2>
struct ComputeSpatialForceSecondOrderDerivativesForwardStep
    : public fusion::JointUnaryVisitorBase<ComputeSpatialForceSecondOrderDerivativesForwardStep<
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

    const JointIndex i = jmodel.id();
    const JointIndex parent = model.parents[i];
    Motion &ov = data.ov[i];
    Motion &oa = data.oa[i];
    Motion &vJ = data.v[i];

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
    ColsBlock J_cols = jmodel.jointCols(
        data.J); // data.J has all the phi (in world frame) stacked in columns
    ColsBlock psid_cols =
        jmodel.jointCols(data.psid); // psid_cols is the psi_dot in world frame
    ColsBlock psidd_cols = jmodel.jointCols(
        data.psidd); // psidd_cols is the psi_dotdot in world frame
    ColsBlock dJ_cols =
        jmodel.jointCols(data.dJ); // This here is phi_dot in world frame

    J_cols.noalias() = data.oMi[i].act(
        jdata.S()); // J_cols is just the phi in world frame for a joint
    vJ = data.oMi[i].act(jdata.v());
    motionSet::motionAction(
        ov, J_cols, psid_cols); // This ov here is v(p(i)), psi_dot calcs
    motionSet::motionAction(
        oa, J_cols, psidd_cols); // This oa here is a(p(i)) , psi_dotdot calcs
    motionSet::motionAction<ADDTO>(
        ov, psid_cols,
        psidd_cols); // This ov here is v(p(i)) , psi_dotdot calcs
    ov += vJ;
    oa += (ov ^ vJ) +
          data.oMi[i].act(jdata.S() * jmodel.jointVelocitySelector(a) +
                          jdata.c());
    motionSet::motionAction(
        ov, J_cols, dJ_cols); // This here is phi_dot, here ov used is v(p(i)) +
                              // vJ Composite rigid body inertia
    Inertia &oY = data.oYcrb[i];

    oY = data.oMi[i].act(model.inertias[i]);
    data.oh[i] = oY * ov;

    data.of[i] = oY * oa + oY.vxiv(ov); // f_i in world frame

    data.doYcrb[i] = oY.variation(ov);
    addForceCrossMatrix(data.oh[i], data.doYcrb[i]); // BC{i}
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

template <typename Scalar, int Options,
          template <typename, int> class JointCollectionTpl,
          typename Tensor1, typename Tensor2, typename Tensor3,
          typename Tensor4>
struct ComputeSpatialForceSecondOrderDerivativesBackwardStep
    : public fusion::JointUnaryVisitorBase<ComputeSpatialForceSecondOrderDerivativesBackwardStep<
          Scalar, Options, JointCollectionTpl, Tensor1, Tensor2,
          Tensor3, Tensor4>> {
  typedef ModelTpl<Scalar, Options, JointCollectionTpl> Model;
  typedef DataTpl<Scalar, Options, JointCollectionTpl> Data;

  typedef boost::fusion::vector<const Model &, Data &, const std::vector<Tensor1> &,
                                const std::vector<Tensor2> &, const std::vector<Tensor3> &,
                                const std::vector<Tensor4> &>
      ArgsType;

  template <typename JointModel>
  static void algo(const JointModelBase<JointModel> &jmodel, const Model &model,
                   Data &data, const std::vector<Tensor1> &d2fc_dqdq,
                   const std::vector<Tensor2> &d2fc_dvdv, const std::vector<Tensor3> &d2fc_dada,
                   const std::vector<Tensor4> &d2fc_dadq) {
    typedef typename Data::Motion Motion;
    typedef typename Data::Force Force;
    typedef typename Data::Inertia Inertia;
    typedef typename Model::JointIndex JointIndex;
    typedef typename Motion::ActionMatrixType ActionMatrixType;
    typedef typename Data::Matrix6 Matrix6;
    typedef typename Data::Vector6r Vector6r;
    typedef typename Data::Vector6c Vector6c;

    const JointIndex i = jmodel.id();
    const JointIndex i_idx = jmodel.idx_v();
    JointIndex j_idx, k_idx;
    const JointIndex parent = model.parents[i];
    // std::cout << "------------------------------------" << std::endl;
    // std::cout << "i = " << i << std::endl;
    // std::cout << "i_idx = " << i_idx << std::endl;

    const Inertia &oYcrb = data.oYcrb[i];  // IC{i}
    const Matrix6 &oBcrb = data.doYcrb[i]; // BC{i}

    std::vector<Tensor1> &d2fc_dqdq_ = const_cast<std::vector<Tensor1> &>(d2fc_dqdq);
    std::vector<Tensor2> &d2fc_dvdv_ = const_cast<std::vector<Tensor2> &>(d2fc_dvdv);

    Vector6c u1;
    Vector6c u2;
    Matrix6  u3;
    Matrix6  u4;
    Vector6c u5;
    Matrix6  u7;
    Matrix6  u8;
    Vector6c u9;
    Vector6c u10;
    Vector6r u11;
    Vector6r u12;
    Vector6c u13;
    Vector6c fci_Sp;
    Vector6c tmp_vec, tmp_vec1;

    Matrix6 Bicphii;
    Matrix6 oBicpsidot;
    Matrix6 fic_cross;

    Matrix6 Bic_phij;
    Matrix6 Bic_psijt_dot, Bic_psikt_dot;
    Matrix6 BCi_St;
    Matrix6 ICi_St;

    Matrix6 r0, ICi_Sp;

    Vector6c s1, s2, s4, s5, s6, s7, s8, s9, s10, s11, s12 ;
    Matrix6  s3, s13;
  
    ForceCrossMatrix(data.of[i], fic_cross); // cmf_bar(f{i}) 

    for (int p = 0; p < model.nvs[i]; p++) {
      const Eigen::DenseIndex ip = model.idx_vs[i] + p;

      // std::cout << "ip = " << ip << std::endl;
      const MotionRef<typename Data::Matrix6x::ColXpr> S_i = data.J.col(ip);          // S{i}(:,p)
      const MotionRef<typename Data::Matrix6x::ColXpr> psid_dm = data.psid.col(ip);   // psi_dot for p DOF
      const MotionRef<typename Data::Matrix6x::ColXpr> psidd_dm = data.psidd.col(ip); // psi_ddot for p DOF
      const MotionRef<typename Data::Matrix6x::ColXpr> phid_dm = data.dJ.col(ip);     // phi_dot for p DOF
      const ActionMatrixType S_iA = S_i.toActionMatrix();                             //(S{i}(:,p) )x matrix

      ICi_Sp = Bicphii = oYcrb.variation(S_i);  // S{i}(p)x*IC{i} - IC{i} S{i}(p)x
      Force f_tmp = oYcrb * S_i;                // IC{i}S{i}(:,p)
      ForceCrossMatrix(f_tmp, r0);              // cmf_bar(IC{i}S{i}(:,p))
      Bicphii += r0;

      oBicpsidot = oYcrb.variation(psid_dm);      // new Bicpsidot in world frame
      f_tmp = oYcrb * psid_dm; // IC{i}psid{i}(:,p)
      addForceCrossMatrix(f_tmp, oBicpsidot); // cmf_bar(IC{i}psid{i}(:,p))
     
      fci_Sp.noalias() = fic_cross * S_i.toVector(); // cmf_bar(f{i}) S{i}(:,p)

      //u1
      Force f_tmp2 = oYcrb*(psid_dm + phid_dm); // ICi*(psid_p + Sd_p);
      u1.noalias() = f_tmp2.toVector(); //ICi*(psid_p + Sd_p);

      //u2
      Force f_tmp3 = oYcrb * S_i; // ICi*S_p;
      u2.noalias() = f_tmp3.toVector(); //ICi*S_p;
      
      //u3
      u3.noalias() = oBicpsidot - S_iA.transpose() * oBcrb - oBcrb * S_iA;  // Bicpsidot + S{i}(p)x*BC{i}- BC {i}S{i}(p)x;

      // u5
      u5.noalias() = oBcrb * psid_dm.toVector() + (oYcrb * psidd_dm).toVector() + fci_Sp; //BCi*psid_p + ICi*psidd_p + fci_Sp         
      Force f_tmp4;
      f_tmp4.toVector().noalias() = u5;
      
      //u4
      ForceCrossMatrix(f_tmp4, u4); // crf_bar(u5);

      //u7
      f_tmp.toVector().noalias() = oBcrb * S_i.toVector(); //BCi*S_p
      ForceCrossMatrix(f_tmp + f_tmp2, u7); // u7

      // u8
      ForceCrossMatrix(f_tmp3, u8);
      u8 += oYcrb.vxi(S_i); // crf_bar(ICi*S_p) + S_ix*ICi

      JointIndex j = i;
      j_idx = model.idx_vs[j];

      while (j > 0) {

        // std::cout << "j = " << j << std::endl;
        // std::cout << "j_idx = " << j_idx << std::endl;

        for (int q = 0; q < model.nvs[j]; q++) {
          const Eigen::DenseIndex jq = model.idx_vs[j] + q;
          std::cout << "jq = " << jq << std::endl;
          const MotionRef<typename Data::Matrix6x::ColXpr> S_j = data.J.col(jq);
          const MotionRef<typename Data::Matrix6x::ColXpr> psid_dj = data.psid.col(jq);   // psi_dot{j}(:,q)
          const MotionRef<typename Data::Matrix6x::ColXpr> psidd_dj = data.psidd.col(jq); // psi_ddot{j}(:,q)
          const MotionRef<typename Data::Matrix6x::ColXpr> phid_dj = data.dJ.col(jq);     // phi_dot{j}(:,q)
          const ActionMatrixType crfSt = S_j.toActionMatrix();                             //(S{i}(:,p) )x matrix
           const ActionMatrixType S_jA = S_j.toActionMatrix();                             //(S{i}(:,p) )x matrix

          BCi_St = -S_jA.transpose() * oBcrb - oBcrb * S_jA  ; // S_j x* BC{i} - BC{i} S_j x

          ICi_St = Bic_phij      = oYcrb.variation(S_j);  // S_j x* IC{i} - IC{i} S_j x
          ForceCrossMatrix(oYcrb * S_j, r0);            // cmf_bar(IC{i}S{j}(:,q))
          Bic_phij += r0;

          Bic_psijt_dot = oYcrb.variation(psid_dj);       // psi_dot{j}(:,q) x* IC{i} - IC{i} psi_dot{j}(:,q) x
          ForceCrossMatrix(oYcrb * psid_dj, r0); // cmf_bar(IC{i} * psi_dot{j}(:,q))
          Bic_psijt_dot += r0;

          // s1 = psid_t + Sd_t
          s1.noalias() = (psid_dj + phid_dj).toVector();

          // s2 = BCi*psid_t + ICi*psidd_t + fCi_bar * S_t
          s2.noalias() = oBcrb * psid_dj.toVector()
                    + (oYcrb * psidd_dj).toVector()
                    + fic_cross * S_j.toVector();

          // s3 = Bic_psijt_dot + BCi_St
          s3.noalias() = Bic_psijt_dot + BCi_St; // 6x6

          // s4 = ICi * S_t
          s4.noalias() = (oYcrb * S_j).toVector(); // 6x1 (Force)

          // s5 = ICi*s1
          s5.noalias() = (oYcrb * (psid_dj + phid_dj)).toVector();

          // s6 = u3*psid_t + ICi_Sp*psidd_t + crfSt*u5
          s6.noalias() = u3 * psid_dj.toVector()
                    + (ICi_Sp * psidd_dj.toVector())
                    - (crfSt.transpose() * u5);  

          // s7 = Bic_phii * S_t
          s7.noalias() = Bicphii * S_j.toVector();

          // s8 = crfSt*u2
          s8.noalias() = -crfSt.transpose() * u2;

          // s9 = ICi_Sp * S_t
          s9.noalias() = ICi_Sp * S_j.toVector();

          // s10 = Bic_phii*psid_t + u7*S_t
          s10.noalias() = Bicphii * psid_dj.toVector()
                      + (u7 * S_j.toVector());

          // s11 = u3*S_t + ICi_Sp*(psid_t + Sd_t)
          s11.noalias()  = u3 * S_j.toVector() + ICi_Sp * s1;

          // s12 = u8 * S_t
          s12.noalias() = u8 * S_j.toVector();

          // s13 = crfSt*ICi + crf_bar(s4)
          Matrix6 crf_s4; 
          ForceCrossMatrix(oYcrb * S_j, crf_s4); // crf_bar of s4
          s13 = -crfSt.transpose() * oYcrb.matrix() + crf_s4;

          JointIndex k = j;
          k_idx = model.idx_vs[k];

          while (k > 0) {
            // std::cout << "k = " << k << std::endl;
            // std::cout << "k_idx = " << k_idx << std::endl;

            for (int r = 0; r < model.nvs[k]; r++) {
              const Eigen::DenseIndex kr = model.idx_vs[k] + r;
              // std::cout << "kr = " << kr << std::endl;
              const MotionRef<typename Data::Matrix6x::ColXpr> S_k(data.J.col(kr));
              const MotionRef<typename Data::Matrix6x::ColXpr> psid_dk = data.psid.col(kr);   // psi_dot{k}(:,r)
              const MotionRef<typename Data::Matrix6x::ColXpr> psidd_dk = data.psidd.col(kr); // psi_ddot{k}(:,r)
              const MotionRef<typename Data::Matrix6x::ColXpr> phid_dk = data.dJ.col(kr);     // phi_dot{k}(:,r)
              const ActionMatrixType crfSk = S_k.toActionMatrix();                             //(S{i}(:,p) )x matrix
              const ActionMatrixType crmpsidk = psid_dk.toActionMatrix();
          
              Bic_psikt_dot = oYcrb.variation(psid_dk);       // psi_dot{k}(:,q) x* IC{i} - IC{i} psi_dot{k}(:,q) x
              ForceCrossMatrix(oYcrb * psid_dk, r0);       // cmf_bar(IC{i} * psi_dot{k}(:,q))
              Bic_psikt_dot += r0;

              // k <= j <= i

              // expr-1 SO-q
              tmp_vec = s3*psid_dk.toVector() + ICi_St*psidd_dk.toVector() - crfSk.transpose() * s2;
              hess_assign(d2fc_dqdq_.at(i_idx), tmp_vec , 0, jq, kr, 1, 6); // d2fc_dqdq_(i)(1:6, jq, kr) 
               
              get_vec_from_tens3_v1_gen(d2fc_dada.at(i_idx), tmp_vec1, 6 , jq, kr);
            

              if ((tmp_vec -tmp_vec1).norm()>1e-3)
              {
                std::cout << "SO expr-1" << std::endl;
                std::cout << "i_idx = " << i_idx << " ,jq = " << jq << " ,kr = " << kr << std::endl;
                std::cout << "tmp_vec = \n" << tmp_vec << std::endl;
                std::cout << "tmp_vec1 = \n" << tmp_vec1 << std::endl;
              } 

              if (j != i) { // k <= j < i

                //  expr-5 SO-q 
                tmp_vec = ICi_Sp * psidd_dk.toVector() + u4 * S_k.toVector() + u3 * psid_dk.toVector();
                hess_assign(d2fc_dqdq_.at(j_idx), tmp_vec, 0, kr, ip, 1, 6); // d2fc_dqdq_(j)(1:6, kr, ip)
              
                get_vec_from_tens3_v1_gen(d2fc_dada.at(j_idx), tmp_vec1, 6 , kr, ip);

              if ((tmp_vec -tmp_vec1).norm()>1e-3)
              {
                std::cout << "SO expr-5" << std::endl;
                std::cout << "i_idx = " << i_idx << " ,jq = " << jq << " ,kr = " << kr << std::endl;
                std::cout << "tmp_vec = \n" << tmp_vec << std::endl;
                std::cout << "tmp_vec1 = \n" << tmp_vec1 << std::endl;
              } 

                // expr-6 SO-q
                hess_assign(d2fc_dqdq_.at(j_idx), tmp_vec, 0, ip, kr, 1, 6); // d2fc_dqdq_(j)(1:6, ip, kr) 
                get_vec_from_tens3_v1_gen(d2fc_dada.at(j_idx), tmp_vec1, 6 , ip, kr);
              if ((tmp_vec -tmp_vec1).norm()>1e-3)
              {
                std::cout << "SO expr-6" << std::endl;
                std::cout << "i_idx = " << i_idx << " ,jq = " << jq << " ,kr = " << kr << std::endl;
                std::cout << "tmp_vec = \n" << tmp_vec << std::endl;
                std::cout << "tmp_vec1 = \n" << tmp_vec1 << std::endl;
              } 

              }

              if (k != j) { // k < j <= i

                // expr-2 SO-q  d2fc_dq{i}(:, kk(r), jj(t)) = d2fc_dq{i}(:, jj(t), kk(r));
                get_vec_from_tens3_v1_gen(d2fc_dqdq_.at(i_idx), tmp_vec, 6 , jq, kr);
                hess_assign(d2fc_dqdq_.at(i_idx), tmp_vec, 0, kr, jq, 1, 6); // d2fc_dqdq_(i)(1:6, kr, jq)
        
                get_vec_from_tens3_v1_gen(d2fc_dada.at(i_idx), tmp_vec1, 6 , kr, jq);

                if ((tmp_vec -tmp_vec1).norm()>1e-3)
                {
                  std::cout << "SO expr-2" << std::endl;
                  std::cout << "i_idx = " << i_idx << " ,jq = " << jq << " ,kr = " << kr << std::endl;
                  std::cout << "tmp_vec = \n" << tmp_vec << std::endl;
                  std::cout << "tmp_vec1 = \n" << tmp_vec1 << std::endl;
                } 

                // expr-3 SO-q
                // d2fc_dq{k}(:, ii(p), jj(t))
                hess_assign(d2fc_dqdq_.at(k_idx), s6 , 0, ip, jq, 1, 6); // d2fc_dqdq_(k)(1:6, ip, jq)
           
                get_vec_from_tens3_v1_gen(d2fc_dada.at(k_idx), tmp_vec1, 6 , ip, jq);

                if ((s6 -tmp_vec1).norm()>1e-3)
                {
                  std::cout << "SO expr-3" << std::endl;
                  std::cout << "i_idx = " << i_idx << " ,jq = " << jq << " ,kr = " << kr << std::endl;
                  std::cout << "tmp_vec = \n" << s6 << std::endl;
                  std::cout << "tmp_vec1 = \n" << tmp_vec1 << std::endl;
                } 

                if (j != i) { // k < j < i
                  //  expr-4 SO-q   d2fc_dq{k}(:, jj(t), ii(p)) =  d2fc_dq{k}(:, ii(p), jj(t));
                  get_vec_from_tens3_v1_gen(d2fc_dqdq_.at(k_idx), tmp_vec, 6 , ip, jq);
                  hess_assign(d2fc_dqdq_.at(k_idx), tmp_vec, 0, jq, ip, 1, 6); // d2fc_dqdq_(k)(1:6, jq, ip)
                 
                  get_vec_from_tens3_v1_gen(d2fc_dada.at(k_idx), tmp_vec1, 6 , jq, ip);

                  if ((tmp_vec -tmp_vec1).norm()>1e-3)
                  {
                    std::cout << "SO expr-4" << std::endl;
                    std::cout << "i_idx = " << i_idx << " ,jq = " << jq << " ,kr = " << kr << std::endl;
                    std::cout << "tmp_vec = \n" << tmp_vec << std::endl;
                    std::cout << "tmp_vec1 = \n" << tmp_vec1 << std::endl;
                  }

                  // std::cout << "SO expr-4" << std::endl;
                  // std::cout << "i_idx = " << i_idx << "jq = " << jq << "kr = " << kr << std::endl;

                  // std::cout << "tmp_vec = \n" << tmp_vec << std::endl;

                  } else { // k < j = i

                }

              } else { // k = j <= i

              }
              
            }

            k = model.parents[k];
            k_idx = model.idx_vs[k];

          }
        }
        
        
        j = model.parents[j];
        j_idx = model.idx_vs[j];


      }
    }

    if (parent > 0) {
      data.oYcrb[parent] += data.oYcrb[i];
      data.doYcrb[parent] += data.doYcrb[i];
      data.of[parent] += data.of[i];
    }
  }

  // Function for cmf_bar operator
  template <typename ForceDerived, typename M6>
  static void ForceCrossMatrix(const ForceDense<ForceDerived> &f,
                               const Eigen::MatrixBase<M6> &mout) {
    M6 &mout_ = PINOCCHIO_EIGEN_CONST_CAST(M6, mout);
    mout_.template block<3, 3>(ForceDerived::LINEAR, ForceDerived::LINEAR)
        .setZero();
    mout_.template block<3, 3>(ForceDerived::LINEAR, ForceDerived::ANGULAR) =
        mout_.template block<3, 3>(ForceDerived::ANGULAR,
                                   ForceDerived::LINEAR) = skew(-f.linear());
    mout_.template block<3, 3>(ForceDerived::ANGULAR, ForceDerived::ANGULAR) =
        skew(-f.angular());
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
    const Eigen::MatrixBase<TangentVectorType2> &a, const std::vector<Tensor1> &d2fc_dqdq,
    const std::vector<Tensor2> &d2fc_dvdv, const std::vector<Tensor3> &d2fc_dada,
    const std::vector<Tensor4> &d2fc_dadq) {
  // Extra safety here
  PINOCCHIO_CHECK_ARGUMENT_SIZE(
      q.size(), model.nq,
      "The joint configuration vector is not of right size");
  PINOCCHIO_CHECK_ARGUMENT_SIZE(
      v.size(), model.nv, "The joint velocity vector is not of right size");
  PINOCCHIO_CHECK_ARGUMENT_SIZE(
      a.size(), model.nv, "The joint acceleration vector is not of right size");
//   PINOCCHIO_CHECK_ARGUMENT_SIZE(d2tau_dqdq.dimension(0), model.nv);
//   PINOCCHIO_CHECK_ARGUMENT_SIZE(d2tau_dqdq.dimension(1), model.nv);
//   PINOCCHIO_CHECK_ARGUMENT_SIZE(d2tau_dqdq.dimension(2), model.nv);
//   PINOCCHIO_CHECK_ARGUMENT_SIZE(d2tau_dvdv.dimension(0), model.nv);
//   PINOCCHIO_CHECK_ARGUMENT_SIZE(d2tau_dvdv.dimension(1), model.nv);
//   PINOCCHIO_CHECK_ARGUMENT_SIZE(d2tau_dvdv.dimension(2), model.nv);
//   PINOCCHIO_CHECK_ARGUMENT_SIZE(dtau_dqdv.dimension(0), model.nv);
//   PINOCCHIO_CHECK_ARGUMENT_SIZE(dtau_dqdv.dimension(1), model.nv);
//   PINOCCHIO_CHECK_ARGUMENT_SIZE(dtau_dqdv.dimension(2), model.nv);
//   PINOCCHIO_CHECK_ARGUMENT_SIZE(dtau_dadq.dimension(0), model.nv);
//   PINOCCHIO_CHECK_ARGUMENT_SIZE(dtau_dadq.dimension(1), model.nv);
//   PINOCCHIO_CHECK_ARGUMENT_SIZE(dtau_dadq.dimension(2), model.nv);
  assert(model.check(data) && "data is not consistent with model.");

  typedef ModelTpl<Scalar, Options, JointCollectionTpl> Model;
  typedef typename Model::JointIndex JointIndex;

  typedef ComputeSpatialForceSecondOrderDerivativesForwardStep<
      Scalar, Options, JointCollectionTpl, ConfigVectorType, TangentVectorType1,
      TangentVectorType2>
      Pass1;
  for (JointIndex i = 1; i < (JointIndex)model.njoints; ++i) {
    Pass1::run(model.joints[i], data.joints[i],
               typename Pass1::ArgsType(model, data, q.derived(), v.derived(),
                                        a.derived()));
  }

  typedef ComputeSpatialForceSecondOrderDerivativesBackwardStep<
      Scalar, Options, JointCollectionTpl, Tensor1, Tensor2,
      Tensor3, Tensor4>
      Pass2;
  for (JointIndex i = (JointIndex)(model.njoints - 1); i > 0; --i) {
    Pass2::run(model.joints[i],
               typename Pass2::ArgsType(model, data,
                                        const_cast<std::vector<Tensor1> &>(d2fc_dqdq),
                                        const_cast<std::vector<Tensor2> &>(d2fc_dvdv),
                                        const_cast<std::vector<Tensor3> &>(d2fc_dada),
                                        const_cast<std::vector<Tensor4> &>(d2fc_dadq)));
  }
}

} // namespace pinocchio

#endif // ifndef __pinocchio_algorithm_spatial_force_second_order_derivatives_hxx__
