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

  namespace internal
  {
    ///
    /// \brief Internal function that performs the actual computations of
    ///        computeModABADerivatives, handling both the with- and without-fext
    ///        cases via a single code path.
    ///
    template<typename Scalar,
             int Options,
             template<typename,int> class JointCollectionTpl,
             typename ConfigVectorType,
             typename TangentVectorType1,
             typename TangentVectorType2,
             typename TangentVectorType3,
             typename VectorType1,
             typename VectorType2,
             typename VectorType3>
    inline void
    computeModABADerivativesImpl(const ModelTpl<Scalar,Options,JointCollectionTpl> & model_in,
                                 DataTpl<Scalar,Options,JointCollectionTpl> & data,
                                 const Eigen::MatrixBase<ConfigVectorType> & q,
                                 const Eigen::MatrixBase<TangentVectorType1> & v,
                                 const Eigen::MatrixBase<TangentVectorType2> & tau,
                                 const Eigen::MatrixBase<TangentVectorType3> & lambda,
                                 const container::aligned_vector< ForceTpl<Scalar,Options> > & fext, // can be empty
                                 const Eigen::MatrixBase<VectorType1> & aba_partial_dq_mod,
                                 const Eigen::MatrixBase<VectorType2> & aba_partial_dv_mod,
                                 const Eigen::MatrixBase<VectorType3> & aba_partial_dtau_mod)
    {
      PINOCCHIO_CHECK_ARGUMENT_SIZE(q.size(), model_in.nq,"The joint configuration vector q is not of right size");
      PINOCCHIO_CHECK_ARGUMENT_SIZE(v.size(), model_in.nv,"The joint velocity vector v is not of right size");
      PINOCCHIO_CHECK_ARGUMENT_SIZE(tau.size(), model_in.nv,"The joint torque vector tau is not of right size");
      PINOCCHIO_CHECK_ARGUMENT_SIZE(lambda.size(), model_in.nv,"The input vector lambda is not of right size");
      PINOCCHIO_CHECK_ARGUMENT_SIZE(aba_partial_dq_mod.rows(), model_in.nv,"The partial-dq output has wrong number of rows");
      PINOCCHIO_CHECK_ARGUMENT_SIZE(aba_partial_dv_mod.rows(), model_in.nv,"The partial-dv output has wrong number of rows");
      PINOCCHIO_CHECK_ARGUMENT_SIZE(aba_partial_dtau_mod.rows(), model_in.nv,"The partial-dtau output has wrong number of rows");

      // If we do have a non-empty fext, check its size:
      if(!fext.empty())
      {
        PINOCCHIO_CHECK_ARGUMENT_SIZE(fext.size(), (size_t)model_in.njoints,"The size of the external forces fext is not of right size");
      }

      // Check that data is consistent with model
      assert(model_in.check(data) && "data is not consistent with model.");

      //--------------------------------------------------------------------------
      // 2) ALIAS REFERENCES FOR THE OUTPUT BUFFERS
      //--------------------------------------------------------------------------

      VectorType1 & dq_mod_out   = PINOCCHIO_EIGEN_CONST_CAST(VectorType1, aba_partial_dq_mod);
      VectorType2 & dv_mod_out   = PINOCCHIO_EIGEN_CONST_CAST(VectorType2, aba_partial_dv_mod);
      VectorType3 & dtau_mod_out = PINOCCHIO_EIGEN_CONST_CAST(VectorType3, aba_partial_dtau_mod);

      //--------------------------------------------------------------------------
      // 3) TEMPORARILY ZERO OUT GRAVITY, COMPUTE mu = ABA(..., 0*v, lambda)
      //--------------------------------------------------------------------------

      typedef ModelTpl<Scalar,Options,JointCollectionTpl> Model;
      Model & mutable_model = const_cast<Model&>(model_in);

      // Save original gravity
      const MotionTpl<Scalar> original_gravity = mutable_model.gravity;
      // Zero-out gravity to compute mu
      mutable_model.gravity = MotionTpl<Scalar>::Zero();

      Eigen::Matrix<Scalar,Eigen::Dynamic,1> mu = aba(mutable_model, data, q, v*Scalar(0), lambda);

      // Restore original gravity
      mutable_model.gravity = original_gravity;

      //--------------------------------------------------------------------------
      // 4) COMPUTE REAL ddq = ABA(model, data, q, v, tau, fext) [with or w/o fext]
      //--------------------------------------------------------------------------

      Eigen::Matrix<Scalar,Eigen::Dynamic,1> ddq;
      if(fext.empty())
      {
        ddq = aba(model_in, data, q, v, tau); // no external forces
      }
      else
      {
        ddq = aba(model_in, data, q, v, tau, fext); // with external forces
      }

      //--------------------------------------------------------------------------
      // 5) CALL computeModRNEADerivatives(...) WITH THE -mu ARGUMENT
      //--------------------------------------------------------------------------

      if(fext.empty())
      {
        computeModRNEADerivatives(model_in, data, q, v, ddq, -mu);
      }
      else
      {
        computeModRNEADerivatives(model_in, data, q, v, ddq, -mu, fext);
      }

      //--------------------------------------------------------------------------
      // 6) COPY THE RESULTS BACK TO THE USER'S OUTPUT BUFFERS
      //--------------------------------------------------------------------------

      dq_mod_out   = data.dtau_dq_mod;  // partial wrt q
      dv_mod_out   = data.dtau_dv_mod;  // partial wrt v
      dtau_mod_out = mu;                // partial wrt tau
    }
  } // end namespace internal

  template<typename Scalar, int Options, template<typename,int> class JointCollectionTpl,
           typename ConfigVectorType,
           typename TangentVectorType1,
           typename TangentVectorType2,
           typename TangentVectorType3,
           typename VectorType1,
           typename VectorType2,
           typename VectorType3>
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
    // Create an empty aligned_vector for external forces:
    // (This can be static or local; it doesn't matter as long as it remains empty.)
    container::aligned_vector< ForceTpl<Scalar,Options> > empty_fext;

    internal::computeModABADerivativesImpl(model, data,
                                           q, v, tau, lambda,
                                           empty_fext,  // no external forces
                                           aba_partial_dq_mod,
                                           aba_partial_dv_mod,
                                           aba_partial_dtau_mod);
  }

  template<typename Scalar, int Options, template<typename,int> class JointCollectionTpl,
           typename ConfigVectorType,
           typename TangentVectorType1,
           typename TangentVectorType2,
           typename TangentVectorType3,
           typename VectorType1,
           typename VectorType2,
           typename VectorType3>
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
    internal::computeModABADerivativesImpl(model, data,
                                           q, v, tau, lambda,
                                           fext,  // actual external forces
                                           aba_partial_dq_mod,
                                           aba_partial_dv_mod,
                                           aba_partial_dtau_mod);
  }

} // namespace pinocchio

#endif // ifndef __pinocchio_mod_aba_derivatives_hxx__
