//
// Copyright (c) 2015-2022 CNRS INRIA
//

#ifndef __pinocchio_algorithm_modrnea_hxx__
#define __pinocchio_algorithm_modrnea_hxx__

/// @cond DEV

#include "pinocchio/multibody/visitor.hpp"
#include "pinocchio/algorithm/check.hpp"

namespace pinocchio
{
  template<typename Scalar, int Options, template<typename,int> class JointCollectionTpl, typename ConfigVectorType, typename TangentVectorType1, typename TangentVectorType2, typename TangentVectorType3>
  struct ModRneaForwardStep
  : public fusion::JointUnaryVisitorBase< ModRneaForwardStep<Scalar,Options,JointCollectionTpl,ConfigVectorType,TangentVectorType1,TangentVectorType2, TangentVectorType3> >
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

      const JointIndex i = jmodel.id();
      const JointIndex parent = model.parents[i];

      jmodel.calc(jdata.derived(),q.derived(),v.derived());

      data.liMi[i] = model.jointPlacements[i]*jdata.M();

      data.v[i] = jdata.v();
      data.wJ[i] = jdata.S() * jmodel.jointVelocitySelector(lambda);
      
      data.w[i] = data.wJ[i];

      if(parent>0)
      {
        data.v[i] += data.liMi[i].actInv(data.v[parent]);
        data.w[i] += data.liMi[i].actInv(data.w[parent]);    
      }
      
      data.a_gf[i] = jdata.c() + (data.v[i] ^ jdata.v());
      data.a_gf[i] += jdata.S() * jmodel.jointVelocitySelector(a);
      data.a_gf[i] += data.liMi[i].actInv(data.a_gf[parent]);

      model.inertias[i].__mult__(data.v[i],data.h[i]);
      model.inertias[i].__mult__(data.a_gf[i],data.f[i]);
      data.f[i] += data.v[i].cross(data.h[i]); // body-frame spatial force

      data.modtau += data.f[i].dot(data.w[i]); // joint torque contracted with w
    }

  };

 
  template<typename Scalar, int Options, template<typename,int> class JointCollectionTpl, typename ConfigVectorType, typename TangentVectorType1, typename TangentVectorType2, typename TangentVectorType3>
  inline const Scalar &
  modrnea(const ModelTpl<Scalar,Options,JointCollectionTpl> & model,
       DataTpl<Scalar,Options,JointCollectionTpl> & data,
       const Eigen::MatrixBase<ConfigVectorType> & q,
       const Eigen::MatrixBase<TangentVectorType1> & v,
       const Eigen::MatrixBase<TangentVectorType2> & a,
       const Eigen::MatrixBase<TangentVectorType3> & lambda)
  {
    assert(model.check(data) && "data is not consistent with model.");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(q.size(), model.nq, "The configuration vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(v.size(), model.nv, "The velocity vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(a.size(), model.nv, "The acceleration vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(lambda.size(), model.nv, "The input vector is not of right size");
    
    typedef ModelTpl<Scalar,Options,JointCollectionTpl> Model;
    typedef typename Model::JointIndex JointIndex;
    
    data.v[0].setZero();
    data.a_gf[0] = -model.gravity;
    data.modtau = Scalar(0);

    typedef ModRneaForwardStep<Scalar,Options,JointCollectionTpl,ConfigVectorType,TangentVectorType1,TangentVectorType2,TangentVectorType3> Pass1;
    typename Pass1::ArgsType arg1(model,data,q.derived(),v.derived(),a.derived(),lambda.derived());
    for(JointIndex i=1; i<(JointIndex)model.njoints; ++i)
    {
      Pass1::run(model.joints[i],data.joints[i],
                 arg1);
    }

    return data.modtau;
    
  }
  

  template<typename Scalar, int Options, template<typename,int> class JointCollectionTpl, typename ConfigVectorType, typename TangentVectorType1, typename TangentVectorType2, typename TangentVectorType3, typename ForceDerived>
  inline const Scalar &
  modrnea(const ModelTpl<Scalar,Options,JointCollectionTpl> & model,
       DataTpl<Scalar,Options,JointCollectionTpl> & data,
       const Eigen::MatrixBase<ConfigVectorType> & q,
       const Eigen::MatrixBase<TangentVectorType1> & v,
       const Eigen::MatrixBase<TangentVectorType2> & a,
       const Eigen::MatrixBase<TangentVectorType3> & lambda,
       const container::aligned_vector<ForceDerived> & fext)
  {
    PINOCCHIO_CHECK_ARGUMENT_SIZE(fext.size(), model.joints.size());
    assert(model.check(data) && "data is not consistent with model.");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(q.size(), model.nq, "The configuration vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(v.size(), model.nv, "The velocity vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(a.size(), model.nv, "The acceleration vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(lambda.size(), model.nv, "The input vector is not of right size");
    
    typedef ModelTpl<Scalar,Options,JointCollectionTpl> Model;
    typedef typename Model::JointIndex JointIndex;
    
    data.v[0].setZero();
    data.a_gf[0] = -model.gravity;
    data.modtau = Scalar(0);

    typedef ModRneaForwardStep<Scalar,Options,JointCollectionTpl,ConfigVectorType,TangentVectorType1,TangentVectorType2,TangentVectorType3> Pass1;
    for(JointIndex i=1; i<(JointIndex)model.njoints; ++i)
    {
      Pass1::run(model.joints[i],data.joints[i],
                 typename Pass1::ArgsType(model,data,q.derived(),v.derived(),a.derived(),lambda.derived()));
      data.f[i] -= fext[i];
    }
    
    return data.modtau;
  }

} // namespace pinocchio

/// @endcond

#endif // ifndef __pinocchio_algorithm_modrnea_hxx__
