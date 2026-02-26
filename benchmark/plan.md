# Plan: SO Modified Derivatives Benchmark

## Context

Compare four approaches for computing second-order modified (lambda-contracted) derivatives. All four compute the Hessian of a scalar function (lambda*tau for ID, mu*qddot for FD), producing nv x nv matrices. They differ in how much work is analytical vs AD.

### Function hierarchy (ID)

| Level | Function | Input -> Output |
|-------|----------|---------------|
| Base | `modrnea(q,v,a,l)` | -> scalar l*tau |
| FO derivs (full) | `computeRNEADerivatives(q,v,a)` | -> matrices dtau/dq, dtau/dv, M (nv x nv) |
| FO derivs (mod) | `computeModRNEADerivatives(q,v,a,l)` | -> vectors d(l*tau)/dq, d(l*tau)/dv, d(l*tau)/da (nv) |
| SO derivs (mod) | `computeModRNEASecondOrderDerivatives(q,v,a,l)` | -> matrices d2(l*tau)/dq2, etc. (nv x nv) |

### Function hierarchy (FD)

| Level | Function | Input -> Output |
|-------|----------|---------------|
| Base | `modaba(q,v,tau,mu)` | -> scalar mu*qddot |
| FO derivs (full) | `computeABADerivatives(q,v,tau)` | -> matrices dqddot/dq, dqddot/dv, M^-1 (nv x nv) |
| FO derivs (mod) | `computeModABADerivatives(q,v,tau,mu)` | -> vectors d(mu*qddot)/dq, d(mu*qddot)/dv, d(mu*qddot)/dtau (nv) |
| SO derivs (mod) | chain-rule formula | -> matrices d2(mu*qddot)/dq2, etc. (nv x nv) |

### The 4 Approaches

| # | Name | What CasADi traces | AD diffs |
|---|------|---------------------|----------|
| 1 | **Full SO AD** | `modrnea(q,v,a,l)` / `modaba(q,v,tau,mu)` -> scalar -> Hessian | 2 (fwd over rev) |
| 2a | **FO AD over full FO derivs** | `computeRNEADerivatives()` / `computeABADerivatives()` -> nv x nv matrices -> contract with l/mu -> Jacobian | 1 |
| 2b | **FO AD over mod FO derivs** | `computeModRNEADerivatives()` / `computeModABADerivatives()` -> nv vectors -> Jacobian | 1 |
| 3 | **Full analytical** | `computeModRNEASecondOrderDerivatives()` / chain-rule | 0 |

**CasADi only** (no CppAD). Codegen deferred. MATLAB for plotting. 4 bars per model.

## Deliverables

1. `benchmark/bench_modSO_plan.md` - This design document
2. `benchmark/bench_modID_SO.cpp` - ID SO benchmark (accuracy + timing, 4 approaches)
3. `benchmark/bench_modFD_SO.cpp` - FD SO benchmark (accuracy + timing, 4 approaches)
4. MATLAB plotting - New .m functions adapted from `figure_pinocchio_bar_IDSVA_SO.m`, 4 bars per model

## Implementation Steps

### Step 1: Create this plan and git add

### Step 2: bench_modID_SO.cpp - Inverse Dynamics SO benchmark

Per-model structure:

```
ACCURACY (single eval at same q,v,a,lambda):
  Case 3  (analytical):    computeModRNEASecondOrderDerivatives -> dqq, dvv, dvq, dqa
  Case 1  (full SO AD):    CasADi trace modrnea -> scalar -> Hessian -> dqq, dvv, dvq, dqa
  Case 2a (AD over full):  CasADi trace computeRNEADerivatives -> contract l -> Jacobian
  Case 2b (AD over mod):   CasADi trace computeModRNEADerivatives -> Jacobian
  Print: ||case1 - case3||, ||case2a - case3||, ||case2b - case3|| per matrix

TIMING (NBT iterations):
  Time each case. Write avg microseconds to data file.
```

**Case 1** (trace modrnea, take Hessian):
```cpp
// modrnea returns scalar l*tau directly via data.modtau
modrnea(adc_model, adc_data, q_int_ad, v_ad, a_ad, lambda_ad);
::casadi::SX lambda_tau = adc_data.modtau;  // scalar
// Hessian via 2 jacobian calls (fwd over rev)
::casadi::SX grad_q = jacobian(lambda_tau, cs_v_int);   // 1 x nv
::casadi::SX hess_qq = jacobian(grad_q, cs_v_int);      // nv x nv
::casadi::SX grad_v = jacobian(lambda_tau, cs_v);
::casadi::SX hess_vv = jacobian(grad_v, cs_v);          // nv x nv
::casadi::SX hess_vq = jacobian(grad_v, cs_v_int);      // nv x nv
::casadi::SX grad_a = jacobian(lambda_tau, cs_a);
::casadi::SX hess_qa = jacobian(grad_a, cs_v_int);      // nv x nv
```

**Case 2a** (trace computeRNEADerivatives, contract with l, take Jacobian):
```cpp
computeRNEADerivatives(adc_model, adc_data, q_int_ad, v_ad, a_ad);
// Contract: g_q(i) = sum_j l(j) * dtau_dq(j,i)
::casadi::SX g_q(nv, 1), g_v(nv, 1), g_a(nv, 1);
for (int i = 0; i < nv; i++)
  for (int j = 0; j < nv; j++) {
    g_q(i) += lambda_val[j] * adc_data.dtau_dq(j, i);
    g_v(i) += lambda_val[j] * adc_data.dtau_dv(j, i);
    g_a(i) += lambda_val[j] * adc_data.M(j, i);
  }
// 1 Jacobian call (forward)
::casadi::SX hess_qq = jacobian(g_q, cs_v_int);  // nv x nv
::casadi::SX hess_vv = jacobian(g_v, cs_v);
::casadi::SX hess_vq = jacobian(g_v, cs_v_int);
::casadi::SX hess_qa = jacobian(g_a, cs_v_int);
```

**Case 2b** (trace computeModRNEADerivatives, take Jacobian):
```cpp
computeModRNEADerivatives(adc_model, adc_data, q_int_ad, v_ad, a_ad, lambda_ad);
// Outputs already contracted: vectors of length nv
::casadi::SX g_q(nv, 1), g_v(nv, 1), g_a(nv, 1);
for (int i = 0; i < nv; i++) {
  g_q(i) = adc_data.dtau_dq_mod[i];
  g_v(i) = adc_data.dtau_dv_mod[i];
  g_a(i) = adc_data.M_mod[i];
}
// 1 Jacobian call (forward)
::casadi::SX hess_qq = jacobian(g_q, cs_v_int);  // nv x nv
::casadi::SX hess_vv = jacobian(g_v, cs_v);
::casadi::SX hess_vq = jacobian(g_v, cs_v_int);
::casadi::SX hess_qa = jacobian(g_a, cs_v_int);
```

**Case 3** (analytical):
```cpp
computeModRNEASecondOrderDerivatives(model, data, q, v, a, lambda,
    dtau_dqq_mod, dtau_dvv_mod, dtau_dvq_mod, dtau_dqa_mod);
```

**Models**: double_pendulum(2), ur3_robot(6), hyq(18,ff), atlas(36,ff), talos_full_v2(50,ff)

### Step 3: bench_modFD_SO.cpp - Forward Dynamics SO benchmark

Same structure, 4 approaches for FD:

**Case 1**: Trace `modaba(q,v,tau,mu)` -> scalar mu*qddot -> Hessian (2 diffs)

**Case 2a**: Trace `computeABADerivatives(q,v,tau)` -> contract mu^T * daba_dq etc -> Jacobian (1 diff)

**Case 2b**: Trace `computeModABADerivatives(q,v,tau,mu)` -> gradient vectors -> Jacobian (1 diff)

**Case 3**: Chain-rule (from test_modDynamicsSO.cpp):
```
lambda_fd = Minv * mu
computeModRNEASecondOrderDerivatives(model, data, q, v, qddot_fd, lambda_fd, dqq, dvv, dvq, dqa)
computeABADerivatives(model, data, q, v, tau) -> ddq_dq, ddq_dv, Minv
d2(mu*qddot)/dqq = -dqq - dqa*ddq_dq - ddq_dq^T*dqa^T
d2(mu*qddot)/dvv = -dvv
d2(mu*qddot)/dqv = -dvq - dqa*ddq_dv
d2(mu*qddot)/dqtau = -dqa*Minv
```

### Step 4: MATLAB plotting

Adapt `figure_pinocchio_bar_IDSVA_SO.m`. **4 bars per model**:
- Green: Case 3 - Full analytical
- Red: Case 2b - FO AD over modID/modFD
- Blue: Case 2a - FO AD over full FO derivs + contract
- Magenta: Case 1 - Full SO AD

Log y-axis, LaTeX labels, same figure size/formatting as existing plots.

### Step 5: Run and verify
1. Compile -O0, run double_pendulum only -> verify accuracy (norms < 1e-6)
2. Recompile -O3, run all 5 models -> timing data to .txt files
3. Run MATLAB plotter -> save bar chart figures

## Output Data Format
Each {model}.txt in benchmark/data/modID_SO/ and modFD_SO/:
```
<case3_analytical_time_us>
<case1_full_SO_AD_time_us>
<case2a_AD_over_full_FO_time_us>
<case2b_AD_over_mod_FO_time_us>
```

## Compile Command
```
g++ bench_modID_SO.cpp -DNDEBUG -I /usr/include/eigen3 -O3 \
  -I /home/shubham/Desktop/pinocchio/include -I /usr/include \
  -L /usr/local/lib -lpinocchio -lcasadi -ldl -o bench_modID_SO
```

## Key Reference Files
- include/pinocchio/algorithm/modrnea.hpp - modrnea() returns scalar via data.modtau
- include/pinocchio/algorithm/modaba.hpp - modaba() returns scalar mu*qddot
- include/pinocchio/algorithm/mod-rnea-derivatives.hpp - computeModRNEADerivatives() returns gradient vectors
- include/pinocchio/algorithm/mod-aba-derivatives.hpp - computeModABADerivatives() returns gradient vectors
- include/pinocchio/algorithm/mod-rnea-second-order-derivatives.hpp - analytical SO API
- benchmark/AD_mdof_v1.cpp - CasADi trace patterns (lines 235-340 FO, 443-545 SO)
- benchmark/test_modDynamicsSO.cpp - FD chain-rule formula (lines 240-287)
- /home/shubham/Desktop/spatial_v2_extended/plotter/figure_pinocchio_bar_IDSVA_SO.m - plot template
