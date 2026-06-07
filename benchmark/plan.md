# Plan: SO Modified Derivatives Benchmark

## Context

Compare approaches for computing second-order modified (lambda-contracted) derivatives. All approaches compute the Hessian of a scalar function (lambda*tau for ID, mu*qddot for FD), producing nv x nv matrices. They differ in how much work is analytical vs AD.

**Key insight (Pat):** The modified SO algorithm sits between FO (fast) and full SO tensor (slow), showing it's competitive with FO while computing second-order information. Even the "souped up" AD baseline (Case 1: SO AD on modrnea, which fuses lambda into the recursion) is slower than the analytical mod SO.

### Function hierarchy (ID)

| Level | Function | Input -> Output |
|-------|----------|---------------|
| Base | `modrnea(q,v,a,l)` | -> scalar l*tau |
| FO derivs (full) | `computeRNEADerivatives(q,v,a)` | -> matrices dtau/dq, dtau/dv, M (nv x nv) |
| FO derivs (mod) | `computeModRNEADerivatives(q,v,a,l)` | -> vectors d(l*tau)/dq, d(l*tau)/dv, d(l*tau)/da (nv) |
| SO derivs (full) | `ComputeRNEASecondOrderDerivatives(q,v,a)` | -> tensors d2tau/dq2, etc. (nv x nv x nv) |
| SO derivs (mod) | `computeModRNEASecondOrderDerivatives(q,v,a,l)` | -> matrices d2(l*tau)/dq2, etc. (nv x nv) |

### Function hierarchy (FD)

| Level | Function | Input -> Output |
|-------|----------|---------------|
| Base | `modaba(q,v,tau,mu)` | -> scalar mu*qddot |
| FO derivs (full) | `computeABADerivatives(q,v,tau)` | -> matrices dqddot/dq, dqddot/dv, M^-1 (nv x nv) |
| FO derivs (mod) | `computeModABADerivatives(q,v,tau,mu)` | -> vectors d(mu*qddot)/dq, d(mu*qddot)/dv, d(mu*qddot)/dtau (nv) |
| SO derivs (full FD) | chain-rule with full ID SO tensors | -> tensors d2(qddot)/dq2 etc (nv x nv x nv) |
| SO derivs (mod) | chain-rule with mod ID SO matrices | -> matrices d2(mu*qddot)/dq2, etc. (nv x nv) |

### The 6 Timing Approaches (bar chart)

| # | Name | What it computes | Notes |
|---|------|-----------------|-------|
| FO | **FO analytical** | computeRNEADerivativesFaster / aba+computeABADerivativesFaster | Baseline from RAL |
| Full SO | **Full SO analytical (tensor)** | ComputeRNEASecondOrderDerivatives / full FD chain-rule | From T-Ro, O(nv^4) |
| Mod SO | **Mod SO analytical** | computeModRNEASecondOrderDerivatives / mod FD chain-rule | Our new algo, O(nv^2) |
| Case 1 | **Full SO AD (codegen)** | CasADi SO AD on modrnea/modaba, compiled .so | 2 AD diffs, souped-up baseline |
| Case 2a | **FO AD over full FO (codegen)** | CasADi Jacobian of FO derivs + lambda contraction, compiled .so | 1 AD diff |
| Case 2b | **FO AD over mod FO (codegen)** | CasADi Jacobian of mod FO derivs, compiled .so | 1 AD diff |

## Current Status

### Completed
- [x] bench_modID_SO.cpp - timing with CasADi codegen (Cases 1, 2a, 2b, 3)
- [x] bench_modFD_SO.cpp - timing with CasADi codegen (Cases 1, 2a, 2b, 3)
- [x] bench_modID_SO_accuracy.cpp - accuracy checks (Cases 1, 2a, 2b vs analytical)
- [x] bench_modFD_SO_accuracy.cpp - accuracy checks (Cases 1, 2a, 2b vs analytical)
- [x] plot_modSO_benchmarks.py - Python bar charts (4 bars per model)
- [x] CasADi codegen pipeline (--codegen / --eval, codegen/ dir, parallel gcc, skip existing .so)
- [x] Per-case .so skip logic in eval (gracefully handles missing .so files)
- [x] ID codegen complete for all 5 models
- [x] FD codegen complete for double_pendulum, ur3_robot, hyq_f (atlas/talos OOM)

### TODO — New work
- [ ] **Part 1a:** Add full SO (tensor) + lambda contraction to bench_modID_SO_accuracy.cpp
- [ ] **Part 1b:** Add full FD SO chain-rule + mu contraction to bench_modFD_SO_accuracy.cpp
- [ ] **Part 2a:** Add FO + full SO timing to bench_modID_SO.cpp eval phase
- [ ] **Part 2b:** Add FO + full FD SO timing to bench_modFD_SO.cpp eval phase
- [ ] **Part 3:** Update data file format (4 lines -> 6 lines) and add --outdir CLI arg
- [ ] **Part 4:** Update plot_modSO_benchmarks.py for 6 bars, gcc/clang support
- [ ] **Part 5:** Compile with both gcc and clang, run timing, generate plots

## Data File Format (NEW: 6 lines)

Each {model}.txt in benchmark/data/modID_SO/ and modFD_SO/:
```
<FO_analytical_time_us>
<full_SO_analytical_time_us>
<mod_SO_analytical_time_us>
<case1_full_SO_AD_codegen_time_us>
<case2a_AD_full_FO_codegen_time_us>
<case2b_AD_mod_FO_codegen_time_us>
```
(-1 for unavailable cases, e.g. codegen OOM for large models)

## Bar Chart (6 bars per model)

1. FO analytical (cyan)
2. Full SO analytical (orange) — from T-Ro, the slow tensor approach
3. Mod SO analytical (green) — our new algorithm
4. Case 1: Full SO AD codegen (magenta)
5. Case 2a: AD full FO codegen (blue)
6. Case 2b: AD mod FO codegen (red)

Log y-axis. Key visual: Mod SO bar close to FO, far below Full SO.

## Compile Commands

### gcc (timing)
```bash
g++ bench_modID_SO.cpp -DNDEBUG -I /usr/include/eigen3 -O3 \
  -I /home/shubham/Desktop/pinocchio/include -I /usr/include \
  -L /usr/local/lib -lpinocchio -lcasadi -ldl -march=native -o bench_modID_SO
```

### clang (timing)
```bash
clang++ bench_modID_SO.cpp -DNDEBUG -I /usr/include/eigen3 -O3 \
  -I /home/shubham/Desktop/pinocchio/include -I /usr/include \
  -L /usr/local/lib -lpinocchio -lcasadi -ldl -march=native -std=c++11 -o bench_modID_SO_clang
```

### accuracy (no optimization)
```bash
g++ bench_modID_SO_accuracy.cpp -I /usr/include/eigen3 -O0 \
  -I /home/shubham/Desktop/pinocchio/include -I /usr/include \
  -L /usr/local/lib -lpinocchio -lcasadi -ldl -o bench_modID_SO_accuracy
```

## Models
- double_pendulum (2 DOF)
- ur3_robot (6 DOF)
- hyq_f (18 DOF, free-flyer)
- atlas_f (36 DOF, free-flyer)
- talos_full_v2_f (50 DOF, free-flyer)

## Key Reference Files
- `include/pinocchio/algorithm/modrnea.hpp` — modrnea() scalar
- `include/pinocchio/algorithm/modaba.hpp` — modaba() scalar
- `include/pinocchio/algorithm/rnea-second-order-derivatives.hpp` — full SO tensors
- `include/pinocchio/algorithm/mod-rnea-second-order-derivatives.hpp` — mod SO matrices
- `include/pinocchio/utils/tensor_utils.hpp` — tensor helper functions
- `benchmark/FD_SO_deriv.cpp` — full FD SO chain-rule pattern (lines 173-229)
- `benchmark/test_modDynamicsSO.cpp` — mod FD chain-rule + FD validation
