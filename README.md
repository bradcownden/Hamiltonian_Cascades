# Stochastic Cascades

GPU-based model for the stochastic evolution of infinite-dimensional resonant nonlinear Hamiltonian systems with a subset of structured couplings.

This code accompanies the paper:

> A. Biasi, B. Cownden, O. Evnin, A. Iturbe Jabaloyes, *Coherent energy cascades in nonlinear disordered Hamiltonians*, [arXiv:2609.36009](https://arxiv.org/abs/2609.36009)

If you use it, please cite the paper and this software (see [CITATION.cff](CITATION.cff)).

---

## Contents

- [Stochastic Cascades](#stochastic-cascades)
  - [Contents](#contents)
  - [Stochastic Coefficients](#stochastic-coefficients)
  - [Requirements](#requirements)
  - [| libquadmath | | needed when Boost `float128` is used; see `MATH_LIBS` |](#-libquadmath---needed-when-boost-float128-is-used-see-math_libs-)
  - [Building](#building)
  - [Quick start](#quick-start)
  - [Timing \& Debugging](#timing--debugging)
  - [Input parameters](#input-parameters)
    - [Initial data](#initial-data)
    - [Time evolution](#time-evolution)
    - [Stochastic model](#stochastic-model)
  - [Environment variables](#environment-variables)
  - [Output files](#output-files)
  - [Code structure](#code-structure)
  - [Reproducibility \& Data](#reproducibility--data)
  - [Limitations and known issues](#limitations-and-known-issues)
  - [License and acknowledgements](#license-and-acknowledgements)

---

## Stochastic Coefficients

While the primary driver of the evolution has remained the same, there are multiple versions of the stochastic evolution that have built upon one another to produce the final version. Corresponding function versions for coefficient initializations are also present. Here we briefly describe each, including how greater capability was added to each version. 
- kernels.cu: `__global__ void build_random_coefficients` re-samples the couplings from a Guassian distribution at every function call.
- kernels.cu: `__global__ void stochastic_coefficients` evolves stochastic coefficients using host clock() seed,  Box-Muller transform for Gaussian distribution, and Ornstein-Uhlenbeck evolution profile. 
- rk_sm.cu: `__global__ void stochastic_coefficients_sm` evolves stochastic coefficients with optional `run_seed` and `step_seed` inputs for validation (default to off), as well as improved memory management

All stochastic evolution data for Type I and II systems were produced with methods from `rk_sm.cu`.

## Requirements

| Dependency | Tested version | Notes |
|---|---|---|
| CUDA toolkit (nvcc, cuRAND) | CUDA 12.3 | `curand_init`, `curand_normal_double` |
| NVIDIA GPU | compute capability ≥ 7.5 | developed on sm_75, production runs on A100 (sm_80) |
| C++17 host compiler | | GCC recommended (provides `__float128`) |
| Boost (Multiprecision, Math) | v1_88 | header-only |
| libquadmath | | needed when Boost `float128` is used; see `MATH_LIBS` |
---

**Note**: libquadmath and float128 are only compatible with x86_64 architectures. For macOS it must be compiled with a specified gcc-14 or equivalent compiler instead of the default `clang` compiler.

## Building

```bash
make    # defaults: sm_75, Boost in /usr/local/include
make ARCH=-arch=sm_80   # e.g. A100
make BOOST_INC=/usr/include    # Boost elsewhere
make clean
```

## Quick start

```bash
make
cp parameters.txt my_run/ && cd my_run
../cascades CASCADES_RK_SM=1
```

## Timing & Debugging
```bash
make
cp parameters.txt my_run/ && cd my_run
../cascades CASCADES_RK_SM=1 CASCADES_TIMING=1 CASCADES_FINITE_DEBUG=1
```

## Input parameters

**Note**: the file `parameters.txt` must be included in the directory where the program is run

### Initial data

| Key | Symbol | Meaning | Allowed values / default |
|---|---|---|---|
| `N` | $N$ | number of modes | ≤ 512 for the single-block RK path |
| `mu0` | $\mu_0$ | Free parameter | Type I, II, III: $\mu_0 > 0$
| `mu1` | $\mu_1$ | Free parameter | Type I: $\mu_1 = 0$<br>Type II, III: $\mu_1 > 0$
| `mu2` | $\mu_2$ | Free parameter | Type I, II: $\mu_2 = 0$<br>Type III: $\mu_2 > 0$
| `beta0` | $\beta_0$ | Free parameter| Any (real) |
| `beta1` | $\beta_1$ | Free parameter| Any (real) |
| `beta2` | $\beta_2$ | Free parameter| Any (real) |
| `eta` | $\eta$ | Initial data scaling power | $\eta \geq 0$|
| `rho` | $\rho$ | Initial data exponential envelope power | $\rho \geq 0$|
| `data_read` | — | Read or generate initial data |  0 = load initial data from `data_in/`<br>1 = generate initial data |
| `ID_type` | — | Initial data family (when `data_read = 1`) | 0: unseeded complex Gaussian <br>1: unseeded complex Gaussian for $M < 10$, complex Gaussian with exponential decay envelope for $M \geq 10$<br>2: fixed exponential decay |
| `data_in` | — | directory for initial-data binaries | default: `input` |
| `data_out` | — | directory for simulation output | see note under [Output files](#output-files) |

### Time evolution

| Key | Symbol | Meaning |
|---|---|---|
| `dt` | $dt$ | RK4 time step |
| `Nt` | $N_t$| Number of applications of RK4 instances before returning to host |
| `Nsave` | $N_{save}$ | Steps between saving to output files |
| `N_total` | $N_{total}$ | Total number evolution loops; $t_f$ = `N_total · Nsave · Nt · dt` |

### Stochastic model

In the limit of small time steps $dt$, the evolution of the couplings $C_{nmkj}$ is modeled as
$C_{nmkj} (t + \Delta t) \approx (1 - \theta dt) C_{nmkj} (t)$ + $\sigma$ $\sqrt{2 \theta dt}$ $\xi_{nmkj}$
where $\xi_{nmkj}$ is a randomly-sampled Gaussian distribution.

| Key | Symbol | Meaning | Default |
|---|---|---|---|
| `theta` | $\theta$ | OU relaxation rate | None |
| `sigma` | $\sigma$ | OU stationary standard deviation | None |
| `rng_seed` | — | $= 0$: fresh random seed each run<br>$\neq 0$: fixed seed (see [Reproducibility](#reproducibility)) | $0$ |

## Environment variables

| Variable | Effect |
|---|---|
| `CASCADES_RK_SM=1` | use the multi-block RK4 implementation (any N; seeded noise) instead of the single-block one |
| `CASCADES_TIMING=1` | print average GPU time per kernel group for each save interval |
| `CASCADES_FINITE_DEBUG=1` | check device arrays for NaN/Inf after every kernel (slow; debugging only) |

`CASCADES_RK_SM=1` was used to produce data for the publication

## Output files

Each run writes to a new directory `output_<k>` (first unused index) in the working directory. In terms of the dynamic variables,
$A_n = Re(\alpha_n)$, $B_n = Im(\alpha_n)$<br>
$J = \sum |\alpha_n|^2$, $E = \sum n |\alpha_n|^2$

| File | Contents |
|---|---|
| `simulation_parameters.txt` | parameters used for this run (including the resolved `rng_seed`) |
| `out.txt` | run log: timing, relative drift of J and E, warnings |
| `A.bin`, `B.bin` | $A = Re(\alpha_n)$, $B = Im(\alpha_n)$: `N` doubles per save, `N_total + 1` saves (incl. t = 0) |
| `t.bin` | time of each save (1 double per save) |
| `J.bin`, `E.bin` | J and E at each save (1 double per save) |
| `V.bin` | unused, always 0 |
| `fload_f.bin` | generating sequence data |
| `fload_logf.bin` | (log) generating sequence data |

The `input/` directory receives `fload_A.bin`, `fload_B.bin`, `fload_f.bin` and `fload_logf.bin` when the initial data is generated. These files can be reused with `data_read = 0`.

All binaries are raw little-endian `float64` without a header. Example reader:

```python
import numpy as np
N = 400  # from simulation_parameters.txt
A = np.fromfile("output_0/A.bin").reshape(-1, N)
B = np.fromfile("output_0/B.bin").reshape(-1, N)
t = np.fromfile("output_0/t.bin")
alpha = A + 1j * B
```

## Code structure

| File | Role |
|---|---|
| `main.cu` | driver: setup, time loop, diagnostics, output |
| `main.cuh` | shared declarations, parameter structs, error-check macros |
| `io.cu` | parameter parsing, paths, launch-configuration checks |
| `initial_data.cpp/.hpp` | high-precision initial data generation |
| `kernels.cu` | coefficient kernels (deterministic, stochastic, odd/even corrections) |
| `rk.cu` | single-block RK4 step (legacy) |
| `rk_sm.cu` | seeded noise kernels and multi-block RK4 step |
| `bigfloat.hpp`, `complex128.hpp` | high-precision real/complex types (Boost) |

## Reproducibility & Data

Parameter values for generating figure data (see Zenodo DOI for data files) use $\theta = 1, \sigma=\sqrt{\mu_0}$ and:
- Type III cascades: $\mu_0 = 3, \mu_1 = 1, \mu_2 = 1, \beta_0 = 0, \beta_1 = 0, \beta_2 = 0$
- Type II cascades: $\mu_0 = 1, \mu_1 = 1, \mu_2 = 0, \beta_0 = -2.5, \beta_1 = 2.5, \beta_2 = 0$
- Type I cascades: $\mu_0 = 1, \mu_1 = 0, \mu_2 = 0, \beta_0 = -3.5, \beta_1 = -2.5, \beta_2 = 0$

Data is available for the above at [10.5281/zenodo.23164241](https://doi.org/10.5281/zenodo.23164241)

## Limitations and known issues

There is a limit for the number of modes that can be included during a run, based on the hardware being utilised. Hardcoded limits are set in `main.cu:22` to be
```bash
#define BLOCK_SIZE       1024
```


## License and acknowledgements

Released under the MIT License (see [LICENSE](LICENSE)).

We are grateful for the support from:
- “la Caixa” Foundation (ID 100010434) fellow-
ship LCF/BQ/PI24/12040029 
- Mar&iacute;a de Maeztu grant CEX2023-001318-M funded by
MICIU/AEI /10.13039/501100011033
- The Xunta de Galicia (CIGUS Network of Research Centres
and grant ED431C-2025/11)
- IGFAE Summer Fellowships 2025
- This work benefited from
the use of the infrastructures provided by the Galician Supercomputing Center (CESGA)

Third-party components:
- [Boost.Multiprecision](https://www.boost.org/) (Boost Software License 1.0)
- NVIDIA cuRAND, Philox4x32-10 generator (Salmon et al., SC '11)
- Box-Muller sampling following GPU Gems 3, ch. 37 (https://developer.nvidia.com/gpugems/gpugems3)
