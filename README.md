# Statistical Inference for the CSPALT-UGR Model
### Classical & Bayesian Estimation under Progressive Type-II Censoring

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [The UGR Distribution](#2-the-ugr-distribution)
3. [The CSPALT Setup](#3-the-cspalt-setup)
4. [Progressive Type-II Censoring](#4-progressive-type-ii-censoring)
5. [Censoring Schemes Used](#5-censoring-schemes-used)
6. [Classical Estimation — MLE](#6-classical-estimation--mle)
7. [Bayesian Estimation](#7-bayesian-estimation)
8. [MCMC via Metropolis-Hastings Gibbs](#8-mcmc-via-metropolis-hastings-gibbs)
9. [Loss Functions](#9-loss-functions)
10. [Interval Estimation](#10-interval-estimation)
11. [Performance Metrics](#11-performance-metrics)
12. [Simulation Study](#12-simulation-study)
13. [Real Data Analysis](#13-real-data-analysis)
14. [Output Files Guide](#14-output-files-guide)
15. [How to Run the Code](#15-how-to-run-the-code)
16. [Key Results Summary](#16-key-results-summary)
17. [Repository Structure](#17-repository-structure)

---

## 1. Project Overview

This project is a **B.Tech thesis** implementing classical and Bayesian inference for the
**Constant Stress Partially Accelerated Life Test (CSPALT)** model, where lifetimes follow
the **Unit Generalized Rayleigh (UGR)** distribution under **Progressive Type-II censoring**.

**Goal:** Estimate three model parameters — `α` (shape), `β` (scale), `λ` (acceleration
factor) — using multiple competing methods and compare their performance.

**Methods implemented:**

| Method | Type | Estimator |
|---|---|---|
| MLE | Classical | Point estimate via L-BFGS-B optimization |
| ACI | Classical | Asymptotic Confidence Interval (Fisher information) |
| Bayes-SELF (Gamma prior) | Bayesian | Posterior mean |
| Bayes-LINEX s=+2 (Gamma prior) | Bayesian | Asymmetric loss estimate |
| Bayes-LINEX s=-2 (Gamma prior) | Bayesian | Asymmetric loss estimate |
| Bayes-SELF (Gamma-Dirichlet prior) | Bayesian | Posterior mean |
| Bayes-LINEX s=+2 (GD prior) | Bayesian | Asymmetric loss estimate |
| Bayes-LINEX s=-2 (GD prior) | Bayesian | Asymmetric loss estimate |
| BCI | Bayesian | Equal-tail credible interval |

---

## 2. The UGR Distribution

The **Unit Generalized Rayleigh (UGR)** distribution is defined on the interval (0, 1),
making it naturally suited for modelling lifetimes expressed as proportions or reliability
scores.

A random variable X ~ UGR(α, β) has the following functions (Normal stress condition):

**PDF:**
```
f₁(x; α, β) = 2αβ²·(-ln x / x)·exp(-(β·(-ln x))²)·[1 - exp(-(β·(-ln x))²)]^(α-1)
```

**Survival Function:**
```
S₁(x; α, β) = [1 - exp(-(β·(-ln x))²)]^α
```

**CDF:**
```
F₁(x; α, β) = 1 - S₁(x; α, β)
```

**Parameters:**
- `α > 0` — shape parameter controlling tail heaviness
- `β > 0` — scale parameter

**Inverse CDF (used for simulation):**
```
F₁⁻¹(u) = exp(-sqrt(-ln(1 - (1-u)^(1/α))) / β)
```

---

## 3. The CSPALT Setup

In a **Constant Stress Partially Accelerated Life Test (CSPALT)**, units are split into
two groups:

- **Group 1 — Normal stress:** Units run under standard operating conditions.
  Lifetimes follow UGR(α, β).

- **Group 2 — Accelerated stress:** Units run under elevated stress.
  Lifetimes are compressed by an **acceleration factor λ > 1**, so they follow UGR(αλ, β).

**Accelerated condition functions:**

```
f₂(x; α, β, λ) = 2αβ²λ·(-ln x / x)·exp(-(β·(-ln x))²)·[1 - exp(-(β·(-ln x))²)]^(αλ-1)

S₂(x; α, β, λ) = [1 - exp(-(β·(-ln x))²)]^(αλ)
```

The constraint `λ > 1` means the accelerated stress always shortens lifetime — this is
enforced throughout the estimation.

**Parameters to estimate:** θ = (α, β, λ)

---

## 4. Progressive Type-II Censoring

**Why censoring?** In life-testing, it is impractical (or too expensive) to observe all
failures. Censoring lets the experimenter terminate early while preserving statistical
information.

**Progressive Type-II censoring** generalises ordinary Type-II censoring:

1. Fix total sample size `n` and number of observed failures `m < n`.
2. Specify a **censoring scheme** R = (R₁, R₂, …, Rₘ) where Rᵢ is the number of
   surviving units removed at the i-th failure.
3. Constraint: `m + ΣRᵢ = n`.

At each of the `m` ordered failures, a predetermined number Rᵢ of the surviving units
are randomly withdrawn from the test. This allows interim removals, not just end-of-study
removals.

**In this study:** n = 40, m = 30 → 10 total censored units distributed according to
the scheme R.

---

## 5. Censoring Schemes Used

Seven distinct censoring patterns are studied (all satisfy m=30, n=40):

| Scheme | Pattern | Description |
|---|---|---|
| **CS1** | R = (0,…,0, **10**) | All 10 removals at the last failure — right censoring |
| **CS2** | R = (**10**, 0,…,0) | All 10 removals at the first failure — left censoring |
| **CS3** | R = (1,1,…,1,**10×ones**,0,…) | 1 unit removed at each of the first 10 failures |
| **CS4** | R = (0,…,0, **10**, 0,…) | 10 removals at the 11th failure — middle censoring |
| **CS5** | R = (2,2,2,2,2, 0,…) | 2 units removed at each of the first 5 failures |
| **CS6** | R = (0,…,0, **5**, 0,…) | 5 removals at the 16th failure — late-middle censoring |
| **CS7** | R = (0,…,0, **3**, 0,…, **2**, 0,…) | Split removals: 3 at step 6, 2 at step 17 |

These schemes test robustness: CS1 is the most information-rich (all data until the end),
CS2 is the most aggressive early withdrawal.

---

## 6. Classical Estimation — MLE

**Maximum Likelihood Estimation (MLE)** finds θ̂ = (α̂, β̂, λ̂) that maximises the
log-likelihood function:

```
ℓ(θ) = Σ ln f₁(x₁ᵢ; α,β) + Σ Rᵢ·ln S₁(x₁ᵢ; α,β)
      + Σ ln f₂(x₂ⱼ; α,β,λ) + Σ Rⱼ·ln S₂(x₂ⱼ; α,β,λ)
```

The first sum is over the `m` observed normal-stress failures, with the Rᵢ terms
accounting for the progressively censored units. Similarly for the accelerated group.

**Implementation:**
```r
optim(par = c(1, 0.5, 2),
      fn  = logLik_CSPALT,
      method = "L-BFGS-B",
      lower  = c(0.001, 0.001, 1.001))  # enforces λ > 1
```

**Standard Errors** are derived from the **Fisher information matrix** (inverse of the
observed Hessian at the MLE), computed numerically via `numDeriv::hessian()`.

---

## 7. Bayesian Estimation

Bayesian inference combines the likelihood with a **prior distribution** over θ via
Bayes' theorem:

```
π(θ | data) ∝ L(data | θ) · π(θ)
```

Two competing priors are implemented:

### 7.1 Gamma Prior
Independent Gamma priors on each parameter:
```
α ~ Gamma(a₁=2, b₁=1)
β ~ Gamma(a₂=2, b₂=1)
λ ~ Gamma(a₃=2, b₃=1)
```

This is a conjugate-style, weakly informative prior. The hyperparameters (a=2, b=1) give
a prior mean of 2 and allow broad uncertainty.

### 7.2 Gamma-Dirichlet (GD) Prior
A **joint** prior that couples α and β through a Dirichlet-like structure:

```
log π_GD(α, β, λ) =
    lgamma(a₁+a₂) - lgamma(a₁) - lgamma(a₂)
  + (a₀ - a₁ - a₂)·ln(α+β)
  + (a₁-1)·ln(α) + (a₂-1)·ln(β)
  - b₀·(α+β)
  + ln Gamma(λ; a₃, b₃)
```

where a₀=3, b₀=1, a₁=2, a₂=2, a₃=2, b₃=1.

The GD prior captures **dependence between α and β**, which is realistic since both
govern the shape of the same distribution. λ remains independent.

---

## 8. MCMC via Metropolis-Hastings Gibbs

Since the posterior has no closed form, samples are drawn using a
**component-wise Metropolis-Hastings (MH) within Gibbs** sampler:

**Algorithm:**
```
For each MCMC iteration i = 2, …, 10000:
  1. Propose α* ~ |Normal(αᵢ₋₁, σ_α=0.05)|
     Accept with probability min(1, π(α*,β,λ|data) / π(αᵢ₋₁,β,λ|data))

  2. Propose β* ~ |Normal(βᵢ₋₁, σ_β=0.02)|
     Accept with probability min(1, π(α,β*,λ|data) / π(α,βᵢ₋₁,λ|data))

  3. Propose λ* ~ Normal(λᵢ₋₁, σ_λ=0.02)  [reflected at λ=1]
     Accept with probability min(1, π(α,β,λ*|data) / π(α,β,λᵢ₋₁|data))
```

**MCMC Settings:**
```
Total iterations : 10,000
Burn-in          :  2,500   (first 25% discarded)
Thinning         :      5   (every 5th sample kept)
Effective draws  :  1,500   per parameter
```

The small proposal widths (0.05 for α, 0.02 for β and λ) are tuned to achieve
reasonable acceptance rates (~20–50%), visible in the console output.

**Convergence is assessed via Trace Plots** (see Section 21 of the R code and the
PNG files).

---

## 9. Loss Functions

Bayesian point estimates depend on the choice of **loss function**:

### Squared Error Loss (SELF)
```
L_SELF(θ̂, θ) = (θ̂ - θ)²
```
The Bayes estimator under SELF is the **posterior mean**:
```
θ̂_SELF = E[θ | data] ≈ (1/N) Σ θ⁽ⁱ⁾
```

### LINEX Loss (Linear-Exponential)
```
L_LINEX(θ̂, θ) = exp(s·(θ̂ - θ)) - s·(θ̂ - θ) - 1
```
The Bayes estimator under LINEX is:
```
θ̂_LINEX = -(1/s) · ln E[exp(-s·θ) | data]
         ≈ -(1/s) · ln[(1/N) Σ exp(-s·θ⁽ⁱ⁾)]
```

Two values of `s` are used:

| `s` | Behaviour |
|---|---|
| `s = +2` | Penalises **overestimation** more heavily |
| `s = -2` | Penalises **underestimation** more heavily |

LINEX is particularly useful in reliability engineering where asymmetric costs apply
(e.g., shipping a product with shorter-than-expected life is worse than being conservative).

---

## 10. Interval Estimation

### Asymptotic Confidence Interval (ACI)
Based on MLE + Fisher information:
```
CI: θ̂ ± z_{α/2} · SE(θ̂)      where SE = sqrt(diag(Fisher⁻¹))
```
α = 0.05 → z = 1.96. This interval is valid asymptotically (large sample).

### Bayesian Credible Interval (BCI)
Equal-tail interval from the MCMC posterior sample:
```
BCI: [Q_{0.025}(θ⁽ⁱ⁾), Q_{0.975}(θ⁽ⁱ⁾)]
```
Unlike a frequentist CI, the BCI has the direct probability interpretation:
"There is a 95% posterior probability that θ lies in this interval."

---

## 11. Performance Metrics

For a simulation with `NSIM = 100` replications, each estimator θ̂ is evaluated by:

| Metric | Formula | Meaning |
|---|---|---|
| **AE** (Average Estimate) | (1/N)·Σ θ̂ⱼ | Closeness to true value (bias indicator) |
| **MSE** (Mean Squared Error) | (1/N)·Σ (θ̂ⱼ - θ)² | Combined bias + variance |
| **CP** (Coverage Probability) | (1/N)·Σ 𝟙[θ ∈ CIⱼ] | Should be ≈ 0.95 for a good CI |
| **AL** (Average Length) | (1/N)·Σ (Upperⱼ - Lowerⱼ) | Precision — shorter is better |

A good estimator has **small MSE** and a good CI has **CP near 0.95 with small AL**.

---

## 12. Simulation Study

The full simulation loops over:
- **2 parameter settings:**
  - Setting 1: (α, β, λ) = (1.6, 0.5, 2)
  - Setting 2: (α, β, λ) = (2.5, 1.2, 3)
- **7 censoring schemes** (CS1–CS7)
- **100 simulations** per configuration

**Data generation** uses the probability integral transform on the inverse CDF:
```r
Generate_Progressive(n=40, m=30, R=R, alpha, beta, lambda, accelerated=TRUE/FALSE)
```

The algorithm uses the **Balakrishnan-Sandhu** method for generating progressively
censored order statistics via a product of Beta random variables.

**Total configurations:** 2 × 7 = 14 per method, 9 methods → results in 9 output tables.

---

## 13. Real Data Analysis

**Dataset:** Insulating fluid breakdown times (in minutes) from a dielectric breakdown
experiment. Data are log-transformed to fit the (0,1) support of the UGR distribution:

```
Normal stress:   x_normal = exp(-c(7.74, 17.05, 20.46, 21.02, 22.66,
                                    43.40, 47.30, 139.07))
Accelerated:     x_acc    = exp(-c(0.27, 0.40, 0.69, 0.79, 2.75,
                                    3.91, 9.88, 13.95, 15.93, 27.80))
```

Both MLE and Bayesian (Gamma prior, Gamma-Dirichlet prior) estimates are computed on
this data, with BCI intervals reported.

**Results summary (Gamma prior):**

| Parameter | MLE | SELF | LINEX (s=2) | 95% BCI |
|---|---|---|---|---|
| α | 0.2214 | 0.2181 | 0.2147 | [0.1220, 0.3473] |
| β | 0.0140 | 0.0154 | 0.0154 | [0.0089, 0.0227] |
| λ | 1.001  | 1.177  | 1.154  | [1.0048, 1.6104] |

The acceleration factor λ > 1 is confirmed, though it is close to 1, suggesting a
moderate acceleration effect in this dataset.

---

## 14. Output Files Guide

| File | Contents |
|---|---|
| `Table1_MLE.csv` | MLE point estimates: AE and MSE for all 14 configurations |
| `Table2_ACI.csv` | Asymptotic CI: coverage probability (CP) and average length (AL) |
| `Table3_SELF_Gamma.csv` | Bayes SELF under Gamma prior: AE and MSE |
| `Table4_LINEX2_Gamma.csv` | Bayes LINEX (s=+2) under Gamma prior |
| `Table5_LINEXN2_Gamma.csv` | Bayes LINEX (s=−2) under Gamma prior |
| `Table6_BCI.csv` | Bayesian credible interval: CP and AL |
| `Table7_SELF_GD.csv` | Bayes SELF under Gamma-Dirichlet prior |
| `Table8_LINEX2_GD.csv` | Bayes LINEX (s=+2) under GD prior |
| `Table9_LINEXN2_GD.csv` | Bayes LINEX (s=−2) under GD prior |
| `RealData_Results.csv` | Real data estimates (Gamma prior) |
| `RealData_Results_100.csv` | Real data estimates (MCMC with N=100 chains) |
| `RealData_Results_20.csv` | Real data estimates (MCMC with N=20 chains) |
| `TracePlot_Alpha.png` | MCMC trace plot for α (real data) |
| `TracePlot_Beta.png` | MCMC trace plot for β (real data) |
| `TracePlot_Lambda.png` | MCMC trace plot for λ (real data) |
| `TracePlot_Alpha_100.png` | Trace for α with 100-iteration burn-in variant |
| `TracePlot_Beta_100.png` | Trace for β with 100-iteration burn-in variant |
| `TracePlot_Lambda_100.png` | Trace for λ with 100-iteration burn-in variant |
| `2201MC30_THESIS.pdf` | Full B.Tech thesis with derivations and analysis |
| `Estimation Code.R` | Complete R source code |

---

## 15. How to Run the Code

### Prerequisites
```r
install.packages(c("stats", "numDeriv", "MASS", "coda", "ggplot2"))
```

### Quick start
```r
source("Estimation Code.R")
```

### Global settings (top of script)
```r
NSIM   <- 100    # number of simulation replications
N_MCMC <- 10000  # total MCMC iterations
BURNIN <- 2500   # iterations to discard
THIN   <- 5      # thinning interval
```

Reducing `NSIM` to 10 and `N_MCMC` to 2000 gives a fast test run in ~5 minutes.
The full run (NSIM=100, N_MCMC=10000, all 14 settings) takes several hours.

### Sections at a glance

| Section | What it does |
|---|---|
| 1–2 | Load libraries, set seed, global constants |
| 3 | UGR PDF, CDF, survival functions (normal + accelerated) |
| 4 | Inverse CDF for data generation |
| 5–6 | Define 7 censoring schemes and 2 parameter settings |
| 7 | Progressive sample generator |
| 8–9 | Log-likelihood and MLE via optim + Hessian |
| 10–12 | Gamma and GD log-priors, log-posteriors |
| 13 | MH-within-Gibbs MCMC sampler |
| 14–15 | Bayes estimators (SELF, LINEX) and credible intervals |
| 16 | MSE, ABS, CP, AL performance metrics |
| 17–18 | Result data frames + main simulation loop |
| 19 | Export all tables to CSV |
| 20–21 | Real data analysis + trace plots |

---

## 16. Key Results Summary

From the simulation study (representative findings):

**MLE** performs well but can be biased for small effective sample sizes (aggressive
censoring like CS2). The MSE for λ tends to be larger than for α and β because λ is
constrained to (1, ∞) and the boundary at 1.001 is hit in some replications.

**Coverage probability** for ACI is generally close to 0.95 (range 0.88–0.97 across
schemes), with CS6 showing slightly lower coverage for λ.

**Bayesian SELF estimates** under both priors achieve **lower MSE** than MLE in most
configurations, especially for Setting 1. The GD prior provides marginally tighter
estimates than the independent Gamma prior when α and β are both small.

**LINEX with s=+2 vs s=−2:** When true λ is large (Setting 2, λ=3), underestimation
is more common and s=−2 (penalising underestimation) yields lower average loss.

**Censoring scheme impact:** CS4 and CS2 tend to produce the lowest MSE for α,
suggesting that withdrawals concentrated early or at a specific middle point preserve
more tail information about the UGR distribution.

---

## 17. Repository Structure

```
Statistical-Inference-main/
│
├── Estimation Code.R            ← Main R script (all 21 sections)
│
├── 2201MC30_THESIS.pdf          ← Full thesis document
│
├── RealData_Results.csv         ← Real data: all estimators
├── RealData_Results_100.csv     ← Real data: 100-chain variant
├── RealData_Results_20.csv      ← Real data: 20-chain variant
│
├── Table1_MLE.csv               ← Simulation: MLE
├── Table1_MLE_Results_100.csv   ← MLE with n=100 variant
├── Table1_MLE_Results_20.csv    ← MLE with n=20 variant
├── Table2_ACI.csv               ← ACI intervals
├── Table2_SELF_Gamma_100.csv
├── Table2_SELF_Gamma_20.csv
├── Table3_SELF_Gamma.csv        ← Bayes SELF / Gamma prior
├── Table3_LINEX2_Gamma_100.csv
├── Table3_LINEX2_Gamma_20.csv
├── Table4_LINEX2_Gamma.csv      ← Bayes LINEX s=+2 / Gamma
├── Table4_LINEXN2_Gamma_100.csv
├── Table4_LINEXN2_Gamma_20.csv
├── Table5_LINEXN2_Gamma.csv     ← Bayes LINEX s=−2 / Gamma
├── Table5_SELF_GD_100.csv
├── Table5_SELF_GD_20.csv
├── Table6_BCI.csv               ← Bayesian credible intervals
├── Table7_SELF_GD.csv           ← Bayes SELF / GD prior
├── Table8_LINEX2_GD.csv         ← Bayes LINEX s=+2 / GD
├── Table9_LINEXN2_GD.csv        ← Bayes LINEX s=−2 / GD
│
├── TracePlot_Alpha.png          ← MCMC convergence: α
├── TracePlot_Beta.png           ← MCMC convergence: β
├── TracePlot_Lambda.png         ← MCMC convergence: λ
├── TracePlot_Alpha_100.png
├── TracePlot_Beta_100.png
├── TracePlot_Lambda_100.png
├── TracePlot_Alpha_20.png
├── TracePlot_Beta_20.png
└── TracePlot_Lambda_20.png
```

---

## Glossary

| Term | Meaning |
|---|---|
| UGR | Unit Generalized Rayleigh — a distribution on (0,1) |
| CSPALT | Constant Stress Partially Accelerated Life Test |
| MLE | Maximum Likelihood Estimator |
| ACI | Asymptotic Confidence Interval (frequentist) |
| MCMC | Markov Chain Monte Carlo |
| MH | Metropolis-Hastings acceptance-rejection step |
| SELF | Squared Error Loss Function |
| LINEX | Linear-Exponential loss function |
| BCI | Bayesian Credible Interval (equal-tail) |
| GD prior | Gamma-Dirichlet joint prior on (α, β) |
| CP | Coverage Probability |
| AL | Average Length of interval |
| MSE | Mean Squared Error |
| AE | Average Estimate (across simulations) |
| Burn-in | Initial MCMC samples discarded before convergence |
| Thinning | Keeping every k-th MCMC sample to reduce autocorrelation |

---

*B.Tech Thesis — Statistical Inference Project*
*Roll No.: 2201MC30*
