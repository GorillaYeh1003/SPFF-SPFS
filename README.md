# Stochastic Particle Flow Filter and Smoother — Lorenz 96 examples

Runnable tutorial notebooks for the **Stochastic Particle Flow Filter (SPFF)** and the **Stochastic Particle Flow
Smoother (SPFS)**, demonstrated on the 1000-dimensional Lorenz 96 system.

Particle flow methods push a set of interacting particles from the prior to the posterior along a gradient velocity
field in pseudo time, with no reweighting and no resampling. The stochastic variant adds a noise term, which makes it
an unbiased sampler for any ensemble size. Each notebook is self-contained — model, truth run, observations, algorithm,
sequential DA experiment, and diagnostics — and carries the derivations in its own markdown cells, so it can be read
top to bottom as a tutorial.

## The notebooks

| notebook | method | backward propagation | kernel width |
| --- | --- | --- | --- |
| `Lorenz96_SPFF.ipynb` | **filter** — deterministic `PFF` and stochastic `SPFF` | — | fixed or adaptive |
| `Lorenz96_SPFS.ipynb` | **smoother** — `SPFS` | exact **adjoint** model | fixed or adaptive |
| `Lorenz96_SPFS_ensAvg.ipynb` | **smoother** — `SPFS` | **ensemble-averaged sensitivity matrix** | fixed |

The two smoother notebooks are otherwise identical, so they diff cleanly against each other.

## Method in brief

Each particle $\mathbf{x}^i$ is advanced in pseudo time $s$ by

$$
\mathbf{x}^i_{s+1} = \mathbf{x}^i_{s} + \Delta s\,\frac{1}{N_p}\mathbf{B}\sum_{j=1}^{N_p}
\left[\mathbf{K}(\mathbf{x}^i_s,\mathbf{x}^j_s)\,\nabla_{\mathbf{x}^j_s}\log p(\mathbf{x}^j_s\mid\mathbf{y}) + \nabla_{\mathbf{x}^j_s}\cdot\mathbf{K}(\mathbf{x}^i_s,\mathbf{x}^j_s)\right] + \sqrt{2\Delta s}\,\boldsymbol{\beta}^i
$$

The first kernel term attracts particles toward the posterior mode; the second repels them, keeping the ensemble spread
open. The noise term $\boldsymbol{\beta}^i$ is what distinguishes the stochastic variants from the deterministic `PFF`,
and the localized prior covariance $\mathbf{B}$ acts as a preconditioner. With $N_p=1$ the flow reduces to a
variational method, which is why the filter behaves like an ensemble of 3D-Vars and the smoother like an ensemble of
4D-Vars — except that here the particles interact through the kernel.

The smoother updates the particles at the **start** of the assimilation window, so the gradient sums contributions from
every observation time $t_k$ in the window and each one has to be propagated back to time zero:

$$
\nabla_{\mathbf{x}_{s,0}}\log p(\mathbf{x}_{s,0}\mid\mathbf{y}) =
\sum_{k=1}^{L}\widetilde{\mathbf{H}}_k^T\mathbf{R}^{-1}\left(\mathbf{y}_k-\widetilde{H}_k(\mathbf{x}_{s,0})\right)
-\mathbf{B}^{-1}(\mathbf{x}_{s,0}-\mathbf{x}_b)
$$

The two smoother notebooks differ only in how $\widetilde{\mathbf{H}}_k^T$ is formed:

$$
\underbrace{\widetilde{\mathbf{H}}_k^T = \mathbf{M}_{0\rightarrow k}^T\mathbf{H}_k^T}_{\text{adjoint}}
\qquad\qquad
\underbrace{\widetilde{\mathbf{H}}_k^T = \mathbf{C}_{\mathbf{x}_0\mathbf{x}_0}^{-1}\mathbf{C}_{\mathbf{x}_0\mathbf{y}_k}}_{\text{ensemble-averaged}}
$$

The adjoint form is exact and gives every particle its own gradient. The ensemble-averaged form is a linear regression
estimated from the particles themselves, so it needs no adjoint model — but it gives one common gradient direction to
all particles, which cannot represent a multimodal sensitivity. It therefore needs a higher observation frequency and a
shorter window, and it degrades in strongly nonlinear regimes.

## Requirements

Python 3.11 with NumPy, SciPy, Matplotlib and Numba. The versions these notebooks were developed against:

```
python 3.11    numpy 1.23.5    scipy 1.11.2    matplotlib 3.7.2    numba 0.59.1
```

```bash
conda create -n research python=3.11 numpy=1.23.5 scipy=1.11.2 matplotlib=3.7.2 numba=0.59.1 jupyterlab
conda activate research
jupyter lab
```

Numba is required — the model and the noise computation are JIT-compiled.

## Running an experiment

Run the cells top to bottom. The algorithm routines read `Nx`, `Ny`, `dt`, `F` and `obs_type` from the notebook
namespace rather than taking them as arguments, so after changing any configuration cell you must re-run the cells
below it.

Three choices drive an experiment, each in its own cell:

1. **`obs_type`** (section 2.3) — the observation operator.
2. **`Assim_method`** and **`iter_num`** (section 3.2) — `'PFF'` or `'SPFF'` in the filter notebook, and the number of
   pseudo-time iterations per assimilation cycle.
3. **`kernel_width_type`** (section 3.2, filter and adjoint smoother) — `'fixed'` keeps the kernel width at
   $\alpha_0$; `'adaptive'` rescales it once per cycle by $\rho=\overline{\left(\sigma(|x|)/\sigma(x)\right)^2}$
   diagnosed from the prior. Since $\rho\le 1$ this can only narrow the kernel, and it narrows most where the ensemble
   straddles zero — exactly where $H(x)=x^2$ makes the posterior bimodal and a wide kernel would drag all the particles
   into a single mode.

### Observation operators

Observations cover 25% of the system, sampling every fourth variable ($n_y = n_x/4 = 250$), with Gaussian noise
$\mathcal{N}(0,\epsilon\mathbf{I})$ added to the truth in observation space.

| `obs_type` | $H(x)$ | $\epsilon$ | likelihood | available in |
| --- | --- | --- | --- | --- |
| `linear` | $x$ | 0.5 | Gaussian, unimodal | all three notebooks |
| `square` | $x^2$ | 1.0 | bimodal | all three, but see the note below |
| `exponential` | $\exp(x/6)$ | 0.01 | unimodal but non-Gaussian | `Lorenz96_SPFS_ensAvg.ipynb` |

`square` is wired up in all three notebooks, but it is not expected to work with the ensemble-averaged smoother: a
single averaged gradient cannot capture a bimodal likelihood, and those experiments fail to converge at any kernel
width. Use the adjoint smoother for bimodal likelihoods.

### Settings that belong together

`iter_num`, the step size `ds` and the kernel width are tuned per observation operator. The values built into the
notebooks are the ones that converge:

| notebook | `obs_type` | `iter_num` | `ds` | kernel width $\alpha_0$ | obs. every | window |
| --- | --- | --- | --- | --- | --- | --- |
| `Lorenz96_SPFF` | `linear` | 150 | 0.05 | $1/N_p$ (PFF), $n_x$ (SPFF) | 20 | — |
| `Lorenz96_SPFF` | `square` | 500 | 0.001 | same | 20 | — |
| `Lorenz96_SPFS` | `linear` | 150 | 0.05 | $n_x$ | 10 | 20 |
| `Lorenz96_SPFS` | `square` | 1000 | 0.0001 | $n_x$ | 10 | 20 |
| `Lorenz96_SPFS_ensAvg` | `linear` | 150 | 0.05 | $n_x$ | 5 | 10 |
| `Lorenz96_SPFS_ensAvg` | `exponential` | 500 | 0.001 | $n_x/2$ | 5 | 10 |

The ensemble-averaged smoother's shorter period and window are not cosmetic: with the adjoint's 10/20 setting those
experiments blow up. Its kernel width is equally load-bearing — 500 (at $n_x=1000$) is the value that converges.

Inside the flow routine the pseudo-time step `ds` then adapts on its own: if the velocity-field norm grows by more than
2% the step is reduced and the iteration rolled back, and after seven accepted steps it grows again.

## The experimental setup

All three notebooks share the same perfect-model experiment, with fixed seeds so runs are reproducible:

- **Model** — Lorenz 96, $n_x = 1000$, forcing $F = 8$, fourth-order Runge–Kutta at $\Delta t = 0.01$.
- **Truth** — 2500 steps from uniform random initial conditions, with a one-time forcing perturbation
  ($F = 9$ on every fifth component) at the first step to break symmetry.
- **Ensemble** — 20 members, drawn at model step 1000 by perturbing the truth with $\mathcal{N}(0, 2\mathbf{I})$.
- **Localization** — the prior covariance is regularized by a Schur product with a Gaussian distance matrix
  (decorrelation length scale `r_in = 4`). This is what makes a 1000×1000 covariance usable with 20 members; do not
  remove it.

`integration_time_step` (default 500, maximum 1500) sets how many model steps of sequential DA to run.

## What the notebooks produce

Section 4 of each notebook compares the experiment against a no-assimilation (`NoDA`) free run:

- the particles flowing from prior to posterior in pseudo time, for a single assimilation cycle (filter notebook);
- the ensemble trajectories against the truth for one state component;
- RMSE and ensemble spread in observation space, prior and posterior at each cycle, with the small-ensemble spread
  adjustment $\sqrt{(N_p+1)/N_p}$ — for a well-calibrated ensemble these two curves should track each other;
- the kernel width actually used at each cycle (filter and adjoint smoother).

## Runtime

These are research-scale runs. Measured on one laptop at $n_x = 1000$, a single accepted pseudo-time iteration costs
roughly:

| | s / iteration |
| --- | --- |
| `PFF` | 0.11 |
| `SPFF` | 0.16 |
| `SPFS` (adjoint) | 0.20 |
| `SPFS` (ensemble-averaged) | 0.41 |

Multiply by `iter_num` and by the number of assimilation cycles — `integration_time_step / assim_period` for the
filter, `/ assim_window` for the smoothers. The defaults therefore range from roughly ten minutes for a linear
experiment to several hours for `square` or `exponential`. Both smoothers re-integrate the whole window forward after
every accepted step, which is what makes them the expensive notebooks.

For a quick look, lower `integration_time_step` first, then `iter_num`; reducing `Nx` to 100 also works, but the kernel
width scales with $n_x$, so the tuned values above no longer apply directly.

Note that `SPFF` and `SPFS` print the velocity-field norm, the iteration count and $\max|x|$ at every pseudo-time step,
which is thousands of lines per cycle. Comment those prints out before a long batch run.

## Reference

Yeh, H.-L., and P. J. van Leeuwen. *Unbiased fully nonlinear data assimilation. Part II: the Stochastic Particle Flow
Smoother.* Department of Atmospheric Science, Colorado State University, and Department of Meteorology, University of
Reading.

Part I of the same series introduces the Stochastic Particle Flow Filter. The matrix-valued kernel used by the
deterministic `PFF` follows Hu and van Leeuwen (2021).
