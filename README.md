# Stochastic particle flow filter and smoother on Lorenz 96

This repository holds three Jupyter notebooks that run the Stochastic Particle Flow Filter (SPFF) and the Stochastic
Particle Flow Smoother (SPFS) on the 1000-variable Lorenz 96 model. Each notebook builds its own truth run,
observations and ensemble, runs a sequential data assimilation experiment, and plots the results. The derivations sit
in the markdown cells, so a notebook read from top to bottom also works as a tutorial.

## Notebooks

| Notebook | Method | How observations reach the window start | Kernel width |
| --- | --- | --- | --- |
| `Lorenz96_SPFF.ipynb` | Filter: deterministic `PFF` and stochastic `SPFF` | n/a (no window) | fixed or adaptive |
| `Lorenz96_SPFS.ipynb` | Smoother: `SPFS` | adjoint model | fixed or adaptive |
| `Lorenz96_SPFS_ensAvg.ipynb` | Smoother: `SPFS` | ensemble-averaged sensitivity matrix | fixed |

`Lorenz96_SPFS_ensAvg.ipynb` started as a copy of `Lorenz96_SPFS.ipynb`, so a diff of the two shows what changes
between the adjoint formulation and the ensemble-averaged one.

## How the method works

Particle flow moves a set of interacting particles from the prior to the posterior along a gradient velocity field in
pseudo time $s$. Nothing is reweighted or resampled. Each particle $\mathbf{x}^i$ is updated as

$$
\mathbf{x}^i_{s+1} = \mathbf{x}^i_{s} + \Delta s\,\frac{1}{N_p}\mathbf{B}\sum_{j=1}^{N_p}\left[\mathbf{K}(\mathbf{x}^i_s,\mathbf{x}^j_s)\,\nabla_{\mathbf{x}^j_s}\log p(\mathbf{x}^j_s\mid\mathbf{y}) + \nabla_{\mathbf{x}^j_s}\cdot\mathbf{K}(\mathbf{x}^i_s,\mathbf{x}^j_s)\right] + \sqrt{2\Delta s}\,\boldsymbol{\beta}^i
$$

The kernel term pulls the particles toward the mode of the posterior, and the divergence term pushes them apart so the
ensemble keeps its spread. $\mathbf{B}$, the localized prior covariance, works as a preconditioner. The stochastic
variants add the noise term $\sqrt{2\Delta s}\,\boldsymbol{\beta}^i$, which makes the flow an unbiased sampler for any
ensemble size; the deterministic `PFF` leaves it out. With a single particle ($N_p=1$) the update reduces to a
variational method, so the filter behaves like an ensemble of 3D-Vars and the smoother like an ensemble of 4D-Vars,
with the particles coupled through the kernel.

The smoother updates the particles at the start of the assimilation window. Its gradient adds up one term for every
observation time $t_k$ in the window, and each term has to be carried back to time zero:

$$
\nabla_{\mathbf{x}_{s,0}}\log p(\mathbf{x}_{s,0}\mid\mathbf{y}) = \sum_{k=1}^{L}\widetilde{\mathbf{H}}_k^T\mathbf{R}^{-1}\left(\mathbf{y}_k-\widetilde{H}_k(\mathbf{x}_{s,0})\right) - \mathbf{B}^{-1}(\mathbf{x}_{s,0}-\mathbf{x}_b)
$$

The two smoother notebooks build $\widetilde{\mathbf{H}}_k^T$ in different ways:

$$
\underbrace{\widetilde{\mathbf{H}}_k^T = \mathbf{M}_{0\rightarrow k}^T\mathbf{H}_k^T}_{\text{adjoint}}
\qquad\qquad
\underbrace{\widetilde{\mathbf{H}}_k^T = \mathbf{C}_{\mathbf{x}_0\mathbf{x}_0}^{-1}\mathbf{C}_{\mathbf{x}_0\mathbf{y}_k}}_{\text{ensemble-averaged}}
$$

The adjoint form is exact and gives each particle its own gradient. The ensemble-averaged form is a linear regression
fitted to the particles, so no adjoint model is needed. The price is a single gradient direction shared by all
particles. That direction cannot represent a multimodal sensitivity, the method degrades in strongly nonlinear regimes,
and it needs more frequent observations and a shorter window than the adjoint version.

## Requirements

The notebooks were developed with Python 3.11, NumPy 1.23.5, SciPy 1.11.2, Matplotlib 3.7.2 and Numba 0.59.1. Numba
is not optional: the Lorenz 96 model and the noise computation are JIT-compiled.

```bash
conda create -n research python=3.11 numpy=1.23.5 scipy=1.11.2 matplotlib=3.7.2 numba=0.59.1 jupyterlab
conda activate research
jupyter lab
```

## Running an experiment

Run the cells in order. The algorithm functions read `Nx`, `Ny`, `dt`, `F` and `obs_type` from the notebook's global
namespace instead of taking them as arguments, so after you change a configuration cell, re-run every cell below it.

A good first run is the filter notebook with `obs_type = 'linear'`, followed by the same notebook with `'square'` to
see what a nonlinear observation operator does.

Three settings define an experiment:

1. `obs_type`, in section 2.3, picks the observation operator.
2. `Assim_method` and `iter_num`, in section 3.2, pick `'PFF'` or `'SPFF'` (filter notebook only) and the number of
   pseudo-time iterations per assimilation cycle.
3. `kernel_width_type`, in section 3.2 of the filter (SPFF only) and of the adjoint smoother, is `'fixed'` or
   `'adaptive'`. The fixed option keeps the kernel width at $\alpha_0$. The adaptive one multiplies it once per cycle
   by $\rho=\overline{\left(\sigma(|x|)/\sigma(x)\right)^2}$, computed from the prior ensemble. Because $\rho\le 1$, the
   kernel can only get narrower, and it narrows most for components whose ensemble straddles zero. Those are the
   components where $H(x)=x^2$ makes the posterior bimodal and a wide kernel would pull every particle into one mode.

### Observation operators

Observations cover a quarter of the state. Every fourth variable is observed, $n_y = n_x/4 = 250$ in all, and Gaussian
noise $\mathcal{N}(0,\epsilon\mathbf{I})$ is added to the truth in observation space.

| `obs_type` | $H(x)$ | $\epsilon$ | Likelihood | Available in |
| --- | --- | --- | --- | --- |
| `linear` | $x$ | 0.5 | Gaussian, unimodal | all three notebooks |
| `square` | $x^2$ | 1.0 | bimodal | all three notebooks (see below) |
| `exponential` | $\exp(x/6)$ | 0.01 | unimodal, non-Gaussian | `Lorenz96_SPFS_ensAvg.ipynb` |

`square` runs in all three notebooks, but with the ensemble-averaged smoother it does not converge at any kernel width,
because one shared gradient cannot follow a bimodal likelihood. For bimodal likelihoods, use the adjoint smoother.

### Settings tuned together

`iter_num`, the step size `ds` and the kernel width were tuned for each observation operator. The defaults in the
notebooks are the combinations that converge:

| Notebook | `obs_type` | `iter_num` | `ds` | Kernel width $\alpha_0$ | Observe every | Window |
| --- | --- | --- | --- | --- | --- | --- |
| `Lorenz96_SPFF` | `linear` | 150 | 0.05 | $1/N_p$ (PFF), $n_x$ (SPFF) | 20 | n/a |
| `Lorenz96_SPFF` | `square` | 500 | 0.001 | $1/N_p$ (PFF), $n_x$ (SPFF) | 20 | n/a |
| `Lorenz96_SPFS` | `linear` | 150 | 0.05 | $n_x$ | 10 | 20 |
| `Lorenz96_SPFS` | `square` | 1000 | 0.0001 | $n_x$ | 10 | 20 |
| `Lorenz96_SPFS_ensAvg` | `linear` | 150 | 0.05 | $n_x$ | 5 | 10 |
| `Lorenz96_SPFS_ensAvg` | `exponential` | 500 | 0.001 | $n_x/2$ | 5 | 10 |

Don't borrow the adjoint smoother's settings for the ensemble-averaged one. With observations every 10 steps and a
20-step window, those experiments blow up. For exponential observations, a kernel width of 500 at $n_x = 1000$ is the
value that converges.

`ds` also adapts during a run. When the norm of the velocity field grows by more than 2%, the step shrinks and the
iteration is rolled back; after seven accepted steps it grows again.

## The experiment

All three notebooks use the same perfect-model setup with fixed random seeds, so runs are reproducible. The model is
Lorenz 96 with $n_x = 1000$ and forcing $F = 8$, integrated by fourth-order Runge-Kutta with $\Delta t = 0.01$. The
truth run lasts 2500 steps from uniform random initial conditions, and at its first step the forcing is raised to
$F = 9$ on every fifth component to break the symmetry. At step 1000 the truth is perturbed with
$\mathcal{N}(0, 2\mathbf{I})$ to make a 20-member ensemble. `integration_time_step` (default 500, maximum 1500) sets
how many model steps of sequential assimilation follow.

The prior covariance is localized by a Schur product with a Gaussian distance matrix, with decorrelation length
`r_in = 4`. A 1000×1000 covariance estimated from 20 members is unusable without it, so leave it in.

## Output

Section 4 of each notebook compares the experiment with a free run without assimilation (`NoDA`). It plots:

- the particles moving from prior to posterior in pseudo time, for one assimilation cycle (filter notebook only);
- the ensemble trajectories against the truth for one state component;
- RMSE and ensemble spread in observation space, prior and posterior at every cycle, with the spread scaled by
  $\sqrt{(N_p+1)/N_p}$ for the small ensemble (in a well-calibrated ensemble the two curves track each other);
- the kernel width used at each cycle (filter and adjoint smoother).

## Runtime

Cost of one accepted pseudo-time iteration at $n_x = 1000$, measured on the machine the notebooks were developed on:

| Method | Seconds per iteration |
| --- | --- |
| `PFF` | 0.11 |
| `SPFF` | 0.16 |
| `SPFS`, adjoint | 0.20 |
| `SPFS`, ensemble-averaged | 0.41 |

A full run costs that figure times `iter_num` times the number of assimilation cycles, which is
`integration_time_step / assim_period` for the filter and `integration_time_step / assim_window` for the smoothers.
With the defaults, that ranges from about ten minutes (filter, `linear`) to several hours (`square` or
`exponential`). The smoothers cost more because they re-integrate the whole window after every accepted step.

To try something quickly, lower `integration_time_step` first and `iter_num` second. Setting `Nx = 100` also works,
but the kernel width scales with $n_x$, so the tuned values above no longer carry over.

`SPFF` and `SPFS` print the velocity-field norm, the iteration count and $\max|x|$ at every pseudo-time step, which
adds up to thousands of lines per cycle. Comment those lines out before a long batch run.

## Reference

Yeh, H.-L. & van Leeuwen, P.J. (2026) Unbiased fully nonlinear data assimilation. Part 1: The stochastic particle flow
filter. Quarterly Journal of the Royal Meteorological Society, e70345. Available from: https://doi.org/10.1002/qj.70345

Yeh, H.-L. & van Leeuwen, P.J. (2026) Unbiased fully nonlinear data assimilation. Part II: The stochastic particle flow
smoother. Quarterly Journal of the Royal Meteorological Society, e70346. (In production.)

The matrix-valued kernel in the deterministic `PFF` follows Hu and van Leeuwen (2021).
