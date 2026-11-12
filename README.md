# ESTG: Planck 2018 + BBN constraints and LISA prospects

Code and paper for **Entropic Scalar-Tensor Gravity (ESTG)**, a scalar-tensor
theory with curvature coupling F(φ) = M_Pl² exp(2βφ/M_Pl).

- Paper: [`paper/ESTG_LISA_Paper_Final.pdf`](paper/ESTG_LISA_Paper_Final.pdf)
  (LaTeX source alongside)
- Analysis code: [`analysis/`](analysis/) (detailed instructions in Turkish:
  [`analysis/README.md`](analysis/README.md))

## Main results

ESTG with a canonical kinetic term is Brans–Dicke theory with
ω_BD = 1/(4β²). Its cosmological limit is Brans–Dicke + Λ, which we solve with
`hi_class` and fit to the Planck 2018 likelihoods and a BBN prior.

| Quantity | Result (Planck 2018 + BBN, profile likelihood) |
|---|---|
| Coupling | β < 0.024 (95%, one-sided) |
| Brans–Dicke parameter | ω_BD > 421 |
| Running of the Planck mass today | α_M,0 < 3.7 × 10⁻³ |
| Best fit | β = 0 (GR); ΔAIC = +2 |
| LISA standard sirens, 10–30 events | σ(α_M,0) ≈ 0.04–0.07 (weaker than the CMB bound) |

Also shown in the paper: the effective-field-theory parameterisation
α_M ∝ Ω_DE with α_B = −α_M is gradient-unstable for every non-zero amplitude,
and a running Planck mass leaves the waveform phase unchanged. LISA tests it
through the gravitational-wave luminosity distance.

## Repository layout

| Path | Contents |
|---|---|
| `analysis/setup_wsl.sh` | One-time setup on Linux/WSL: Miniforge, Cobaya, `hi_class` (built from source), Planck 2018 data |
| `analysis/cosmo/` | Cobaya theory wrapper for `hi_class` (`estg_theory.py`), config generator, profile likelihood (`profile_beta.py`, `run_profile.sh`), post-processing |
| `analysis/cosmo/results/` | Best fit at every β grid point, summary table, figure |
| `analysis/lisa/` | Standard-siren forecast with the Brans–Dicke M*(z) from `hi_class`; check of the waveform-phase degeneracy |
| `paper/` | Paper source, PDF and figures |

## Reproducing the results

On Linux or WSL2 (tested on Ubuntu under WSL2, 16 threads):

```bash
cd analysis
bash setup_wsl.sh            # Miniforge, Cobaya, hi_class, Planck data (~20-40 min)
cd cosmo
bash run_profile.sh          # profile likelihood in beta (~1 h on 2 cores)
python profile_beta.py --summary-only
cd ../lisa
python bd_siren_shape.py     # needs hiclassy (estg environment)
python standard_siren_forecast.py --shape-file bd_shape.txt
```

Versions used for the paper: `hi_class` 3.4.0.1 (commit in
[`analysis/HICLASS_COMMIT.txt`](analysis/HICLASS_COMMIT.txt)), Cobaya 3.6.2,
Python 3.11.

## Superseded code (kept for the record; do not use)

Versions of this repository up to commit `10b88bc` reported results that are
**not valid**: |α_M| < 0.027, R − 1 = 0.0003, χ² ≈ 2844, ln B = −0.93,
ΔAIC = 1.92 and a LISA sensitivity σ(α_M) ≈ 2.8 × 10⁻⁴. The "Planck" likelihood
in that code was a set of hand-written Gaussians, not the Planck data. The LISA
Fisher result came from a phase term that is an exact combination of the
chirp-mass and coalescence-phase derivatives; with the luminosity distance free,
the Fisher matrix is singular (see `analysis/lisa/phase_degeneracy_check.py`).

The following files belong to that version and are superseded by `analysis/`:

- `run_all_analyses.py`, `notebooks/`, `notebooks/mcmc_chain_merged.txt`
- `lisa/fisher_matrix.py`, `lisa/waveform_estg.py` (the LISA noise curve in
  `waveform_estg.py` is still correct and is used for Fig. 2 of the paper)
- `montepython/estg_likelihood.py` (non-functional stub), `config/base.param`,
  `Dockerfile`
- `paper/fig1_mcmc_corner.png`
- `paper/ESTG_Teorik_ve_Sayisal_Sorular_Yanitlar.*` (answers based on the old
  analysis)

## License

MIT, see [`LICENSE`](LICENSE).
