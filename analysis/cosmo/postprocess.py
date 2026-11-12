"""
Summarise the ESTG and GR Cobaya runs.

Usage (from analysis/cosmo, after the MCMC and minimize runs have finished):
    python postprocess.py --chains-dir chains

Reports, all computed from the chain and minimum files (no numbers typed in):
  * Gelman-Rubin R-1 from the last line of the .progress file
  * beta, alpha_M0, alpha_B0: mean, std, 68%/95% one-sided upper limits;
    the matching lower limit on omega_BD = 1/(4 beta^2)
  * Savage-Dickey Bayes factor B01 = p(beta=0|d)/pi(beta=0), with its sign spelled out
  * chi^2 (Planck likelihoods + BBN), Delta chi^2 and Delta AIC (Delta k = 1)
"""

import argparse
import glob
import os

import numpy as np
from getdist import loadMCSamples

BETA_PRIOR = (0.0, 0.1)  # must match make_configs.py


def weighted_quantile(x, w, q):
    order = np.argsort(x)
    cdf = np.cumsum(w[order]) / np.sum(w)
    return x[order][np.searchsorted(cdf, q)]


def read_last_rminus1(prefix):
    progress = prefix + ".progress"
    if not os.path.exists(progress):
        return None
    with open(progress, encoding="utf-8") as fh:
        header = None
        last = None
        for line in fh:
            if line.startswith("#"):
                header = line.lstrip("#").split()
            elif line.strip():
                last = line.split()
    if header is None or last is None or "Rminus1" not in header:
        return None
    return float(last[header.index("Rminus1")])


def read_minimum(prefix):
    """Return dict column -> value from Cobaya's '<prefix>.minimum.txt' file
    (getdist format: '#'-header with column names, then one row).
    The '<prefix>.minimum' file is a human-readable table and is not used."""
    path = prefix + ".minimum.txt"
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        lines = [ln for ln in fh if ln.strip()]
    header = [ln for ln in lines if ln.startswith("#")][0].lstrip("#").split()
    row = [ln for ln in lines if not ln.startswith("#")][0].split()
    return dict(zip(header, map(float, row)))


def chi2_planck_bbn(minimum):
    # 'chi2' = -2 ln L summed over likelihoods (Planck). BBN is an external
    # prior, so add it back: -2 ln pi_bbn = 2 * minuslogprior__bbn.
    return minimum["chi2"] + 2.0 * minimum.get("minuslogprior__bbn", 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains-dir", default="chains")
    args = ap.parse_args()

    estg_prefix = os.path.join(args.chains_dir, "estg_mcmc", "estg")
    if not glob.glob(estg_prefix + ".*.txt"):
        raise SystemExit(f"No ESTG chains found at {estg_prefix}.*.txt")

    print("=== ESTG MCMC ===")
    rm1 = read_last_rminus1(estg_prefix)
    print(f"Gelman-Rubin R-1 (last check): {rm1}")

    samples = loadMCSamples(estg_prefix, settings={"ignore_rows": 0.3})
    n_eff = samples.getEffectiveSamples(samples.index["beta"])
    print(f"Samples after 30% burn-in: {samples.numrows}, effective (beta): {n_eff:.0f}")

    w = samples.weights
    # alpha_B0 = -alpha_M0 exactly in Brans-Dicke, so it adds nothing here.
    for name in ("beta", "alpha_M0"):
        x = samples[name]
        print(f"{name}: mean {samples.mean(name):.5f}, std {samples.std(name):.5f}, "
              f"68% upper {weighted_quantile(x, w, 0.68):.5f}, "
              f"95% upper {weighted_quantile(x, w, 0.95):.5f}")
    b95 = weighted_quantile(samples["beta"], w, 0.95)
    print(f"=> beta < {b95:.4f} (95%)  <=>  omega_BD > {1 / (4 * b95**2):.0f}")

    # Savage-Dickey at beta = 0 (GR), flat prior on beta. beta = 0 is the prior
    # edge; getdist applies its boundary correction using the ranges Cobaya saved.
    dens = samples.get1DDensity("beta")
    dens.normalize("integral", in_place=True)
    post0 = float(dens.Prob(0.0))
    prior0 = 1.0 / (BETA_PRIOR[1] - BETA_PRIOR[0])
    b01 = post0 / prior0
    print(f"Savage-Dickey: p(beta=0|d) = {post0:.2f}, pi(beta=0) = {prior0:.2f}")
    print(f"  B01 = {b01:.3f}, ln B01 = {np.log(b01):+.2f} (>0 favours GR, <0 favours ESTG)")
    print(f"  NOTE: B01 scales with the prior width on beta ({BETA_PRIOR}); quote it.")

    print("\n=== Best fits (Planck + BBN) ===")
    m_estg = read_minimum(os.path.join(args.chains_dir, "estg_minimize", "estg"))
    m_gr = read_minimum(os.path.join(args.chains_dir, "gr_minimize", "gr"))
    if m_estg is None or m_gr is None:
        print("Minimum files missing: run estg_minimize.yaml and gr_minimize.yaml.")
        return
    c_estg = chi2_planck_bbn(m_estg)
    c_gr = chi2_planck_bbn(m_gr)
    d_chi2 = c_estg - c_gr
    print(f"chi2 GR   = {c_gr:.2f}")
    print(f"chi2 ESTG = {c_estg:.2f}  (best-fit beta = {m_estg.get('beta', float('nan')):.4f})")
    print(f"Delta chi2 = {d_chi2:+.2f}, Delta AIC = Delta chi2 + 2 = {d_chi2 + 2:+.2f}")
    for k in sorted(m_estg):
        if k.startswith("chi2__"):
            print(f"  {k}: ESTG {m_estg[k]:.2f}  GR {m_gr.get(k, float('nan')):.2f}")


if __name__ == "__main__":
    main()
