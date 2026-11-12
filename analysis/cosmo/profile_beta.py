"""
Profile likelihood of Planck 2018 + BBN in the ESTG coupling beta.

For each beta on a grid, beta is held fixed and all other parameters
(LCDM + A_planck) are optimised with Cobaya's minimizer. The profile
Delta chi^2(beta) then gives the upper limit on beta without an MCMC:
  * beta >= 0 with the best fit at/near the boundary -> one-sided 95% limit at
    Delta chi^2 = 2.71; Delta chi^2 = 3.84 is also reported.
  * beta = 0 is GR (checked: chi^2 differs from the LCDM run by 0.016), so
    Delta AIC = chi^2_min - chi^2(beta=0) + 2.

chi^2 here is -2 ln(L x pi_BBN x pi_Aplanck) up to a constant that is the same
for every grid point (all other priors are flat and beta is fixed).

Each point is written to <out>/beta_<value>/; finished points are skipped on a
re-run, so the script can be stopped (Ctrl+C) and started again at any time.
Each point starts from the previous point's best fit.

Usage (inside WSL, from analysis/cosmo, after make_configs.py):
    OMP_NUM_THREADS=2 nice -n 19 python profile_beta.py --out ~/estg/chains/profile
    python profile_beta.py --out ~/estg/chains/profile --only 0.0     # one point
    python profile_beta.py --out ~/estg/chains/profile --summary-only
"""

import argparse
import copy
import json
import os
import time

import numpy as np
import yaml

from postprocess import read_minimum

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GRID = [0.0, 0.004, 0.008, 0.012, 0.016, 0.020, 0.025, 0.030, 0.040]
SAMPLED = ["logA", "n_s", "H0", "omega_b", "omega_cdm", "tau_reio", "A_planck"]
# Planck 2018 TT,TE,EE+lowE+lensing best fit: starting point for the first beta
PLANCK_START = dict(logA=3.044, n_s=0.9649, H0=67.36, omega_b=0.02237,
                    omega_cdm=0.1200, tau_reio=0.0544, A_planck=1.0)


def point_dir(out, beta):
    return os.path.join(out, f"beta_{beta:.4f}")


def point_info(beta, start, out):
    with open(os.path.join(HERE, "configs", "estg_minimize.yaml"), encoding="utf-8") as fh:
        info = yaml.safe_load(fh)
    params = info["params"]
    params["beta"] = float(beta)
    # A_planck: same Gaussian prior as plik_lite defines, so we can set its start.
    params["A_planck"] = {"prior": {"dist": "norm", "loc": 1.0, "scale": 0.0025},
                          "ref": 1.0, "proposal": 0.0005, "latex": r"y_\mathrm{cal}"}
    for p in SAMPLED:
        params[p]["ref"] = float(start[p])
    info["sampler"] = {"minimize": {"ignore_prior": False, "best_of": 1}}
    info["output"] = os.path.join(point_dir(out, beta), "estg")
    return info


def load_point(out, beta):
    m = read_minimum(os.path.join(point_dir(out, beta), "estg"))
    if m is None:
        return None
    m["chi2_eff"] = 2.0 * m["minuslogpost"]
    return m


def run_point(beta, start, out):
    from cobaya.run import run  # imported here so --summary-only works without Cobaya
    t0 = time.time()
    run(point_info(beta, start, out), force=True)
    m = load_point(out, beta)
    m["wall_seconds"] = time.time() - t0
    with open(os.path.join(point_dir(out, beta), "timing.json"), "w", encoding="utf-8") as fh:
        json.dump({"beta": beta, "wall_seconds": m["wall_seconds"]}, fh)
    return m


def crossing(betas, d, level):
    """First beta above the minimum where Delta chi^2 crosses `level` (linear interp)."""
    i0 = int(np.argmin(d))
    for i in range(i0, len(betas) - 1):
        if d[i] < level <= d[i + 1]:
            return betas[i] + (level - d[i]) * (betas[i + 1] - betas[i]) / (d[i + 1] - d[i])
    return None


def summarise(out, grid):
    rows = [(b, load_point(out, b)) for b in grid]
    rows = [(b, m) for b, m in rows if m is not None]
    if not rows:
        print("No finished points yet.")
        return
    betas = np.array([b for b, _ in rows])
    chi2 = np.array([m["chi2_eff"] for _, m in rows])
    dchi2 = chi2 - chi2.min()
    print(f"\n{'beta':>7} {'omega_BD':>10} {'alpha_M0':>10} {'H0':>7} {'chi2_eff':>10} {'dchi2':>7} {'time':>7}")
    for (b, m), d in zip(rows, dchi2):
        tfile = os.path.join(point_dir(out, b), "timing.json")
        t = json.load(open(tfile))["wall_seconds"] / 60 if os.path.exists(tfile) else float("nan")
        print(f"{b:7.4f} {m.get('omega_BD', float('nan')):10.0f} {m.get('alpha_M0', float('nan')):10.2e} "
              f"{m['H0']:7.2f} {m['chi2_eff']:10.3f} {d:7.3f} {t:6.1f}m")
    if 0.0 in betas:
        c0 = chi2[list(betas).index(0.0)]
        print(f"\nchi2(best) - chi2(beta=0, GR) = {chi2.min() - c0:+.3f}  ->  "
              f"Delta AIC = {chi2.min() - c0 + 2:+.3f}")
    for level, label in ((2.71, "95% one-sided"), (3.84, "Delta chi2 = 3.84")):
        b = crossing(betas, dchi2, level)
        if b is None:
            print(f"{label}: not reached on the finished grid points")
        else:
            print(f"{label}: beta < {b:.4f}  <=>  omega_BD > {1 / (4 * b * b):.0f}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        h0 = np.array([m["H0"] for _, m in rows])
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8, 3.3), dpi=300)
        ax.plot(betas, dchi2, "o-", color="#1f4e79")
        for level, lab in ((2.71, "95% one-sided"), (3.84, r"$\Delta\chi^2=3.84$")):
            ax.axhline(level, ls=":", color="gray", lw=1)
            ax.text(betas[0], level + 0.2, lab, fontsize=7, color="gray")
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(r"$\Delta\chi^2$ (profile)")
        ax.set_title("Planck 2018 + BBN", fontsize=10)
        ax2.plot(betas, h0, "s-", color="#8c2d04")
        ax2.set_xlabel(r"$\beta$")
        ax2.set_ylabel(r"best-fit $H_0$ [km s$^{-1}$ Mpc$^{-1}$]")
        ax2.set_title(r"$H_0$ along the profile", fontsize=10)
        fig.tight_layout()
        path = os.path.join(out, "profile_beta.png")
        fig.savefig(path)
        print(f"plot: {path}")
    except Exception as excpt:  # plotting is optional
        print(f"(plot skipped: {excpt})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/estg/chains/profile"))
    ap.add_argument("--grid", type=float, nargs="+", default=DEFAULT_GRID)
    ap.add_argument("--only", type=float, help="run just this beta value")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    grid = sorted(set(args.grid + ([args.only] if args.only is not None else [])))

    if not args.summary_only:
        todo = [args.only] if args.only is not None else grid
        start = dict(PLANCK_START)
        for beta in grid:
            done = load_point(args.out, beta)
            if done is not None:
                print(f"beta={beta:.4f}: already done (chi2_eff={done['chi2_eff']:.3f}), skipping")
                start = {p: done[p] for p in SAMPLED}
                continue
            if beta not in todo:
                continue
            print(f"\n=== beta = {beta:.4f} ===", flush=True)
            m = run_point(beta, start, args.out)
            print(f"beta={beta:.4f}: chi2_eff={m['chi2_eff']:.3f} in {m['wall_seconds'] / 60:.1f} min",
                  flush=True)
            start = {p: m[p] for p in SAMPLED}
    summarise(args.out, grid)


if __name__ == "__main__":
    main()
