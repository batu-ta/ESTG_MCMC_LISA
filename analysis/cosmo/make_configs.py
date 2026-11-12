"""
Generate the Cobaya input files for the ESTG and GR (LCDM) runs.

Usage (inside WSL, from analysis/cosmo):
    python make_configs.py --chains-dir ~/estg/chains

Writes configs/{estg,gr}_{test,mcmc,minimize}.yaml. ESTG is run as hi_class's
covariant Brans-Dicke model with omega_BD = 1/(4 beta^2) (see estg_theory.py);
GR is LCDM through the same hi_class build, so both models share code,
precision settings and likelihoods and their chi^2 values compare directly.
"""

import argparse
import copy
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))

# Primordial deuterium prior on omega_b h^2.
# This is the value quoted in the paper (Cooke et al. 2018). Check it against
# the paper's table before publishing: the result depends on the d(p,gamma)3He
# rate used.
BBN_OMBH2_MEAN = 0.0222
BBN_OMBH2_SIGMA = 0.0005

LIKELIHOODS = {
    # Native-Python Planck 2018 likelihoods: no clik/Fortran build needed.
    "planck_2018_lowl.TT": None,
    "planck_2018_lowl.EE": None,
    "planck_2018_highl_plik.TTTEEE_lite_native": None,
    "planck_2018_lensing.native": None,
}

# BBN enters as a Gaussian external prior on the sampled omega_b. (As an external
# *likelihood* it would claim omega_b for itself and classy, which receives only
# unclaimed parameters, would never see it.) postprocess.py adds
# 2*minuslogprior__bbn back into chi^2 so it counts as data in Delta chi^2/AIC.
EXTERNAL_PRIORS = {
    "bbn": (
        f"lambda omega_b: -0.5*((omega_b-{BBN_OMBH2_MEAN})/{BBN_OMBH2_SIGMA})**2"
    ),
}

LCDM_PARAMS = {
    "logA": {
        "prior": {"min": 1.61, "max": 3.91},
        "ref": {"dist": "norm", "loc": 3.05, "scale": 0.001},
        "proposal": 0.001,
        "latex": r"\log(10^{10} A_\mathrm{s})",
        "drop": True,
    },
    "A_s": {"value": "lambda logA: 1e-10*np.exp(logA)", "latex": r"A_\mathrm{s}"},
    "n_s": {
        "prior": {"min": 0.8, "max": 1.2},
        "ref": {"dist": "norm", "loc": 0.965, "scale": 0.004},
        "proposal": 0.002,
        "latex": r"n_\mathrm{s}",
    },
    "H0": {
        "prior": {"min": 40, "max": 100},
        "ref": {"dist": "norm", "loc": 67.4, "scale": 0.5},
        "proposal": 0.5,
        "latex": "H_0",
    },
    "omega_b": {
        "prior": {"min": 0.005, "max": 0.1},
        "ref": {"dist": "norm", "loc": 0.0224, "scale": 0.0001},
        "proposal": 0.0001,
        "latex": r"\Omega_\mathrm{b} h^2",
    },
    "omega_cdm": {
        "prior": {"min": 0.001, "max": 0.99},
        "ref": {"dist": "norm", "loc": 0.120, "scale": 0.001},
        "proposal": 0.0005,
        "latex": r"\Omega_\mathrm{c} h^2",
    },
    "tau_reio": {
        "prior": {"min": 0.01, "max": 0.8},
        "ref": {"dist": "norm", "loc": 0.055, "scale": 0.006},
        "proposal": 0.003,
        "latex": r"\tau_\mathrm{reio}",
    },
    "Omega_m": {"latex": r"\Omega_\mathrm{m}"},
    "sigma8": {"latex": r"\sigma_8"},
}

# ESTG coupling. omega_BD = 1/(4 beta^2); the sign of beta does not enter.
# Upper edge 0.1 is above the paper's dipole-radiation bound (0.075).
BETA_PRIOR = (0.0, 0.1)
BETA_SAMPLED = {
    "prior": {"min": BETA_PRIOR[0], "max": BETA_PRIOR[1]},
    "ref": {"dist": "uniform", "min": 0.002, "max": 0.02},
    "proposal": 0.003,
    "latex": r"\beta",
}

ESTG_DERIVED = {
    "alpha_M0": {"latex": r"\alpha_{M,0}"},
    "alpha_B0": {"latex": r"\alpha_{B,0}"},
    "omega_BD": {"latex": r"\omega_\mathrm{BD}"},
}


def theory_block(model):
    # hi_class itself is imported as the installed 'hiclassy' package.
    return {
        "estg_theory.HiClassESTG": {
            "python_path": HERE,
            "model": "brans_dicke" if model == "estg" else "gr",
            "extra_args": {
                # Planck 2018 baseline neutrino treatment
                "N_ur": 2.0328,
                "N_ncdm": 1,
                "m_ncdm": 0.06,
                "T_ncdm": 0.71611,
            },
        }
    }


def build(model, mode, out_root):
    params = copy.deepcopy(LCDM_PARAMS)
    if model == "estg":
        params["beta"] = copy.deepcopy(BETA_SAMPLED)
        params.update(copy.deepcopy(ESTG_DERIVED))

    info = {
        "theory": theory_block(model),
        "likelihood": copy.deepcopy(LIKELIHOODS),
        "prior": copy.deepcopy(EXTERNAL_PRIORS),
        "params": params,
        "output": os.path.join(out_root, f"{model}_{mode}", model),
    }
    if mode == "test":
        info["sampler"] = {"evaluate": None}
        info["debug"] = False
    elif mode == "mcmc":
        info["sampler"] = {
            "mcmc": {
                "covmat": "auto",
                "Rminus1_stop": 0.01,
                "Rminus1_cl_stop": 0.2,
                "max_tries": 10000,
                "measure_speeds": True,
            }
        }
    elif mode == "minimize":
        # Maximise likelihood x BBN prior. All other priors are flat, so this is
        # the maximum-likelihood point of Planck+BBN.
        info["sampler"] = {"minimize": {"ignore_prior": False, "best_of": 4}}
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains-dir", default=os.path.join(HERE, "chains"))
    args = ap.parse_args()

    cfg_dir = os.path.join(HERE, "configs")
    os.makedirs(cfg_dir, exist_ok=True)
    for model in ("estg", "gr"):
        for mode in ("test", "mcmc", "minimize"):
            info = build(model, mode, args.chains_dir)
            path = os.path.join(cfg_dir, f"{model}_{mode}.yaml")
            with open(path, "w", encoding="utf-8") as fh:
                yaml.safe_dump(info, fh, sort_keys=False)
            print("wrote", path)


if __name__ == "__main__":
    main()
