"""
LISA standard-siren forecast for the running Planck mass alpha_M.

Physics
-------
A running Planck mass changes the friction term of tensor modes,
h'' + (2 + alpha_M) H h' + k^2 h = 0, so the GW amplitude decays as 1/(M_* a)
instead of 1/a. The waveform *phase* is unchanged; what changes is the
luminosity distance inferred from the amplitude (Belgacem et al. 2018,
Lagos et al. 2019):

    d_L^GW(z) / d_L^EM(z) = M_*(0) / M_*(z) = exp( 1/2 * int_0^z alpha_M(z') / (1+z') dz' )

With alpha_M(a) = alpha_M0 * Omega_DE(a) / Omega_DE0 (paper Eq. 15) on a flat
LCDM background, Omega_DE(a)/Omega_DE0 = 1 / E(z)^2.

The test therefore needs sirens with an electromagnetic redshift (massive
black-hole binaries with a counterpart). For each siren we "measure" d_L^GW
with a fractional error made of
  * instrumental error (input, default 5%),
  * weak lensing,  sigma_lens/d_L = 0.066 [(1-(1+z)^-0.25)/0.25]^1.8
    (Tamanini et al. 2016), optionally reduced by delensing,
  * peculiar velocity, sigma_v/d_L = [1 + c(1+z)^2/(H(z) d_L)] * v/c, v = 500 km/s
    (Kocsis et al. 2006).
H0 and Omega_m are free with Gaussian priors (default: Planck 2018 widths) and
are marginalised over in the Fisher matrix.

Usage:
    python standard_siren_forecast.py                  # default scenarios
    python standard_siren_forecast.py --sigma-inst 0.02 --delensing 0.5
"""

import argparse

import numpy as np

C_KMS = 299792.458


def E(z, om):
    return np.sqrt(om * (1 + z) ** 3 + 1 - om)


def comoving_distance(z, h0, om, n=2000):
    zz = np.linspace(0.0, z, n)
    return C_KMS / h0 * np.trapezoid(1.0 / E(zz, om), zz)


def dl_em(z, h0, om):
    return (1 + z) * comoving_distance(z, h0, om)


def friction_integral(z, om, n=2000):
    """I(z) = int_0^z [Omega_DE(z')/Omega_DE0] / (1+z') dz'."""
    zz = np.linspace(0.0, z, n)
    return np.trapezoid(1.0 / (E(zz, om) ** 2 * (1 + zz)), zz)


# Optional tabulated shape g(z) = (d_GW/d_EM - 1)/alpha_M0, e.g. Brans-Dicke from
# hi_class (bd_siren_shape.py). None -> alpha_M proportional to Omega_DE.
SHAPE = None


def dl_gw(z, h0, om, alpha_m0):
    if SHAPE is not None:
        return dl_em(z, h0, om) * (1.0 + alpha_m0 * SHAPE(z))
    return dl_em(z, h0, om) * np.exp(0.5 * alpha_m0 * friction_integral(z, om))


def sigma_frac(z, h0, om, sigma_inst, delensing, v_pec=500.0):
    lens = 0.066 * ((1 - (1 + z) ** -0.25) / 0.25) ** 1.8 * (1 - delensing)
    d = dl_em(z, h0, om)
    hz = h0 * E(z, om)
    pv = (1 + C_KMS * (1 + z) ** 2 / (hz * d)) * v_pec / C_KMS
    return np.sqrt(sigma_inst**2 + lens**2 + pv**2)


def fisher_sigma_alpha(zs, sigma_inst, delensing, h0=67.4, om=0.315,
                       sig_h0=0.5, sig_om=0.0073, alpha_m0=0.0):
    """Marginalised 1-sigma error on alpha_M0 from sirens at redshifts zs."""
    theta0 = np.array([alpha_m0, h0, om])
    steps = np.array([1e-3, 1e-3 * h0, 1e-3 * om])
    F = np.zeros((3, 3))
    for z in zs:
        d0 = dl_gw(z, *theta0[1:], theta0[0])
        sig = sigma_frac(z, h0, om, sigma_inst, delensing) * d0
        grad = np.zeros(3)
        for i in range(3):
            tp, tm = theta0.copy(), theta0.copy()
            tp[i] += steps[i]
            tm[i] -= steps[i]
            grad[i] = (dl_gw(z, tp[1], tp[2], tp[0]) - dl_gw(z, tm[1], tm[2], tm[0])) / (2 * steps[i])
        F += np.outer(grad, grad) / sig**2
    # Gaussian priors on H0 and Omega_m (e.g. from the CMB analysis)
    F[1, 1] += 1 / sig_h0**2
    F[2, 2] += 1 / sig_om**2
    cov = np.linalg.inv(F)
    return np.sqrt(cov[0, 0]), 1 / np.sqrt(F[0, 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma-inst", type=float, default=0.05,
                    help="fractional instrumental d_L error per siren")
    ap.add_argument("--delensing", type=float, default=0.0,
                    help="fraction of lensing scatter removed (0..1)")
    ap.add_argument("--zmin", type=float, default=0.5)
    ap.add_argument("--zmax", type=float, default=4.0)
    ap.add_argument("--shape-file", help="table 'z g(z)' from bd_siren_shape.py")
    args = ap.parse_args()

    global SHAPE
    if args.shape_file:
        tab = np.loadtxt(args.shape_file)
        SHAPE = lambda z: np.interp(z, tab[:, 0], tab[:, 1])  # noqa: E731
        print(f"alpha_M(z) shape from {args.shape_file}")
    else:
        print("alpha_M(z) = alpha_M0 * Omega_DE(z)/Omega_DE0")
    print(f"sigma_inst = {args.sigma_inst:.0%}, delensing = {args.delensing:.0%}")
    print("H0, Omega_m marginalised with Planck-like priors (0.5 km/s/Mpc, 0.0073)\n")

    z1 = 1.0
    print(f"d_GW/d_EM - 1 at z={z1} for alpha_M0 = 0.01: "
          f"{dl_gw(z1, 67.4, 0.315, 0.01) / dl_em(z1, 67.4, 0.315) - 1:.2e}")
    s_marg, s_cond = fisher_sigma_alpha([z1], args.sigma_inst, args.delensing)
    print(f"single siren at z=1 (the paper's case): sigma(alpha_M0) = {s_marg:.3f} "
          f"(H0, Om fixed: {s_cond:.3f})\n")

    print(f"{'N sirens':>9} {'sigma(alpha_M0)':>16}   (z uniform in [{args.zmin}, {args.zmax}])")
    for n in (1, 3, 10, 30, 100):
        zs = np.linspace(args.zmin, args.zmax, n) if n > 1 else [0.5 * (args.zmin + args.zmax)]
        s_marg, _ = fisher_sigma_alpha(zs, args.sigma_inst, args.delensing)
        print(f"{n:>9} {s_marg:>16.4f}")
    print("\nLISA MBHB sirens with an EM counterpart: a few to a few tens over 4 years,")
    print("depending on the population model (Tamanini et al. 2016). Choose N accordingly.")


if __name__ == "__main__":
    main()
