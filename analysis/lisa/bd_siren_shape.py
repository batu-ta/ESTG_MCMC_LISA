"""
GW-to-EM luminosity distance ratio for ESTG (Brans-Dicke, omega_BD = 1/(4 beta^2))
computed from hi_class's background, for the LISA standard-siren forecast.

    d_L^GW(z) / d_L^EM(z) = M_*(0) / M_*(z) = sqrt( M_*^2(0) / M_*^2(z) )

Writes analysis/lisa/bd_shape.txt with columns
    z   g(z)
where g(z) = [d_GW/d_EM(z) - 1] / alpha_M0 at small beta (the ratio is linear in
alpha_M0 ~ 6 beta^2 there), for use with
    python standard_siren_forecast.py --shape-file bd_shape.txt
and prints the ratio for beta at the Planck+BBN bound.

Needs hiclassy (run inside the WSL 'estg' environment).
"""

import os

import numpy as np
from hiclassy import HiClass

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = dict(A_s=2.1e-9, n_s=0.965, H0=67.36, omega_b=0.02237, omega_cdm=0.1200,
            tau_reio=0.0544, N_ur=2.0328, N_ncdm=1, m_ncdm=0.06, T_ncdm=0.71611)
BD = dict(Omega_Lambda=0, Omega_fld=0, Omega_smg=-1, gravity_model="brans_dicke",
          M2_tuning_smg="yes", M2_today_smg=1.0, pert_initial_conditions_smg="zero",
          a_min_stability_test_smg=1e-6, a_ini_over_a_today_default=1e-10,
          a_ini_test_qs_smg=1e-10, pert_ic_ini_z_ref_smg=1e9)


def ratio_curve(beta, z_out):
    c = HiClass()
    c.set(dict(BASE, **BD, parameters_smg=f"0.7, {1 / (4 * beta**2)!r}, 1.0, 0.0"))
    c.compute(["background"])
    bg = c.get_background()
    z = bg["z"][::-1]                      # increasing z
    m2 = bg["M*^2_smg"][::-1]
    alpha_m0 = float(bg["M2_running_smg"][-1])
    m2_z = np.interp(z_out, z, m2)
    m2_0 = np.interp(0.0, z, m2)
    c.struct_cleanup(); c.empty()
    return np.sqrt(m2_0 / m2_z), alpha_m0


def main():
    z_out = np.linspace(0.0, 8.0, 161)
    # shape at small coupling (linear regime)
    r, am = ratio_curve(0.005, z_out)
    g = (r - 1.0) / am
    path = os.path.join(HERE, "bd_shape.txt")
    np.savetxt(path, np.c_[z_out, g], header="z  g(z) = (d_GW/d_EM - 1)/alpha_M0  [Brans-Dicke, hi_class]")
    print(f"wrote {path}  (alpha_M0 = {am:.3e} at beta = 0.005)")
    # linearity check and the effect at the Planck+BBN bound
    for beta in (0.0244, 0.04):
        r_b, am_b = ratio_curve(beta, z_out)
        lin = 1.0 + am_b * g
        print(f"beta = {beta}: alpha_M0 = {am_b:.3e}")
        for zz in (0.5, 1.0, 2.0, 4.0):
            i = int(np.argmin(np.abs(z_out - zz)))
            print(f"   z = {zz:3.1f}: d_GW/d_EM - 1 = {r_b[i] - 1:+.2e}   (linear model: {lin[i] - 1:+.2e})")


if __name__ == "__main__":
    main()
