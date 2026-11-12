"""
Why the old lisa/fisher_matrix.py result (sigma(alpha_M) ~ 2.8e-4) is not a
measurement.

1. The phase term used there,
       dPsi/dalpha_M = (5/192) (G Mc/c^3)^(-5/3) (f_min^(-5/3) - f^(-5/3)),
   is an exact linear combination of dPsi/dphi_c (a constant) and
   dPsi/dln(Mc) (proportional to f^(-5/3)). The phase carries no independent
   information on alpha_M.
2. The only thing that breaks the degeneracy is the Mc^(5/6) scaling of the
   amplitude. Then sigma(alpha_M) = sigma_amp(ln Mc)/k, with
   sigma_amp(ln Mc) = 1/((5/6) SNR) and k = (128/192) pi^(5/3).
3. The amplitude also scales as 1/d_L. Adding ln d_L as a free parameter (it is
   always unknown) makes the Fisher matrix singular to machine precision.

Run:  python phase_degeneracy_check.py
"""

import numpy as np

G = 6.67430e-11
C = 299792458.0
MSUN = 1.98847e30
MPC = 3.085677581e22


def psd(f):
    L = 2.5e9
    fs = C / (2 * np.pi * L)
    p_oms = (1.5e-11) ** 2 * (1 + (2e-3 / f) ** 4)
    p_acc = (3e-15) ** 2 * (1 + (0.4e-3 / f) ** 2) * (1 + (f / 8e-3) ** 4)
    s = (10 / (3 * L**2)) * (p_oms + 4 * p_acc / (2 * np.pi * f) ** 4) * (1 + 0.6 * (f / fs) ** 2)
    s += 9e-45 * f ** (-7 / 3) * np.exp(-f**0.138 - 221 * f * np.sin(521 * f)) * (
        1 + np.tanh(1680 * (0.00113 - f)))
    return s


def fisher(m1, m2, z=1.0, f_min=1e-5, with_dl=False, dl_mpc=6700.0):
    M1, M2 = m1 * MSUN * (1 + z), m2 * MSUN * (1 + z)
    mt = M1 + M2
    mc = (M1 * M2) ** 0.6 / mt**0.2
    f_isco = C**3 / (6**1.5 * np.pi * G * mt)
    f = np.logspace(np.log10(f_min), np.log10(f_isco), 20000)
    amp = np.sqrt(5 / 24) / np.pi ** (2 / 3) * (C / (dl_mpc * MPC)) * (G * mc / C**3) ** (5 / 6) * f ** (-7 / 6)
    w = amp**2 / psd(f)
    u = np.pi * G * mc * f / C**3
    # phase and log-amplitude derivatives for (ln Mc, t_c, phi_c, alpha_M[, ln d_L])
    dpsi = [-(5 / 128) * u ** (-5 / 3), 2 * np.pi * f, -np.ones_like(f),
            (5 / 192) * (G * mc / C**3) ** (-5 / 3) * (f_min ** (-5 / 3) - f ** (-5 / 3))]
    dlna = [5 / 6 * np.ones_like(f), 0 * f, 0 * f, 0 * f]
    if with_dl:
        dpsi.append(0 * f)
        dlna.append(-np.ones_like(f))
    n = len(dpsi)
    F = np.array([[np.trapezoid(4 * (dpsi[i] * dpsi[j] + dlna[i] * dlna[j]) * w, f)
                   for j in range(n)] for i in range(n)])
    snr = np.sqrt(np.trapezoid(4 * w, f))
    return F, snr, f, mc


def main():
    # 1. exact linear dependence of the alpha_M phase derivative
    _, _, f, mc = fisher(1e6, 1e6)
    u = np.pi * G * mc * f / C**3
    X = np.vstack([-(5 / 128) * u ** (-5 / 3), -np.ones_like(f)]).T
    y = (5 / 192) * (G * mc / C**3) ** (-5 / 3) * (1e-5 ** (-5 / 3) - f ** (-5 / 3))
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    print("1) dPsi/dalpha_M as a combination of dPsi/dlnMc and dPsi/dphi_c:")
    print(f"   max relative residual = {np.max(np.abs(X @ coef - y)) / np.max(np.abs(y)):.1e}\n")

    k = (128 / 192) * np.pi ** (5 / 3)
    print("2) old setup vs. amplitude-only prediction 1/((5/6) SNR k):")
    for m in [(1e6, 1e5), (1e6, 1e6), (1e7, 1e7)]:
        F, snr, *_ = fisher(*m)
        s = np.sqrt(np.linalg.inv(F)[3, 3])
        print(f"   {m[0]:.0e}+{m[1]:.0e}: SNR={snr:6.1f}  sigma={s:.3e}  "
              f"prediction={1 / (5 / 6 * snr * k):.3e}")

    print("\n3) add ln d_L as a free parameter:")
    for m in [(1e6, 1e5), (1e6, 1e6), (1e7, 1e7)]:
        F, _, _, _ = fisher(*m, with_dl=True)
        d = np.sqrt(np.diag(F))
        ev = np.linalg.eigvalsh(F / np.outer(d, d))
        print(f"   {m[0]:.0e}+{m[1]:.0e}: smallest normalised eigenvalue = {ev.min():.1e} "
              f"(machine epsilon ~ 2e-16) -> singular, sigma(alpha_M) undefined")


if __name__ == "__main__":
    main()
