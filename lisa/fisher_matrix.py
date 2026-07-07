import numpy as np

def fisher_alpha_M(f, h_gr, alpha_M, Mc, Sn, f_min=1e-5):
    """
    Fisher matrisi yaklaşımı ile α_M için hassasiyeti hesaplar.
    """
    G = 6.67430e-11
    c = 299792458.0
    prefactor = (5.0 / 192.0) * (G * Mc / c**3)**(-5.0/3.0)
    dpsi_dalpha = prefactor * (f_min**(-5.0/3.0) - f**(-5.0/3.0))
    F = 4 * np.trapz((dpsi_dalpha * np.abs(h_gr))**2 / Sn, f)
    return np.sqrt(1.0 / F)
