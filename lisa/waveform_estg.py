"""
LISA için ESTG (Entropik Skaler-Tensör Kütleçekimi) dalga formu ve gürültü modeli.
Referans: Robson, Cornish & Liu (2019), Class. Quantum Grav. 36 105011.
"""

import numpy as np

# Temel Fiziksel Sabitler (SI)
G = 6.67430e-11        # m^3 kg^-1 s^-2
c = 299792458.0        # m s^-1
M_SUN = 1.98847e30     # kg
MPC = 3.085677581e22   # m


def lisa_psd(f, include_confusion=True):
    """
    Robson, Cornish & Liu (2019) analitik gökyüzü-ortalamalı LISA gürültü spektral yoğunluğu Sn(f).
    
    Parametreler:
    - f: Frekans dizisi (Hz)
    - include_confusion: 4 yıllık galaktik çift beyaz cüce karmaşa gürültüsü dahil edilsin mi?
    
    Dönüş:
    - S_n: LISA gerinim gürültü PSD'si (1/Hz)
    """
    L = 2.5e9  # LISA kol uzunluğu (metre)
    f_star = c / (2.0 * np.pi * L)  # ~19.09 mHz transfer frekansı

    # Optik Metroloji Sistemi (OMS) pozisyon gürültüsü
    P_oms = (1.5e-11)**2 * (1.0 + (2.0e-3 / f)**4)

    # Test Kütlesi ivme gürültüsü
    P_acc = (3.0e-15)**2 * (1.0 + (0.4e-3 / f)**2) * (1.0 + (f / 8.0e-3)**4)

    # Enstrümantal gerinim gürültüsü
    S_n = (10.0 / (3.0 * L**2)) * (P_oms + 4.0 * P_acc / (2.0 * np.pi * f)**4) * (1.0 + 0.6 * (f / f_star)**2)

    if include_confusion:
        # 4 yıllık gözlem için analitik galaktik karmaşa gürültüsü uyumu
        A = 9.0e-45
        alpha = 0.138
        beta = -221.0
        kappa = 521.0
        gamma = 1680.0
        f_k = 0.00113
        S_c = A * f**(-7.0 / 3.0) * np.exp(-f**alpha + beta * f * np.sin(kappa * f)) * (1.0 + np.tanh(gamma * (f_k - f)))
        S_n += S_c

    return S_n


def gr_taylorf2_amplitude(f, Mc, dL):
    """
    Genel Görelilik için 0PN TaylorF2 dalga formu gökyüzü-ortalamalı genliği A(f).
    
    Parametreler:
    - f: Frekans dizisi (Hz)
    - Mc: Chirp kütlesi (kg)
    - dL: Lüminosite mesafesi (m)
    """
    prefactor = np.sqrt(5.0 / 24.0) / (np.pi**(2.0 / 3.0)) * (c / dL)
    u = (G * Mc / c**3)**(5.0 / 6.0)
    return prefactor * u * f**(-7.0 / 6.0)


def estg_phase(f, alpha_M, Mc, f_min=1e-5):
    """
    ESTG teorisinden kaynaklanan ek faz sapması Delta_Psi(f).
    
    Parametreler:
    - f: Frekans dizisi (Hz)
    - alpha_M: ESTG parametresi
    - Mc: Chirp kütlesi (kg)
    - f_min: LISA alt kesim frekansı (Hz)
    """
    factor = (5.0 / 192.0) * (G * Mc / c**3)**(-5.0 / 3.0)
    integral = (f_min**(-5.0 / 3.0) - f**(-5.0 / 3.0))
    return alpha_M * factor * integral


def snr_estg(f, h_gr_or_Mc, alpha_M_or_dL=None, Mc=None, Sn=None, f_min=1e-5):
    """
    ESTG dalga formu için SNR hesaplar. Hem eski hem yeni parametre formatını destekler.
    """
    # Eski format: snr_estg(f, h_gr, alpha_M, Mc, Sn, f_min)
    if Sn is not None:
        h_gr = h_gr_or_Mc
        alpha_M = alpha_M_or_dL
        delta_psi = estg_phase(f, alpha_M, Mc, f_min)
        h_estg = h_gr * np.exp(1j * delta_psi)
        try:
            snr2 = 4.0 * np.trapezoid(np.abs(h_estg)**2 / Sn, f)
        except AttributeError:
            snr2 = 4.0 * np.trapz(np.abs(h_estg)**2 / Sn, f)
        return np.sqrt(snr2)
    else:
        # Yeni format: snr_estg(f, Mc, dL)
        Mc_val = h_gr_or_Mc
        dL_val = alpha_M_or_dL
        A = gr_taylorf2_amplitude(f, Mc_val, dL_val)
        Sn_val = lisa_psd(f)
        try:
            snr2 = 4.0 * np.trapezoid(A**2 / Sn_val, f)
        except AttributeError:
            snr2 = 4.0 * np.trapz(A**2 / Sn_val, f)
        return np.sqrt(snr2)
