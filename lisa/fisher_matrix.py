"""
LISA verileri üzerinde ESTG parametreleri için çok parametreli (Marginalized)
Fisher Bilgi Matrisi (FIM) analizi.
"""

import numpy as np
from lisa.waveform_estg import lisa_psd, gr_taylorf2_amplitude, G, c


def fisher_analysis_estg(f_min=1e-5, M1_msun=1e6, M2_msun=1e6, z=1.0, dL_mpc=6700.0, alpha_M_fid=0.0):
    """
    Parametre vektörü theta = [ln(Mc), t_c, phi_c, alpha_M] için
    kovaryans matrisini ters çevirerek gerçekçi 1-sigma duyarlılıklarını hesaplar.
    
    Dönüş:
    - Dict: snr, sigma_alpha_M_marginalized, sigma_alpha_M_unmarginalized, cov, corr, f_isco
    """
    M_SUN = 1.98847e30
    MPC = 3.085677581e22

    # Kırmızıya kaymış kütleler
    M1 = M1_msun * M_SUN * (1.0 + z)
    M2 = M2_msun * M_SUN * (1.0 + z)
    M_total = M1 + M2
    Mc = (M1 * M2)**(3.0 / 5.0) / (M_total)**(1.0 / 5.0)
    dL = dL_mpc * MPC

    # ISCO kesim frekansı
    f_isco = c**3 / (6.0**1.5 * np.pi * G * M_total)

    # Frekans ızgarası
    f = np.logspace(np.log10(f_min), np.log10(f_isco), 2000)
    Sn = lisa_psd(f)
    A = gr_taylorf2_amplitude(f, Mc, dL)

    # 1. Fazın parametrelere göre analitik türevleri
    # dPsi / d(ln Mc)
    u = np.pi * G * Mc * f / c**3
    dPsi_dlnMc = -(5.0 / 128.0) * u**(-5.0 / 3.0)

    # dPsi / d(t_c)
    dPsi_dtc = 2.0 * np.pi * f

    # dPsi / d(phi_c)
    dPsi_dphic = -1.0 * np.ones_like(f)

    # dPsi / d(alpha_M)
    prefactor = (5.0 / 192.0) * (G * Mc / c**3)**(-5.0 / 3.0)
    dPsi_dalphaM = prefactor * (f_min**(-5.0 / 3.0) - f**(-5.0 / 3.0))

    # Genliğin ln(Mc)'ye göre türevi: d(ln A) / d(ln Mc) = 5/6
    dlnA_dlnMc = 5.0 / 6.0

    derivs = [dPsi_dlnMc, dPsi_dtc, dPsi_dphic, dPsi_dalphaM]
    n_params = len(derivs)
    Gamma = np.zeros((n_params, n_params))

    # 2. Fisher Matrisi Elemanlarının Hesaplanması
    for i in range(n_params):
        for j in range(n_params):
            integrand = 4.0 * (derivs[i] * derivs[j]) * (A**2 / Sn)
            if i == 0 and j == 0:
                integrand += 4.0 * (dlnA_dlnMc**2) * (A**2 / Sn)

            try:
                Gamma[i, j] = np.trapezoid(integrand, f)
            except AttributeError:
                Gamma[i, j] = np.trapz(integrand, f)

    # 3. SNR Hesabı
    try:
        snr2 = 4.0 * np.trapezoid(A**2 / Sn, f)
    except AttributeError:
        snr2 = 4.0 * np.trapz(A**2 / Sn, f)
    snr = np.sqrt(snr2)

    # 4. Tek parametreli (unmarginalized) hata
    sigma_unmarg = 1.0 / np.sqrt(Gamma[3, 3])

    # 5. Kovaryans Matrisi (Ters matris) ve Marjinalize Hata
    cov = np.linalg.inv(Gamma)
    sigma_marg = np.sqrt(cov[3, 3])

    # Korelasyon matrisi
    corr_matrix = np.zeros_like(cov)
    for i in range(n_params):
        for j in range(n_params):
            corr_matrix[i, j] = cov[i, j] / np.sqrt(cov[i, i] * cov[j, j])

    return {
        "snr": float(snr),
        "sigma_alpha_M_marginalized": float(sigma_marg),
        "sigma_alpha_M_unmarginalized": float(sigma_unmarg),
        "covariance_matrix": cov,
        "correlation_matrix": corr_matrix,
        "f_isco": float(f_isco),
        "Mc": float(Mc),
    }


def fisher_alpha_M(f, h_gr, alpha_M, Mc, Sn, f_min=1e-5):
    """Eski fonksiyonlarla geriye dönük uyumluluk için sarmalayıcı."""
    prefactor = (5.0 / 192.0) * (G * Mc / c**3)**(-5.0 / 3.0)
    dpsi_dalpha = prefactor * (f_min**(-5.0 / 3.0) - f**(-5.0 / 3.0))
    try:
        F = 4.0 * np.trapezoid((dpsi_dalpha * np.abs(h_gr))**2 / Sn, f)
    except AttributeError:
        F = 4.0 * np.trapz((dpsi_dalpha * np.abs(h_gr))**2 / Sn, f)
    return np.sqrt(1.0 / F)
