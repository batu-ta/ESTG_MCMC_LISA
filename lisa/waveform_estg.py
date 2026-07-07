import numpy as np

def estg_phase(f, alpha_M, Mc, f_min=1e-5):
    """
    ESTG faz düzeltmesini hesaplar.
    
    Parametreler:
    - f: frekans dizisi (Hz)
    - alpha_M: ESTG parametresi
    - Mc: chirp kütlesi (kg)
    - f_min: minimum frekans (Hz)
    
    Dönüş:
    - Delta_psi: faz düzeltmesi (radyan)
    """
    G = 6.67430e-11
    c = 299792458.0
    
    factor = (5.0 / 192.0) * (G * Mc / c**3)**(-5.0/3.0)
    integral = (f_min**(-5.0/3.0) - f**(-5.0/3.0))
    
    return alpha_M * factor * integral

def snr_estg(f, h_gr, alpha_M, Mc, Sn, f_min=1e-5):
    """
    ESTG dalga formu için SNR hesaplar.
    
    Parametreler:
    - f: frekans dizisi (Hz)
    - h_gr: GR dalga formu genliği
    - alpha_M: ESTG parametresi
    - Mc: chirp kütlesi (kg)
    - Sn: gürültü spektral yoğunluğu
    - f_min: minimum frekans (Hz)
    
    Dönüş:
    - SNR değeri
    """
    delta_psi = estg_phase(f, alpha_M, Mc, f_min)
    h_estg = h_gr * np.exp(1j * delta_psi)
    
    # NumPy 2.0 uyumluluğu için trapz -> trapezoid
    try:
        snr2 = 4 * np.trapz(np.abs(h_estg)**2 / Sn, f)
    except AttributeError:
        snr2 = 4 * np.trapezoid(np.abs(h_estg)**2 / Sn, f)
    
    return np.sqrt(snr2)
