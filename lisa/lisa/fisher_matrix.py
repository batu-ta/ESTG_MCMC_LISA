import numpy as np

def fisher_alpha_M(f, h_gr, alpha_M, Mc, Sn, f_min=1e-5):
    """
    Fisher matrisi yaklaşımı ile α_M için hassasiyeti hesaplar.
    
    Parametreler:
    - f: frekans dizisi (Hz)
    - h_gr: GR dalga formu (genlik)
    - alpha_M: ESTG parametresi (test değeri)
    - Mc: chirp kütlesi (kg)
    - Sn: gürültü spektral yoğunluğu (m²/Hz)
    - f_min: minimum frekans (Hz)
    
    Dönüş:
    - sigma_alpha: α_M için 1-sigma belirsizlik
    """
    G = 6.67430e-11
    c = 299792458.0
    
    # Fazın α_M'ye göre türevi (Denklem 21'den)
    prefactor = (5.0 / 192.0) * (G * Mc / c**3)**(-5.0/3.0)
    dpsi_dalpha = prefactor * (f_min**(-5.0/3.0) - f**(-5.0/3.0))
    
    # Fisher matrisi elemanı (tek parametre)
    F = 4 * np.trapz((dpsi_dalpha * np.abs(h_gr))**2 / Sn, f)
    
    # 1-sigma belirsizlik
    return np.sqrt(1.0 / F)
