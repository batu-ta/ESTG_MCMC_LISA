# ESTG MCMC ve LISA Analiz Protokolü

Bu depo, **"Entropik Skaler-Tensör Kütleçekimi (ESTG)"** teorisinin Planck 2018, BBN ve LISA verileriyle kısıtlanmasına yönelik sayısal analiz protokolünü ve simülasyon kodlarını içerir.

## İçindekiler

- `run_all_analyses.py`: Tüm analiz protokolünü (MCMC, AIC/Bayes, LISA Fisher) tek seferde koşturan ana betik
- `lisa/`: 
  - `waveform_estg.py`: Robson et al. (2019) analitik LISA gürültü modeli ($S_n(f)$) ve 0PN TaylorF2 dalga formu
  - `fisher_matrix.py`: Çok parametreli $(\ln \mathcal{M}_c, t_c, \phi_c, \alpha_M)$ marjinalize edilmiş Fisher Bilgi Matrisi
- `montepython/`: ESTG için MontePython likelihood fonksiyonu
- `config/base.param`: CLASS kozmolojik parametre dosyası
- `notebooks/`: 
  - `01_mcmc_analysis.ipynb`: 4 zincirli Metropolis-Hastings MCMC, Gelman-Rubin tanıları ve GetDist analizi
  - `02_model_comparison.ipynb`: $\chi^2$ optimizasyonu, AIC ve Savage-Dickey yoğunluk oranıyla Bayes Faktörü
  - `03_lisa_sensitivity.ipynb`: LISA duyarlılık eğrisi vs dalga formu gerinimi ve SNR hesaplamaları

## Gereksinimler

- Python 3.9+
- NumPy, SciPy, Matplotlib, GetDist

```bash
pip install -r requirements.txt
```

## Hızlı Başlangıç

Tüm analizleri tek komutla çalıştırmak ve doğrulamak için:

```bash
python run_all_analyses.py
```
