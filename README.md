# ESTG MCMC ve LISA Analiz Protokolü

Bu depo, "Entropik Skaler-Tensör Kütleçekimi (ESTG)" teorisinin Planck 2018, BBN ve LISA verileriyle kısıtlanmasına yönelik sayısal analiz protokolünü içerir.

## İçindekiler

- `config/base.param`: CLASS için parametre dosyası
- `montepython/`: ESTG için özel likelihood fonksiyonu
- `lisa/`: ESTG dalga formu ve Fisher matrisi hesaplamaları
- `notebooks/`: Jupyter notebook'lar ile adım adım analiz

## Gereksinimler

- Python 3.9+
- CLASS v3.2.0
- MontePython v3.4
- NumPy, SciPy, Matplotlib, GetDist

## Kurulum

```bash
git clone https://github.com/batu-ta/ESTG_MCMC_LISA.git
cd ESTG_MCMC_LISA
pip install -r requirements.txt
