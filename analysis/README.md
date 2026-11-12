# ESTG: gerçek Planck + BBN analizi ve LISA standart siren tahmini

Bu klasör, `run_all_analyses.py` ve notebook'lardaki elle yazılmış Gaussian
likelihood'ların yerini alan analizi içerir. Eski dosyalar karşılaştırma için
yerinde bırakıldı; makaledeki sayılar bu klasördeki çıktılardan yeniden
üretilmelidir.

| Dosya | Ne yapar |
|---|---|
| `setup_wsl.sh` | WSL2 içinde tek seferlik kurulum: Miniforge, Cobaya, hi_class (`hiclassy`, kaynaktan), Planck 2018 verisi |
| `HICLASS_COMMIT.txt` | Kullanılan hi_class sürümünün git commit'i |
| `cosmo/estg_theory.py` | hi_class için Cobaya teori sınıfı: β → ω_BD = 1/(4β²), α_M0/α_B0 türetilir |
| `cosmo/make_configs.py` | ESTG ve GR için `test`, `mcmc`, `minimize` YAML dosyalarını üretir |
| `cosmo/postprocess.py` | R−1, α_M0 sınırları, Savage–Dickey, Δχ² ve ΔAIC — hepsi zincir dosyalarından |
| `lisa/standard_siren_forecast.py` | α_M'nin d_L^GW/d_L^EM üzerinden LISA ile ölçülebilirliği |
| `lisa/phase_degeneracy_check.py` | Eski Fisher sonucunun (σ ≈ 2.8×10⁻⁴) neden ölçüm olmadığını gösterir |

## Model

Makaledeki Jordan-çerçeve eylemi (Denklem 3), F(φ) = M_P² e^(2βφ/M_P) ve
kanonik kinetik terimle, Φ = F(φ) değişken değişikliği altında **Brans–Dicke**
teorisidir:

    ω_BD = F / F'² = M_P² / (4β² F) ≈ 1 / (4β²)

F, BBN'den sonra M_P²'nin yüzde birkaçı içinde kaldığı için ω_BD neredeyse
sabittir. Kozmolojik ölçeklerde chameleon potansiyeli kozmolojik sabit gibi
davranır (Wang, Hui & Khoury 2012). Bu yüzden ESTG'nin kozmolojik limiti
Brans–Dicke + Λ'dır. hi_class bunu kovaryant olarak `gravity_model = brans_dicke`
ile çözer; bugünkü M_*² = 1'e ayarlanır, yani bugün G = Newton sabiti.

- Örneklenen parametre: **β** ∈ [0, 0.1] (düz prior; işareti fiziğe girmez)
- Türetilenler: ω_BD, hi_class arka planından **α_M0** ve α_B0 (= −α_M0)
- Yaklaşık ilişki: α_M0 ≈ 6 β². Makaledeki "α_M0 ≈ 0.65 β" doğru değil.

**Makaledeki α-parametrizasyonu (Denklem 15) neden kullanılmıyor:** hi_class'ın
`propto_omega` modeli α_B = −α_M (skaler-tensör ilişkisi) ve α_T = 0 ile
**her c_M ≠ 0 için gradient-kararsız**: c_M > 0'da erken zamanda, c_M < 0'da
bugün c_s² < 0. hi_class 3.4.0.1'in kendi `brans_dicke` modelinde
braiding/running oranının tam −1 çıktığı, yani Bellini–Sawicki konvansiyonunun
kullanıldığı sayısal olarak doğrulandı. α ∝ Ω_DE zaman bağımlılığı tutarlı bir
Brans–Dicke arka planının üretebileceği bir şey değil.

**Doğrulama:** β = 0'da ESTG ile GR modelinin toplam χ² farkı 0.016. Bu,
iki kod yolu arasındaki sayısal gürültü düzeyi.

Veri: Planck 2018 low-ℓ TT, low-ℓ EE, high-ℓ TTTEEE (plik_lite, native) ve
lensing, artı ω_b üzerinde BBN Gaussian'ı (0.0222 ± 0.0005, makaledeki değer;
Cooke+2018 tablosundan teyit et). Eski kodlardaki "LIGO O3: |α_M| < 0.05"
terimi kaldırıldı, çünkü atıf yapılan GWTC-2 makalesinde böyle bir sınır yok.

## Çalıştırma

### 0. WSL2 ve Ubuntu'yu D: diskine kur (bir kez)

Ubuntu'nun sanal diski (`ext4.vhdx`) ve analizle ilgili her şey (Miniforge,
hi_class, Planck verisi, zincirler) D:'de durur. C:'ye yalnızca Windows'un
kendi WSL bileşeni (birkaç yüz MB) kurulur.

a) **Yönetici** PowerShell'de WSL'i dağıtım olmadan kur, sonra bilgisayarı
yeniden başlat:

```
wsl --install --no-distribution
```

b) Yeniden başlattıktan sonra normal bir PowerShell'de:

```
wsl --update
wsl --install -d Ubuntu --location D:\WSL\Ubuntu
```

Açılan Ubuntu penceresinde bir kullanıcı adı ve şifre belirle (Windows
şifrenden bağımsız).

c) Doğrula: `D:\WSL\Ubuntu\ext4.vhdx` dosyası var olmalı.

`--location` tanınmazsa: önce `wsl --install -d Ubuntu` ile normal kur, sonra
`wsl --shutdown` ve `wsl --manage Ubuntu --move D:\WSL\Ubuntu` ile D:'ye taşı.

`setup_wsl.sh`, Ubuntu D:'de değilse çalışmayı reddeder.

### 1. Kurulum (Ubuntu terminali)

```
cd /mnt/c/Users/User/ESTG_MCMC_LISA/analysis
bash setup_wsl.sh
```

Kurulum 20–40 dakika sürer ve birkaç GB veri indirir. Hepsi `~/estg`
altına, yani D:'deki Ubuntu diskine gider; zincirler `~/estg/chains`
klasörüne yazılır. Windows Gezgini'nde `\\wsl$\Ubuntu\home\<kullanıcı>\estg`
adresinden görülebilirler. Sonraki her yeni terminalde şunu çalıştır:

```
source ~/estg/miniforge/etc/profile.d/conda.sh && conda activate estg
export COBAYA_PACKAGES_PATH=~/estg/packages
cd /mnt/c/Users/User/ESTG_MCMC_LISA/analysis/cosmo
```

### 2. Duman testi (dakikalar)

```
cobaya-run configs/gr_test.yaml
cobaya-run configs/estg_test.yaml
```

Her iki komut da tek bir noktada hi_class'ı çalıştırıp χ² değerlerini
yazdırmalı (sonlu sayılar, `inf` değil). Planck'ın en iyi uyum noktasına yakın
bir noktada toplam CMB χ² ≈ 1018 civarındadır; rastgele bir referans noktada
daha yüksek çıkması normal. Hata alırsan zincirlere geçme.

### 3a. Profil likelihood (makaledeki sonuç; 2 çekirdekle ~1 saat)

```
bash run_profile.sh            # CORES=1 ile daha hafif çalışır
```

β ızgarasındaki her noktada diğer parametreleri en iyi uyuma getirir. Düşük
öncelikle çalışır; durdurulup yeniden başlatılabilir, biten noktalar atlanır.
Özet, grafik ve en iyi uyum dosyaları `results/` klasörüne
`python profile_beta.py --summary-only` ile yazdırılabilir.

### 3b. MCMC (isteğe bağlı, 1–3 gün)

```
export OMP_NUM_THREADS=4
mpirun -n 4 cobaya-run configs/estg_mcmc.yaml
```

Bu komut 4 zincir koşar ve R−1 < 0.01 olunca kendiliğinden durur. Her
değerlendirme yaklaşık 2–4 saniye sürer. Kesilirse aynı komuta `--resume`
eklenerek devam eder. İstersen aynısını `gr_mcmc.yaml` için de çalıştır.

Pencereyi kapatınca zincirlerin durmaması için komutu `nohup ... > estg.log 2>&1 &`
ile başlat ve ilerlemeyi `tail -f estg.log` ile izle.

### 4. En iyi uyum (saatler)

```
mpirun -n 4 cobaya-run configs/estg_minimize.yaml
mpirun -n 4 cobaya-run configs/gr_minimize.yaml
```

### 5. Sonuçlar

```
python postprocess.py --chains-dir ~/estg/chains
```

### 6. LISA

```
cd ../lisa
python phase_degeneracy_check.py
python standard_siren_forecast.py
python standard_siren_forecast.py --sigma-inst 0.02 --delensing 0.5
```

## Bilinen sınırlamalar

- hi_class (Şubat 2026'dan beri) Python'da `classy` yerine `hiclassy.HiClass`
  olarak geliyor. Cobaya yalnızca `classy` modülünü tanıdığından,
  `estg_theory.py` `hiclassy`'yi kendisi yüklüyor ve Cobaya 3.6.2'nin
  `classy.initialize` adımlarını tekrarlıyor. Cobaya güncellenirse bu kısmı
  kontrol et.
- ω_BD sabit kabul ediliyor. Gerçek ESTG'de ω_BD ∝ 1/F ve F yüzde birkaç
  değiştiği için bu, ω_BD'de yüzde birkaçlık bir yaklaşıklık demek.
- Lensing likelihood'u doğrusal olmayan P(k) ister. Cobaya'nın varsayılanı
  HMcode; bu ΛCDM için kalibre edilmiştir, modifiye kütleçekimde bir
  yaklaşıklıktır. Bunu makalede belirt ya da lensing'siz bir koşuyla karşılaştır.
- Savage–Dickey Bayes faktörü β prior genişliğine bağlıdır ([0, 0.1]) ve
  β = 0 prior'un kenarında. Sayıyı prior ile birlikte ver.
- LISA siren tahmininde Brans–Dicke'nin M*²(z) evrimi `lisa/bd_siren_shape.py`
  ile hi_class'tan alınır (`--shape-file bd_shape.txt`). Bu dosya olmadan
  betik fenomenolojik α_M ∝ Ω_DE biçimini kullanır.
- Siren tahmini basit bir Fisher hesabıdır. Gerçek bir LISA katalog
  simülasyonu (popülasyon modeli, gökyüzü konumu, EM karşılığı olasılığı)
  değildir. Siren sayısını literatürdeki popülasyon modellerine göre seç
  (Tamanini et al. 2016).
- H0 ve Ω_m prior'ları siren sonucunun alt sınırını belirler (çok sayıda
  sirende σ ≈ 0.04'te doyuyor). Prior'ları kendi MCMC posterior'undan almak
  daha tutarlı olur.
