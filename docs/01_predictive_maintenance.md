# Case Study 1: Predictive Maintenance & Machine Workload Estimation

## 1. Business Context & Problem Statement
Operasional pelatihan menjahit sangat bergantung pada ketersediaan mesin praktik (seperti *High Speed*, *Portable*, *Obras*, *Overdeck*, dan *Neci*). Sebelumnya, perawatan dilakukan secara **reaktif** (baru diperbaiki saat rusak), sehingga menimbulkan risiko gangguan jadwal pelatihan dan biaya perbaikan tak terduga.

Tantangan teknis utamanya: **tidak ada sensor IoT fisik** (seperti *runtime counter* atau sensor getaran) yang terpasang pada mesin.

## 2. Analytical & Feature Engineering Approach (IoT Proxy)
Untuk mengatasi ketiadaan sensor fisik, dikembangkan pendekatan **IoT Proxy Feature Engineering** dengan mengintegrasikan riwayat servis aset (`fact_log_mesin`) terhadap kalender beban jam pelatihan (`fact_kehadiran_kelulusan` dan `fact_jadwal_pembelajaran`):

| Fitur | Definisi / Formula | Alasan Bisnis |
| :--- | :--- | :--- |
| `usia_mesin_hari` | Selisih tanggal servis terhadap tanggal pembelian aset | Menangkap degradasi usia mekanis |
| `akumulasi_jp` | Estimasi akumulasi Jam Pelajaran operasional | Proksi *operating hours* tanpa sensor fisik |
| `riwayat_servis_ke` | Urutan frekuensi servis kumulatif per `asset_id` | Menangkap mesin yang sering bermasalah (*chronic assets*) |
| `rasio_beban_harian` | `akumulasi_jp / (usia_mesin_hari + 1)` | Mengukur intensitas pemakaian harian |
| `jp_per_servis` | `akumulasi_jp / max(riwayat_servis_ke, 1)` | Mengukur daya tahan antar-servis |
| `tipe_*` (8 fitur) | One-hot encoding tipe mesin terstandarisasi | Karakteristik keausan berbeda antar tipe mesin |

## 3. Dataset Summary (`data/maintenance_anonymized.csv`)
- **Total Log Servis Bersih:** 86 catatan pada **46 unit mesin unik** (`AST-001`–`AST-046`), rentang 2018–2025.
- **Tipe Mesin Dominan:** `HIGH_SPEED` (29 log), `PORTABLE` (27 log), `MESIN_JAHIT_LAINNYA` (8 log), `BUTTERFLY_MANUAL` (6 log), `SINGER` (6 log), `OBRAS` (4 log), `NECI` (3 log), `OVERDECK` (3 log).
- **Rata-rata Profil Aset:** Usia pakai **723,3 hari** (~2 tahun) dengan rata-rata akumulasi beban **2.120,9 JP**.
- **Distribusi Kondisi (3-Kelas):**
  - `0 — Baik (Normal)`: **45** (52,3%)
  - `1 — Rusak Ringan (Waspada)`: **29** (33,7%)
  - `2 — Rusak Berat / Afkir (Bahaya)`: **12** (14,0%) *(penggabungan 11 Rusak Berat + 1 Rusak Total)*

## 4. Modeling & Honest Evaluation
Seluruh model dibungkus dalam `sklearn.pipeline.Pipeline` (`SimpleImputer(strategy='median')` + estimator).

### A. Klasifikasi Kondisi Mesin
| Model / Eksperimen | Accuracy | Macro-F1 | Weighted-F1 | Catatan Evaluasi |
| :--- | :---: | :---: | :---: | :--- |
| **4-Class Random Forest (Pembanding)** | 38.89% | 0.3072 | — | Gagal pada kelas minoritas ekstrem (`Rusak Total` hanya 1 sampel) |
| **3-Class Baseline Random Forest** | 72.22% | 0.5585 | 0.6988 | Konsolidasi label 3-kelas menstabilkan klasifikasi risiko tinggi |
| **3-Class Cost-Sensitive RF** | 55.56% | 0.3652 | — | Over-penalizing pada sampel kecil |
| **3-Class Gradient Boosting** | 61.11% | 0.4497 | — | Cukup stabil namun kurang baik di kelas minoritas |
| **3-Class GridSearchCV Tuned RF (Deployed)** | 55.56% | **0.5199** | 0.5513 | `balanced_subsample`, `max_depth=3`, `n_estimators=50` — mengurangi overfitting & meningkatkan sensitivitas deteksi kerusakan |

### B. Regresi Estimasi Beban Kerja (`akumulasi_jp`)
| Model | Test MAE (JP) | Test RMSE (JP) | 5-Fold CV MAE (JP) | R² |
| :--- | :---: | :---: | :---: | :---: |
| Linear Regression | 1,889.35 | 2,135.27 | — | -0.3014 |
| **Random Forest Regressor (Deployed)** | **1,714.80** | 2,212.72 | **1,756.90** | -0.3975 |

> **Catatan Transparansi Ilmiah (Small-Sample Limitation):** Nilai $R^2$ negatif pada regresi dan varians CV pada klasifikasi mencerminkan keterbatasan ukuran sampel historis ($n=86$) serta variasi penggunaan aktual per mesin yang belum tercatat pada level unit. Menunjukkan limitasi ini secara jujur justru memperkuat validitas rekomendasi perbaikan pencatatan data di masa depan.

## 5. Prescriptive Business Recommendations
1. **Terapkan Skema Trafik 3-Level:** Gunakan output 3-kelas (Hijau/Kuning/Merah) sebagai dasar jadwal servis preventif sebelum memasuki angkatan kelas intensif (200 JP).
2. **Prioritaskan Tipe `HIGH_SPEED` & `PORTABLE`:** Dua tipe ini menyumbang **65,1%** dari seluruh histori servis sehingga perlu stok suku cadang rutin (jarum, spul, oli, v-belt).
3. **Upgrade Pencatatan Aset:** Catat `asset_id` langsung pada log praktik harian agar fitur `akumulasi_jp` dapat berubah dari estimasi proksi menjadi pengukuran presisi per mesin, sehingga akurasi model meningkat signifikan saat sampel bertambah.
