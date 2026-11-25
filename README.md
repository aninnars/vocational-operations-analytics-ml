# Vocational Training Operations Analytics & End-to-End Machine Learning

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Domain](https://img.shields.io/badge/Domain-DS%20%7C%20ML%20%7C%20DA%20%7C%20BI-0EA5E9)](#)
[![Data Privacy](https://img.shields.io/badge/Data-Anonymized%20%26%20PII--Free-10B981)](docs/DATA_PRIVACY_AND_SCOPE.md)

> **End-to-End Data Science, Machine Learning, Data Analytics, & Business Intelligence Project**  
> Sistem analitik dan pemodelan Machine Learning terintegrasi untuk transformasi operasional **Lembaga Pelatihan Vokasi (Anonymized Case Study)** berbasis data historis 2017–2026+, mencakup tiga pilar keputusan strategis: **Asset Maintenance**, **Student Retention (Drop-Out EWS)**, dan **Curriculum & Alumni Outcome**.

---

## Executive Summary

Institusi pelatihan vokasi menghadapi tiga tantangan operasional utama yang saling terkait:
1. **Downtime Aset Produksi:** Kerusakan mesin jahit secara mendadak mengganggu jadwal kelas praktik karena pencatatan servis historis belum dimanfaatkan untuk prediksi perawatan preventif.
2. **Risiko Drop-Out Peserta:** Meski tingkat drop-out relatif rendah (**4,01%** dari 349 peserta), dampaknya tinggi terhadap efisiensi anggaran program dan kepatuhan kuota pelatihan bersubsidi.
3. **Kesenjangan Kurikulum vs Pasar Kerja:** Institusi membutuhkan bukti empiris modul praktik mana yang paling menentukan apakah alumni **Bekerja di Industri**, **Buka Usaha Mandiri**, atau **Belum Bekerja**.

Proyek ini menyatukan **7 pipeline Machine Learning siap pakai (`scikit-learn`)** dan lapisan **Diagnostic & Prescriptive Analytics** untuk mengubah data operasional pasif menjadi sistem pendukung keputusan (*Decision Support System*).

---

## Architecture & Role Scope

Proyek ini dikerjakan dalam skema kolaborasi tim lintas peran. Diagram berikut menegaskan batas kontribusi yang jelas antara **Data Engineering (Upstream)** dan **Data Science / Analytics (Scope Utama Proyek Ini)**:

```text
┌──────────────────────────────────────────────────────────────────────┐
│  UPSTREAM DATA INFRASTRUCTURE (Data Engineering Team)                │
│  Operational Forms / Sheets ─► Automated ETL ─► Clean Silver Layer   │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │  Anonymized Feature Snapshot
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│  CORE PROJECT SCOPE (DS / ML / DA / BI — Owned End-to-End)           │
│                                                                      │
│  1. Exploratory & Diagnostic Analytics (DA / BI)                     │
│     • Operational KPIs, Pareto Root-Cause, Statistical Correlation   │
│                                                                      │
│  2. Predictive Modeling & Evaluation (DS / ML)                       │
│     • Pillar 1: Predictive Maintenance (3-Class, 4-Class, Regressor) │
│     • Pillar 2: Dual-Stage Drop-Out EWS (LogReg + GradientBoosting)  │
│     • Pillar 3: Curriculum Tracer Study (Decision Tree Schema A & B) │
│                                                                      │
│  3. Unified Serving Layer & Interactive Dashboard                    │
│     • src/inference.py (7 sklearn Pipelines + Business Rules)        │
│     • notebooks/ (3 Local-Only Reproducible Jupyter Notebooks)       │
│     • docs/index.html (Local Modern Analytics Dashboard)             │
└──────────────────────────────────────────────────────────────────────┘
```

Detail lengkap protokol anonimisasi dan batasan peran dapat dibaca di [docs/DATA_PRIVACY_AND_SCOPE.md](docs/DATA_PRIVACY_AND_SCOPE.md).

---

## Three Core Case Studies

### 1. Predictive Maintenance & IoT Proxy Workload Engine
*Detail teknis: [docs/01_predictive_maintenance.md](docs/01_predictive_maintenance.md)*

- **Dataset:** 86 catatan servis pada **46 unit mesin jahit unik** (8 tipe mesin, periode 2018–2025).
- **Inovasi Feature Engineering (IoT Proxy):** Karena mesin tidak memiliki sensor IoT fisik, beban pemakaian direkonstruksi dari jadwal kelas & presensi menjadi `akumulasi_jp` (jam pelajaran), `rasio_beban_harian`, dan `jp_per_servis`.
- **Pendekatan Model:** Konsolidasi label 4-kelas (*sparse class* pada *Rusak Total* $n=1$) menjadi 3-kelas operasional (**Baik**, **Rusak Ringan / Waspada**, **Rusak Berat / Afkir**) dengan `RandomForestClassifier` (*GridSearchCV + balanced_subsample*) dan `RandomForestRegressor` untuk estimasi beban JP.
- **Key Result:** Konsolidasi 3-kelas meningkatkan Macro-F1 secara signifikan dibanding skema 4-kelas (**0.520** tuned / **0.559** baseline vs **0.307** pada 4-kelas) pada kondisi sampel kecil ($n=86$).

### 2. Dual-Stage Early Warning System (EWS) Deteksi Dini Drop-Out
*Detail teknis: [docs/02_dropout_ews.md](docs/02_dropout_ews.md)*

- **Dataset:** **349 peserta** (335 Selesai, 14 Drop-Out — *severe class imbalance* **4,01%**) + 14 log kualitatif alasan keluar.
- **Temuan Kunci (DA):** Rata-rata kehadiran peserta yang selesai adalah **95,74%**, sementara peserta drop-out hanya **57,67%** (selisih **38,07 poin persentase**). Tingkat drop-out tertinggi berada pada program **REGULER (6,74%)** dan **PKW (6,67%)**, sedangkan **PKK** terendah (**1,69%**).
- **Arsitektur Dual-Stage:**
  - **Stage 1 (Day-1 Pre-Enrollment Screening):** `LogisticRegression` dengan *decision threshold* terkalibrasi **`0.25`** untuk memaksimalkan *recall* (**66,7%**, ROC-AUC **0.741**) sejak hari pertama pendaftaran.
  - **Stage 2 (Ongoing Monitoring):** `GradientBoostingClassifier` dengan fitur presensi berjalan (`kehadiran_persen`) dan *threshold* **`0.50`**, mencapai **Precision 100%**, **Recall 66,7%**, **F1 0.800**, dan **ROC-AUC 0.953**.

### 3. Evaluasi Kurikulum Berbasis Tracer Study Alumni
*Detail teknis: [docs/03_curriculum_tracer_study.md](docs/03_curriculum_tracer_study.md)*

- **Dataset:** **315 alumni valid** dengan tingkat serapan total **77,14%** (**148 Bekerja [47,0%]**, **95 Buka Usaha [30,2%]**, **72 Belum Bekerja [22,9%]**).
- **Temuan Statistik & Empiris:**
  - Korelasi terkuat terhadap serapan karir adalah **Skor Uji Kompetensi ($r = 0.913$)** dan **Modul Finishing ($r = 0.814$)**; keduanya juga berkorelasi negatif signifikan terhadap waktu tunggu kerja ($r = -0.460$ dan $r = -0.452$, $p < 10^{-11}$).
  - **Profil Jalur Karir:** Alumni **Bekerja di Industri** unggul pada *Jahit Dasar* (87,96) & *Finishing* (87,17) dengan waktu tunggu tercepat (**4,54 bulan**); alumni **Buka Usaha Mandiri** unggul pada *Ukur Pola* (89,09) & *Jahit Lanjut* (88,31) (waktu tunggu **5,49 bulan**); sedangkan **Belum Bekerja** tertinggal di seluruh modul (~70–74, waktu tunggu **18,15 bulan**).
- **Performa Model Interpretable (`DecisionTreeClassifier`):**
  - **Skema A (4 Modul Praktik):** Accuracy **93,65%**, Macro-F1 **0.944**, 5-Fold CV Macro-F1 **0.895**.
  - **Skema B (4 Modul + Skor Uji Kompetensi):** Accuracy **90,48%**, Macro-F1 **0.916**, 5-Fold CV Macro-F1 **0.929**.

---

## Model Performance Summary

| Pillar | Model Artifact | Algorithm | Key Evaluation Metrics | Operational Role |
| :--- | :--- | :--- | :--- | :--- |
| **Maintenance (3-Class)** | `maintenance_classifier_3class.pkl` | Tuned Random Forest | Macro-F1 `0.520` (Baseline `0.559`) | Prioritas servis preventif 3 tingkat |
| **Maintenance (4-Class)** | `maintenance_classifier_4class.pkl` | Random Forest | Accuracy `38.9%`, Macro-F1 `0.307` | Benchmark pembanding *sparse class* |
| **Maintenance (Workload)** | `maintenance_regressor.pkl` | Random Forest Regressor | Test MAE `1714.8 JP`, 5-Fold CV `1756.9 JP` | Estimasi akumulasi beban jam praktik |
| **Drop-Out Stage 1** | `dropout_stage1_pre.pkl` | Logistic Regression (`thr=0.25`) | Recall `66.7%`, ROC-AUC `0.741` | Skrining risiko awal hari ke-1 |
| **Drop-Out Stage 2** | `dropout_stage2_ongoing.pkl` | Gradient Boosting (`thr=0.50`) | Precision `100%`, F1 `0.800`, ROC-AUC `0.953` | Peringatan dini berbasis presensi aktif |
| **Curriculum Schema A** | `curriculum_dt_skema_a.pkl` | Decision Tree (4 Modules) | Accuracy `93.65%`, Macro-F1 `0.944` | Rekomendasi remedial sebelum ujian |
| **Curriculum Schema B** | `curriculum_dt_skema_b.pkl` | Decision Tree (4 Modules + Exam) | 5-Fold CV F1 `0.929`, Accuracy `90.48%` | Pemetaan jalur karir & inkubasi usaha |

---

## Repository Structure

```text
vocational-operations-analytics-ml/
├── README.md                              # Executive overview & project documentation
├── LICENSE                                # MIT License
├── requirements.txt                       # Reproducible Python dependencies
├── .gitignore                             # Excludes raw data, secrets, DBs, local env
├── data/                                  # 100% local, anonymized & PII-free CSV snapshots
│   ├── maintenance_anonymized.csv         # 86 service logs across 46 masked assets (AST-001..046)
│   ├── dropout_anonymized.csv             # 349 masked students (STU-001..349, Zone A..Other)
│   ├── dropout_reasons_anonymized.csv     # 14 qualitative drop-out reasons (masked IDs)
│   └── curriculum_tracer_anonymized.csv   # 315 masked alumni records (ALM-001..315)
├── models/                                # Serialized scikit-learn Pipelines (imputer + scaler + model)
│   ├── maintenance_classifier_3class.pkl
│   ├── maintenance_classifier_4class.pkl
│   ├── maintenance_regressor.pkl
│   ├── dropout_stage1_pre.pkl
│   ├── dropout_stage2_ongoing.pkl
│   ├── curriculum_dt_skema_a.pkl
│   ├── curriculum_dt_skema_b.pkl
│   └── metadata/                          # Sanitized JSON model cards & evaluation metrics
├── notebooks/                             # 100% local reproducible analysis & modeling notebooks
│   ├── 01_predictive_maintenance_local.ipynb
│   ├── 02_dropout_ews_local.ipynb
│   └── 03_curriculum_tracer_study_local.ipynb
├── src/
│   ├── data_loader.py                     # Local-only CSV loader (zero cloud/API dependencies)
│   ├── inference.py                       # Unified Python ML inference engine + prescriptive rules
│   ├── export_web_artifacts.py            # Exports local CSV analytics + 7 sklearn pipelines to web JSON/JS
│   └── verify_release.py                  # Automated privacy leak scanner + Python vs JS parity test
└── docs/                                  # Local dashboard & Case Study Markdown docs
    ├── .nojekyll                          # Clean static asset serving on GitHub Pages
    ├── index.html                         # Modern Analytics Dark Mode Dashboard (5 Views + Live ML)
    ├── assets/                            # styles.css, charts.js (SVG), ml_engine.js, app.js
    ├── data/                              # dashboard_bundle.json & dashboard_bundle.js (local snapshot)
    ├── 01_predictive_maintenance.md
    ├── 02_dropout_ews.md
    ├── 03_curriculum_tracer_study.md
    └── DATA_PRIVACY_AND_SCOPE.md
```

---

## Local Analytics Dashboard Preview

Dashboard **Modern Analytics (Dark Mode)** berjalan **100% lokal & serverless** di browser — mengeksekusi ke-7 pipeline `scikit-learn` secara real-time di JavaScript dengan paritas penuh terhadap Python `.pkl`.

- **Cara Tercepat:** Cukup **klik ganda** file `docs/index.html` langsung di browser!
- **Atau via Local HTTP Server:**
  ```bash
  python -m http.server 8000 --directory docs
  # Buka http://localhost:8000
  ```

---

## Quickstart & Verification

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Opsional) Regenerasi bundel data & model web dari file lokal
python src/export_web_artifacts.py

# 3. Jalankan audit privasi otomatis + smoke test 7 model + uji paritas Python vs JS
python src/verify_release.py
```

### Example Python Inference Usage

```python
from src.inference import VocationalInferenceEngine

engine = VocationalInferenceEngine()

# Pillar 1: Predictive Maintenance
maint = engine.predict_maintenance(
    usia_mesin_hari=500,
    akumulasi_jp=2200.0,
    riwayat_servis_ke=1,
    jenis_mesin="HIGH_SPEED",
)

# Pillar 2: Dual-Stage Drop-Out EWS
ews = engine.predict_dropout(
    jenis_program="REGULER",
    bulan_mulai=3,
    jenis_kelamin="P",
    usia_masuk=26,
    pendidikan="SMA/SMK/Sederajat",
    jarak_km=7.0,
    zona="Zone A",
    kehadiran_persen=62.0,
)

# Pillar 3: Curriculum Outcome (supports missing module scores via internal pipeline imputer)
curr = engine.predict_curriculum_outcome(
    nilai_ukur_pola=87.5,
    nilai_jahit_dasar=82.0,
    nilai_jahit_lanjut=None,
    nilai_finishing=81.5,
    skor_uji_kompetensi=85.0,
)
```

---

## Status Rilis

- [x] **Stage 1 — Core Analytics & ML Foundation:** Anonimisasi data lokal, pembersihan PII, sanitasi metadata, unified `scikit-learn` inference engine, 3 local-only Jupyter notebooks, dan dokumentasi studi kasus lengkap.
- [x] **Stage 2 — Local Interactive Analytics Dashboard:** Dashboard 5-view berbahasa Indonesia dengan tema *Modern Analytics Dark Mode*, grafik SVG interaktif tanpa CDN, dan simulator *What-If* untuk 7 model ML dengan paritas Python vs JS terverifikasi (`docs/index.html`).
