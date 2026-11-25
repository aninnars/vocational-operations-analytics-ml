# Data Privacy, Anonymization Protocol, & Team Scope Boundaries

Dokumen ini menjelaskan secara transparan bagaimana proyek ini menjaga **etika privasi data mitra** untuk publikasi repositori publik serta **kejelasan batas kontribusi** dalam proyek tim lintas fungsi.

---

## 1. Anonymization & Data Sanitization Protocol

Proyek asli dikerjakan bersama sebuah lembaga pelatihan vokasi pada periode 2026. Untuk kepentingan publikasi repositori terbuka (*public GitHub & live demo*), seluruh artefak di repositori ini telah melewati protokol sanitasi berikut:

| Kategori Sensitif | Perlakuan pada Repositori Publik |
| :--- | :--- |
| **Identitas Institusi** | Disamarkan menjadi *"Vocational Training Center (Anonymized Case Study)"*. |
| **Personal Identifiable Information (PII)** | Seluruh kolom `NAMA`, `NIK`, `NO.HANDPHONE`, `TANGGAL_LAHIR`, dan alamat detail **dihapus permanen** dari dataset publik. |
| **Identitas Peserta & Alumni** | ID operasional asli diganti menjadi *pseudonym* sekuensial (`STU-001`–`STU-349`, `ALM-001`–`ALM-315`) dengan tetap menjaga integritas relasi antar-tabel. |
| **Identitas Aset & Catatan Teknisi** | Kode aset asli diganti menjadi `AST-001`–`AST-046`; nama individu pada catatan perbaikan diganti menjadi label generik (`Panggil teknisi eksternal`). |
| **Atribut Geografis** | Nama wilayah/kecamatan pada dataset & metadata publik dipetakan ke kode wilayah anonim (`Zone A`–`Zone F`, `Other Zones`). |
| **Kredensial & Infrastruktur Live** | Tidak ada URL endpoint live, token API, file database operasional (`.db`), maupun *raw spreadsheet* (`.xlsx`) yang disertakan. |

Setiap perubahan sebelum rilis diverifikasi secara otomatis oleh skrip [`src/verify_release.py`](../src/verify_release.py).

---

## 2. Explicit Role & Scope Boundaries

Proyek ini merupakan hasil kolaborasi tim. Demi transparansi profesional, batas peran dijabarkan sebagai berikut:

### In-Scope (Kontribusi Utama Proyek — DS / ML / DA / BI)
- Perumusan masalah analitik dan metrik evaluasi bisnis untuk seluruh **3 pilar studi**:
  1. *Predictive Maintenance & IoT Proxy Workload Estimation*
  2. *Dual-Stage Early Warning System (EWS) Drop-Out*
  3. *Curriculum Evaluation & Alumni Tracer Study*
- *Exploratory Data Analysis (EDA)*, uji korelasi statistik (Pearson/Spearman), dan analisis akar masalah (*Pareto*).
- *Feature engineering*, penanganan *class imbalance*, *hyperparameter & threshold tuning*, serta serialisasi **7 pipeline `scikit-learn`**.
- Perancangan *unified inference engine* (`src/inference.py`), dokumentasi *case study*, dan lapisan visualisasi analitik (*Stage 2 Dashboard*).

### Out-of-Scope (Dikerjakan oleh Tim Data Engineering / Sistem Operasional)
- Pembangunan sistem formulir input admin (CRUD peserta, presensi, dan log servis).
- Pengembangan pipeline ETL *cloud* harian, validasi integritas NIK, sinkronisasi *foreign key*, dan pengelolaan *Silver Layer Data Warehouse*.
- Pembuatan laporan administratif otomatis untuk pelaporan operasional eksternal.
