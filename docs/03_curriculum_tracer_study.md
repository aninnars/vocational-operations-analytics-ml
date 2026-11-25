# Case Study 3: Curriculum Evaluation & Alumni Career Outcome Prediction

## 1. Business Context & Problem Statement
Keberhasilan lembaga pelatihan vokasi tidak berhenti pada kelulusan peserta, melainkan pada **keterserapan alumni di dunia kerja atau kemandirian berwirausaha**. Institusi membutuhkan evaluasi kurikulum berbasis bukti empiris (*evidence-based curriculum evaluation*) untuk menjawab dua pertanyaan kunci:
1. Modul praktik mana yang paling menentukan status karir alumni (*Bekerja*, *Buka Usaha*, atau *Belum Bekerja*) dan mempercepat waktu tunggu kerja?
2. Bagaimana memprediksi jalur karir peserta sejak sebelum atau sesaat setelah uji kompetensi agar intervensi remedial dan penyaluran kerja tepat sasaran?

## 2. Empirical & Statistical Findings (`data/curriculum_tracer_anonymized.csv`)
- **Total Alumni Valid:** **315 alumni** (`ALM-001`–`ALM-315`).
- **Distribusi Outcome Karir:**
  - **Bekerja (Terserap Industri):** **148 alumni** (**47,0%**)
  - **Buka Usaha (Wirausaha Mandiri):** **95 alumni** (**30,2%**)
  - **Belum Bekerja (Perlu Pendampingan):** **72 alumni** (**22,9%**)
  - **Total Tingkat Serapan (Bekerja + Wirausaha):** **77,14%** (243 dari 315 alumni).

### A. Profil Rata-Rata Nilai Modul & Waktu Tunggu per Jalur Karir
| Status Alumni | Ukur Pola | Jahit Dasar | Jahit Lanjut | Finishing | Skor Uji Komp. | Rata-rata Waktu Tunggu |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bekerja (Industri)** | 80.41 | **87.96** | 81.89 | **87.17** | 85.42 | **4.54 bulan** |
| **Buka Usaha (Wirausaha)** | **89.09** | 83.06 | **88.31** | 81.55 | **86.77** | 5.49 bulan |
| **Belum Bekerja** | 71.42 | 73.85 | 70.20 | 70.15 | 71.07 | 18.15 bulan |

> **Insight Strategis yang Sangat Menonjol:**
> - **Jalur Industri (*Bekerja*)** sangat ditentukan oleh kekuatan eksekusi standar produksi (**Jahit Dasar** & **Finishing/QC**).
> - **Jalur Wirausaha (*Buka Usaha*)** sangat ditentukan oleh penguasaan desain & kustomisasi produk (**Ukur Pola** & **Jahit Lanjut**).
> - **Kelompok Belum Bekerja** mengalami defisit kompetensi merata (selisih **10–17 poin** di bawah kelompok terserap) dan menghadapi waktu tunggu 3–4× lebih lama.

### B. Analisis Korelasi Statistik (Pearson $r$ & Spearman $\rho$)
| Modul / Komponen | $r$ vs Terserap | $p$-value (Terserap) | $r$ vs Waktu Tunggu | $p$-value (Waktu Tunggu) |
| :--- | :---: | :---: | :---: | :---: |
| **Skor Uji Kompetensi** | **0.913** | $3.95 \times 10^{-113}$ | **-0.460** | $6.52 \times 10^{-13}$ |
| **Finishing** | **0.814** | $2.99 \times 10^{-69}$ | **-0.452** | $1.70 \times 10^{-12}$ |
| **Jahit Lanjut** | 0.781 | $3.02 \times 10^{-60}$ | 0.020 | 0.767 (ns) |
| **Jahit Dasar** | 0.754 | $6.32 \times 10^{-54}$ | -0.394 | $1.38 \times 10^{-9}$ |
| **Ukur Pola** | 0.684 | $5.95 \times 10^{-41}$ | 0.021 | 0.755 (ns) |

## 3. Interpretable Machine Learning (`DecisionTreeClassifier`)
Dipilih `DecisionTreeClassifier` di dalam `Pipeline(['imputer', 'classifier'])` agar aturan keputusan (*if–then rules*) dapat dipahami langsung oleh manajemen akademik dan instruktur — sekaligus toleran terhadap nilai modul yang belum lengkap (`NaN`).

| Skema Model | Fitur Input | 5-Fold CV Macro-F1 | Test Accuracy | Test Macro-F1 | Top Feature Importance |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Skema A (Pre-Exam)** | 4 Modul Praktik | 0.8951 | **93.65%** | **0.9437** | `Finishing` (**52.6%**), `Ukur Pola` (**40.6%**) |
| **Skema B (Comprehensive)** | 4 Modul + Skor Uji Komp. | **0.9286** | 90.48% | 0.9155 | `Skor Uji Kompetensi` (**55.2%**), `Ukur Pola` (**24.8%**), `Finishing` (**14.7%**) |

## 4. Prescriptive Curriculum Recommendations
1. **Perkuat Bobot & Standar Kelulusan Modul *Finishing* & *Ukur Pola*:** Kedua modul ini menyumbang **93,2%** *feature importance* pada Skema A — peningkatan kualitas pada keduanya langsung mendongkrak peluang terserap kerja dan memangkas waktu tunggu.
2. **Terapkan *Dual-Track Career Coaching*:**
   - Peserta dengan nilai kuat di *Jahit Dasar & Finishing* $\rightarrow$ prioritaskan penyaluran ke mitra industri garmen/konveksi.
   - Peserta dengan nilai kuat di *Ukur Pola & Jahit Lanjut* $\rightarrow$ berikan modul inkubasi bisnis mikro, *pricing*, dan pemasaran digital.
3. **Sistem Remedial Otomatis Sebelum Uji Kompetensi:** Gunakan Skema A sebelum ujian akhir untuk mendeteksi peserta yang diprediksi masuk kelas `Belum Bekerja`, lalu wajibkan klinik remedial terfokus.
