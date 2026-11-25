# Case Study 2: Dual-Stage Early Warning System (EWS) for Student Drop-Out

## 1. Business Context & Problem Statement
Tingkat drop-out peserta pelatihan vokasi secara historis berada di angka **4,01%** (14 dari 349 peserta). Meski tampak kecil secara persentase, setiap kejadian drop-out berdampak langsung pada hilangnya pemanfaatan kursi pelatihan, potensi tidak tercapainya target penyelesaian program bersubsidi, serta ketidakefisienan alokasi instruktur dan mesin.

Tujuan analitik ini bukan sekadar memprediksi *setelah* peserta berhenti, melainkan membangun **Early Warning System (EWS) 2-Tahap**:
- **Stage 1 (Pre-Enrollment / Hari ke-1):** Mendeteksi profil peserta berisiko tinggi sejak pendaftaran menggunakan data demografis, jarak, dan program.
- **Stage 2 (Ongoing Monitoring):** Memperbarui skor risiko secara presisi tinggi ketika data kehadiran berjalan (`kehadiran_persen`) mulai terbentuk.

## 2. Diagnostic Findings (`data/dropout_anonymized.csv` & `data/dropout_reasons_anonymized.csv`)
- **Total Populasi Analisis:** **349 peserta** (`335 Selesai` [95,99%] vs `14 Keluar / Drop-Out` [4,01%]).
- **Sinyal Perilaku Terkuat — Gap Presensi:**
  - Rata-rata kehadiran peserta **Selesai:** **95,74%**
  - Rata-rata kehadiran peserta **Drop-Out:** **57,67%** *(selisih tajam 38,07 poin persentase)*
- **Distribusi Drop-Out per Program:**
  - `REGULER`: **6,74%** (6 dari 89 peserta)
  - `PKW`: **6,67%** (3 dari 45 peserta)
  - `HARIAN`: **5,41%** (2 dari 37 peserta)
  - `PKK`: **1,69%** (3 dari 178 peserta — paling rendah karena ikatan kuota & komitmen program)
- **Akar Penyebab Kualitatif (14 Kasus Drop-Out):**
  - **Bekerja** (3 kasus), **Kesibukan** (3 kasus), **Menikah** (2 kasus), **Ekonomi** (2 kasus), **Sakit** (2 kasus), dan **Modul terlalu susah** (2 kasus).

## 3. Dual-Stage Modeling & Threshold Engineering
Menghadapi **severe class imbalance (4,01%)**, penggunaan *accuracy* dan *default threshold* `0.50` pada Stage 1 akan gagal menangkap kasus minoritas. Karena itu diterapkan **Pipeline (`SimpleImputer` + `StandardScaler` + Classifier)** dengan *threshold tuning* berbasis Out-of-Fold (OOF):

| Stage | Model Terpilih | Fitur | Decision Threshold | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Stage 1: Pre-Enrollment** | Logistic Regression | 17 fitur (demografi, jarak, program, zona geografis anonim) | **0.25** | 5.41% | **66.67%** | 0.100 | **0.741** |
| **Stage 2: Ongoing Monitoring** | Gradient Boosting | 18 fitur (Stage 1 + `kehadiran_persen`) | **0.50** | **100.00%** | **66.67%** | **0.800** | **0.953** |

### Mengapa Trade-Off Precision–Recall Ini Tepat Secara Bisnis?
- Pada **Stage 1**, biaya *False Positive* sangat rendah (hanya berupa sambutan orientasi lebih hangat dan check-in minggu pertama), sedangkan biaya *False Negative* tinggi (kehilangan peserta tanpa sempat didampingi). Threshold `0.25` sengaja memprioritaskan **Recall 66,7%**.
- Pada **Stage 2**, intervensi bersifat intensif (konseling personal / penyesuaian jadwal). Dengan penambahan fitur kehadiran, **Gradient Boosting** melonjak ke **Precision 100%** dan **ROC-AUC 0.953**, sehingga tim akademik hanya mengintervensi peserta yang benar-benar kritis tanpa *alert fatigue*.

## 4. Prescriptive Action Plan
1. **Trigger Presensi Otomatis:** Jadikan penurunan presensi di bawah **75%** sebagai ambang evaluasi wajib sebelum peserta jatuh ke zona kritis (<60%).
2. **Fleksibilitas Jadwal untuk Program `REGULER` & `PKW`:** Karena 6 dari 14 alasan keluar adalah *Bekerja* dan *Kesibukan*, sediakan opsi kelas pengayaan/pengganti pada akhir pekan.
3. **Klinik Modul Awal:** Untuk mengatasi kendala *Modul terlalu susah*, terapkan evaluasi pemahaman singkat pada 20% awal pertemuan untuk memetakan kebutuhan bimbingan tambahan.
