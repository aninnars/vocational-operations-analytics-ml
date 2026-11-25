"""Unified ML Inference Engine for Vocational Operations Analytics & ML Project.

Wraps all 7 serialized scikit-learn pipelines across 3 operational use-cases:
1. Predictive Maintenance & Machine Workload Estimation (3-class, 4-class, Regressor)
2. Dual-Stage Early Warning System (EWS) for Student Drop-Out (Pre-Enrollment & Ongoing)
3. Curriculum Evaluation & Alumni Career Outcome Prediction (Schema A & Schema B)

All public inputs use anonymized schemas (e.g., Zone A-F / Other Zones) while
preserving exact feature ordering expected by the underlying scikit-learn pipelines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"

MACHINE_TYPES = [
    "BUTTERFLY_MANUAL",
    "HIGH_SPEED",
    "MESIN_JAHIT_LAINNYA",
    "NECI",
    "OBRAS",
    "OVERDECK",
    "PORTABLE",
    "SINGER",
]

PROGRAMS = ["PKK", "PKW", "REGULER", "HARIAN"]
PROGRAM_DEFAULT_JP = {"PKK": 200.0, "PKW": 200.0, "REGULER": 150.0, "HARIAN": 51.0}

ZONES = [
    "Zone A",
    "Zone B",
    "Zone C",
    "Zone D",
    "Zone E",
    "Zone F",
    "Other Zones",
]

EDUCATION_MAP = {
    "SD/Sederajat": 0,
    "SMP/Sederajat": 1,
    "SMA/SMK/Sederajat": 2,
    "Perguruan Tinggi": 3,
}

GENDER_MAP = {"P": 0, "L": 1}


class VocationalInferenceEngine:
    """Lazy-loading inference wrapper for all 7 production ML pipelines."""

    def __init__(self, models_dir: Path | str = MODELS_DIR) -> None:
        self.models_dir = Path(models_dir)
        self._cache: Dict[str, Any] = {}

    def _load(self, filename: str) -> Any:
        if filename not in self._cache:
            path = self.models_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Model artifact not found: {path}")
            self._cache[filename] = joblib.load(path)
        return self._cache[filename]

    # ------------------------------------------------------------------
    # USE-CASE 1: PREDICTIVE MAINTENANCE
    # ------------------------------------------------------------------
    def predict_maintenance(
        self,
        usia_mesin_hari: float,
        akumulasi_jp: float,
        riwayat_servis_ke: int,
        jenis_mesin: str,
    ) -> Dict[str, Any]:
        """Predict 3-class & 4-class machine condition and expected workload (JP)."""
        if jenis_mesin not in MACHINE_TYPES:
            raise ValueError(f"Unsupported jenis_mesin '{jenis_mesin}'. Expected one of {MACHINE_TYPES}")

        rasio_beban_harian = akumulasi_jp / (usia_mesin_hari + 1.0)
        jp_per_servis = akumulasi_jp / max(int(riwayat_servis_ke), 1)

        clf_3c = self._load("maintenance_classifier_3class.pkl")
        clf_4c = self._load("maintenance_classifier_4class.pkl")
        reg = self._load("maintenance_regressor.pkl")

        base_clf = {
            "usia_mesin_hari": float(usia_mesin_hari),
            "akumulasi_jp": float(akumulasi_jp),
            "riwayat_servis_ke": float(riwayat_servis_ke),
            "rasio_beban_harian": float(rasio_beban_harian),
            "jp_per_servis": float(jp_per_servis),
        }
        for mt in MACHINE_TYPES:
            base_clf[f"tipe_{mt}"] = 1.0 if mt == jenis_mesin else 0.0

        df_clf = pd.DataFrame([base_clf])[list(clf_3c.feature_names_in_)]
        pred_3c = int(clf_3c.predict(df_clf)[0])
        proba_3c = clf_3c.predict_proba(df_clf)[0]
        pred_4c = int(clf_4c.predict(df_clf)[0])

        base_reg = {
            "usia_mesin_hari": float(usia_mesin_hari),
            "riwayat_servis_ke": float(riwayat_servis_ke),
        }
        for mt in MACHINE_TYPES:
            base_reg[f"tipe_{mt}"] = 1.0 if mt == jenis_mesin else 0.0
        df_reg = pd.DataFrame([base_reg])[list(reg.feature_names_in_)]
        est_jp = float(reg.predict(df_reg)[0])

        labels_3c = {
            0: "Baik (Normal)",
            1: "Rusak Ringan (Waspada)",
            2: "Rusak Berat / Afkir (Bahaya)",
        }
        labels_4c = {
            0: "Baik",
            1: "Rusak Ringan",
            2: "Rusak Berat",
            3: "Rusak Total",
        }
        actions_3c = {
            0: "Pertahankan perawatan rutin (pembersihan debu & pelumasan berkala).",
            1: "Jadwalkan inspeksi teknisi & servis preventif sebelum kelas intensif berikutnya.",
            2: "Hentikan pemakaian intensif; lakukan overhaul komponen atau pertimbangkan peremajaan aset.",
        }

        return {
            "derived_features": {
                "rasio_beban_harian": round(rasio_beban_harian, 4),
                "jp_per_servis": round(jp_per_servis, 2),
            },
            "classification_3class": {
                "code": pred_3c,
                "label": labels_3c[pred_3c],
                "probabilities": {
                    "Baik": round(float(proba_3c[0]), 4),
                    "Rusak Ringan": round(float(proba_3c[1]), 4),
                    "Rusak Berat / Afkir": round(float(proba_3c[2]), 4),
                },
            },
            "classification_4class": {
                "code": pred_4c,
                "label": labels_4c[pred_4c],
            },
            "estimated_akumulasi_jp": round(est_jp, 2),
            "recommendation": actions_3c[pred_3c],
        }

    # ------------------------------------------------------------------
    # USE-CASE 2: DUAL-STAGE EARLY WARNING SYSTEM (DROP-OUT)
    # ------------------------------------------------------------------
    def _build_dropout_dataframe(
        self,
        model: Any,
        paket_jp_diambil: float,
        bulan_mulai: int,
        jenis_kelamin: str,
        usia_masuk: float,
        pendidikan: str,
        jarak_km: float,
        jenis_program: str,
        zona: str,
        kehadiran_persen: Optional[float] = None,
    ) -> pd.DataFrame:
        if jenis_kelamin not in GENDER_MAP:
            raise ValueError(f"Unsupported jenis_kelamin '{jenis_kelamin}'. Expected one of {list(GENDER_MAP)}")
        if pendidikan not in EDUCATION_MAP:
            raise ValueError(f"Unsupported pendidikan '{pendidikan}'. Expected one of {list(EDUCATION_MAP)}")
        if jenis_program not in PROGRAMS:
            raise ValueError(f"Unsupported jenis_program '{jenis_program}'. Expected one of {PROGRAMS}")
        if zona not in ZONES:
            raise ValueError(f"Unsupported zona '{zona}'. Expected one of {ZONES}")

        cols = list(model.feature_names_in_)
        row: Dict[str, float] = {c: 0.0 for c in cols}

        # Numeric & ordinal features (indices 0..5)
        row[cols[0]] = float(paket_jp_diambil)
        row[cols[1]] = float(bulan_mulai)
        row[cols[2]] = float(GENDER_MAP[jenis_kelamin])
        row[cols[3]] = float(usia_masuk)
        row[cols[4]] = float(EDUCATION_MAP[pendidikan])
        row[cols[5]] = float(jarak_km)

        # Program one-hot slots (indices 6..9: PKK, PKW, REGULER, HARIAN)
        prog_idx = PROGRAMS.index(jenis_program)
        row[cols[6 + prog_idx]] = 1.0

        # Anonymized geographic zone one-hot slots (indices 10..16: Zone A..F, Other Zones)
        zone_idx = ZONES.index(zona)
        row[cols[10 + zone_idx]] = 1.0

        # Stage 2 attendance feature (index 17)
        if len(cols) >= 18:
            row[cols[17]] = np.nan if kehadiran_persen is None else float(kehadiran_persen)

        return pd.DataFrame([row])[cols]

    def predict_dropout(
        self,
        jenis_program: str,
        bulan_mulai: int,
        jenis_kelamin: str,
        usia_masuk: float,
        pendidikan: str,
        jarak_km: float,
        zona: str,
        kehadiran_persen: Optional[float] = None,
        paket_jp_diambil: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run Stage 1 (Pre-Enrollment) and optional Stage 2 (Ongoing) drop-out EWS."""
        jp = float(paket_jp_diambil) if paket_jp_diambil is not None else PROGRAM_DEFAULT_JP[jenis_program]

        model_pre = self._load("dropout_stage1_pre.pkl")
        df_pre = self._build_dropout_dataframe(
            model=model_pre,
            paket_jp_diambil=jp,
            bulan_mulai=bulan_mulai,
            jenis_kelamin=jenis_kelamin,
            usia_masuk=usia_masuk,
            pendidikan=pendidikan,
            jarak_km=jarak_km,
            jenis_program=jenis_program,
            zona=zona,
        )
        prob_pre = float(model_pre.predict_proba(df_pre)[0, 1])
        thr_pre = 0.25
        pred_pre = int(prob_pre >= thr_pre)

        result: Dict[str, Any] = {
            "stage1_pre_enrollment": {
                "probability_dropout": round(prob_pre, 4),
                "threshold": thr_pre,
                "predicted_code": pred_pre,
                "predicted_label": "KELUAR (Berisiko Drop-Out)" if pred_pre == 1 else "SELESAI (Risiko Rendah)",
            }
        }

        if kehadiran_persen is not None:
            model_ong = self._load("dropout_stage2_ongoing.pkl")
            df_ong = self._build_dropout_dataframe(
                model=model_ong,
                paket_jp_diambil=jp,
                bulan_mulai=bulan_mulai,
                jenis_kelamin=jenis_kelamin,
                usia_masuk=usia_masuk,
                pendidikan=pendidikan,
                jarak_km=jarak_km,
                jenis_program=jenis_program,
                zona=zona,
                kehadiran_persen=kehadiran_persen,
            )
            prob_ong = float(model_ong.predict_proba(df_ong)[0, 1])
            thr_ong = 0.50
            pred_ong = int(prob_ong >= thr_ong)
            result["stage2_ongoing_monitoring"] = {
                "probability_dropout": round(prob_ong, 4),
                "threshold": thr_ong,
                "predicted_code": pred_ong,
                "predicted_label": "KELUAR (Berisiko Drop-Out)" if pred_ong == 1 else "SELESAI (Aman)",
            }

            if pred_ong == 1:
                result["recommendation"] = (
                    "Intervensi segera: lakukan konseling personal, evaluasi kendala kehadiran/transportasi, "
                    "dan tawarkan fleksibilitas jadwal praktik."
                )
            elif pred_pre == 1:
                result["recommendation"] = (
                    "Status saat ini terkendali, namun profil awal berisiko; pantau presensi mingguan secara berkala."
                )
            else:
                result["recommendation"] = "Peserta berada pada jalur penyelesaian yang stabil; pertahankan monitoring standar."
        else:
            result["recommendation"] = (
                "Berikan orientasi intensif & check-in minggu pertama karena profil awal masuk kategori risiko."
                if pred_pre == 1
                else "Profil awal stabil; ikuti alur orientasi standar."
            )

        return result

    # ------------------------------------------------------------------
    # USE-CASE 3: CURRICULUM EVALUATION & ALUMNI CAREER OUTCOME
    # ------------------------------------------------------------------
    def predict_curriculum_outcome(
        self,
        nilai_ukur_pola: Optional[float],
        nilai_jahit_dasar: Optional[float],
        nilai_jahit_lanjut: Optional[float],
        nilai_finishing: Optional[float],
        skor_uji_kompetensi: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Predict alumni career trajectory using Schema A (4 modules) and Schema B (4 modules + exam)."""
        model_a = self._load("curriculum_dt_skema_a.pkl")
        model_b = self._load("curriculum_dt_skema_b.pkl")

        def _val(x: Optional[float]) -> float:
            return np.nan if x is None else float(x)

        labels = {
            0: "Bekerja (Terserap Industri)",
            1: "Buka Usaha (Wirausaha Mandiri)",
            2: "Belum Bekerja (Perlu Pendampingan)",
        }
        actions = {
            0: "Arahkan ke kanal penyaluran kerja mitra garmen/konveksi dan perkuat portofolio kualitas jahitan.",
            1: "Berikan pendampingan bisnis mikro, penetapan harga jasa, dan pemasaran digital untuk usaha mandiri.",
            2: "Prioritaskan remedial pada modul Finishing & Ukur Pola sebelum uji kompetensi ulang.",
        }

        df_a = pd.DataFrame(
            [
                {
                    "nilai_ukur_pola": _val(nilai_ukur_pola),
                    "nilai_jahit_dasar": _val(nilai_jahit_dasar),
                    "nilai_jahit_lanjut": _val(nilai_jahit_lanjut),
                    "nilai_finishing": _val(nilai_finishing),
                }
            ]
        )[list(model_a.feature_names_in_)]
        pred_a = int(model_a.predict(df_a)[0])
        proba_a = model_a.predict_proba(df_a)[0]

        result: Dict[str, Any] = {
            "skema_a_4modul": {
                "code": pred_a,
                "label": labels[pred_a],
                "probabilities": {
                    "Bekerja": round(float(proba_a[0]), 4),
                    "Buka Usaha": round(float(proba_a[1]), 4),
                    "Belum Bekerja": round(float(proba_a[2]), 4),
                },
            }
        }

        df_b = pd.DataFrame(
            [
                {
                    "nilai_ukur_pola": _val(nilai_ukur_pola),
                    "nilai_jahit_dasar": _val(nilai_jahit_dasar),
                    "nilai_jahit_lanjut": _val(nilai_jahit_lanjut),
                    "nilai_finishing": _val(nilai_finishing),
                    "skor_uji_kompetensi": _val(skor_uji_kompetensi),
                }
            ]
        )[list(model_b.feature_names_in_)]
        pred_b = int(model_b.predict(df_b)[0])
        proba_b = model_b.predict_proba(df_b)[0]

        result["skema_b_komprehensif"] = {
            "code": pred_b,
            "label": labels[pred_b],
            "probabilities": {
                "Bekerja": round(float(proba_b[0]), 4),
                "Buka Usaha": round(float(proba_b[1]), 4),
                "Belum Bekerja": round(float(proba_b[2]), 4),
            },
        }
        primary_code = pred_b if skor_uji_kompetensi is not None else pred_a
        result["recommendation"] = actions[primary_code]
        return result
