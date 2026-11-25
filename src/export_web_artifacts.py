"""Export Local Datasets, Analytics Summaries, and 7 ML Pipelines to Static Web Bundle.

Generates:
- docs/data/dashboard_bundle.json  (clean JSON for GitHub Pages & inspection)
- docs/data/dashboard_bundle.js    (window.PORTFOLIO_BUNDLE wrapper for zero-CORS local file:// preview)

All model weights and tree structures are exported directly from local `.pkl`
pipelines so client-side JavaScript inference matches Python `src/inference.py`
with floating-point parity (< 1e-5). Zero PII or original regional identifiers
are included in the exported bundle.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
META_DIR = MODELS_DIR / "metadata"
OUT_DIR = REPO_ROOT / "docs" / "data"


def _round_list(arr: Any, ndigits: int = 6) -> List[float]:
    return [round(float(x), ndigits) for x in arr]


def _serialize_tree(tree_: Any, is_classifier: bool) -> Dict[str, Any]:
    """Compact array-based serialization of a sklearn Tree object."""
    values = tree_.value
    if is_classifier:
        # Normalize each node's class counts into probabilities (exact sklearn behavior)
        flat_probs = []
        for node_val in values:
            vec = np.asarray(node_val[0], dtype=float)
            total = vec.sum()
            probs = (vec / total) if total > 0 else np.zeros_like(vec)
            flat_probs.append([round(float(p), 6) for p in probs])
        val_out: Any = flat_probs
    else:
        val_out = [round(float(v[0][0]), 6) for v in values]

    return {
        "f": [int(x) for x in tree_.feature],
        "t": [round(float(x), 6) for x in tree_.threshold],
        "l": [int(x) for x in tree_.children_left],
        "r": [int(x) for x in tree_.children_right],
        "v": val_out,
    }


def _export_pipeline(pkl_path: Path, kind: str) -> Dict[str, Any]:
    pipe = joblib.load(pkl_path)
    out: Dict[str, Any] = {"kind": kind}

    steps = dict(pipe.steps)
    if "imputer" in steps:
        out["imputer_medians"] = _round_list(steps["imputer"].statistics_, 6)
    if "scaler" in steps:
        out["scaler_mean"] = _round_list(steps["scaler"].mean_, 6)
        out["scaler_scale"] = _round_list(steps["scaler"].scale_, 6)

    est = pipe.steps[-1][1]
    if kind == "logistic_regression":
        out["coef"] = _round_list(est.coef_[0], 8)
        out["intercept"] = round(float(est.intercept_[0]), 8)
    elif kind == "decision_tree_classifier":
        out["tree"] = _serialize_tree(est.tree_, is_classifier=True)
    elif kind == "random_forest_classifier":
        out["n_classes"] = int(len(est.classes_))
        out["trees"] = [_serialize_tree(t.tree_, is_classifier=True) for t in est.estimators_]
    elif kind == "random_forest_regressor":
        out["trees"] = [_serialize_tree(t.tree_, is_classifier=False) for t in est.estimators_]
    elif kind == "gradient_boosting_binary":
        n_feat = len(out.get("imputer_medians", []))
        init_raw = float(est._raw_predict_init(np.zeros((1, n_feat)))[0, 0])
        out["learning_rate"] = round(float(est.learning_rate), 8)
        out["init_raw"] = round(init_raw, 8)
        out["trees"] = [_serialize_tree(row[0].tree_, is_classifier=False) for row in est.estimators_]
    else:
        raise ValueError(f"Unsupported export kind: {kind}")

    return out


def _build_analytics_payload() -> Dict[str, Any]:
    df_m = pd.read_csv(DATA_DIR / "maintenance_anonymized.csv")
    df_d = pd.read_csv(DATA_DIR / "dropout_anonymized.csv")
    df_r = pd.read_csv(DATA_DIR / "dropout_reasons_anonymized.csv")
    df_t = pd.read_csv(DATA_DIR / "curriculum_tracer_anonymized.csv")

    with open(META_DIR / "maintenance_metadata.json", "r", encoding="utf-8") as f:
        meta_m = json.load(f)
    with open(META_DIR / "dropout_metadata.json", "r", encoding="utf-8") as f:
        meta_d = json.load(f)
    with open(META_DIR / "curriculum_metadata.json", "r", encoding="utf-8") as f:
        meta_c = json.load(f)

    # --- Pillar 1: Maintenance analytics ---
    df_m["tahun_servis"] = pd.to_datetime(df_m["tanggal_servis"], errors="coerce").dt.year
    maint_by_type = (
        df_m.groupby("jenis_mesin", as_index=False)
        .agg(
            total_logs=("asset_id", "count"),
            unique_assets=("asset_id", "nunique"),
            baik=("kondisi_3_kelas", lambda s: int((s == 0).sum())),
            rusak_ringan=("kondisi_3_kelas", lambda s: int((s == 1).sum())),
            rusak_berat=("kondisi_3_kelas", lambda s: int((s == 2).sum())),
            avg_usia_hari=("usia_mesin_hari", "mean"),
            avg_akumulasi_jp=("akumulasi_jp", "mean"),
        )
        .sort_values("total_logs", ascending=False)
    )
    maint_by_type["avg_usia_hari"] = maint_by_type["avg_usia_hari"].round(1)
    maint_by_type["avg_akumulasi_jp"] = maint_by_type["avg_akumulasi_jp"].round(1)

    maint_by_year = (
        df_m.dropna(subset=["tahun_servis"])
        .groupby("tahun_servis", as_index=False)
        .agg(
            total=("asset_id", "count"),
            baik=("kondisi_3_kelas", lambda s: int((s == 0).sum())),
            waspada=("kondisi_3_kelas", lambda s: int((s == 1).sum())),
            bahaya=("kondisi_3_kelas", lambda s: int((s == 2).sum())),
        )
        .sort_values("tahun_servis")
    )
    maint_by_year["tahun_servis"] = maint_by_year["tahun_servis"].astype(int)

    maint_actions = (
        df_m["tindakan_servis"].value_counts().head(8).reset_index()
    )
    maint_actions.columns = ["tindakan", "jumlah"]

    maint_scatter = df_m[
        [
            "asset_id",
            "jenis_mesin",
            "usia_mesin_hari",
            "akumulasi_jp",
            "riwayat_servis_ke",
            "rasio_beban_harian",
            "kondisi_3_kelas",
            "kondisi",
            "tindakan_servis",
        ]
    ].to_dict(orient="records")

    # Feature importance from 3-class RF
    pipe_m3 = joblib.load(MODELS_DIR / "maintenance_classifier_3class.pkl")
    rf_m3 = pipe_m3.steps[-1][1]
    maint_fi = [
        {"feature": f, "importance": round(float(imp), 4)}
        for f, imp in sorted(
            zip(meta_m["features_classification"], rf_m3.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )[:8]
    ]

    # --- Pillar 2: Drop-Out analytics ---
    do_by_prog = (
        df_d.groupby("jenis_program", as_index=False)
        .agg(
            total=("student_id", "count"),
            dropout=("status_dropout", "sum"),
            avg_attendance=("kehadiran_persen", "mean"),
        )
    )
    do_by_prog["dropout"] = do_by_prog["dropout"].astype(int)
    do_by_prog["selesai"] = do_by_prog["total"] - do_by_prog["dropout"]
    do_by_prog["dropout_rate"] = (do_by_prog["dropout"] / do_by_prog["total"] * 100).round(2)
    do_by_prog["avg_attendance"] = do_by_prog["avg_attendance"].round(1)
    do_by_prog = do_by_prog.sort_values("dropout_rate", ascending=False)

    do_by_zone = (
        df_d.groupby("zona", as_index=False)
        .agg(
            total=("student_id", "count"),
            dropout=("status_dropout", "sum"),
            avg_jarak_km=("jarak_km", "mean"),
        )
        .sort_values("total", ascending=False)
    )
    do_by_zone["dropout"] = do_by_zone["dropout"].astype(int)
    do_by_zone["dropout_rate"] = (do_by_zone["dropout"] / do_by_zone["total"] * 100).round(2)
    do_by_zone["avg_jarak_km"] = do_by_zone["avg_jarak_km"].round(1)

    # Attendance histogram buckets for Selesai vs Drop-Out
    bins = [0, 50, 60, 70, 80, 90, 101]
    labels = ["<50%", "50–59%", "60–69%", "70–79%", "80–89%", "90–100%"]
    df_d["att_bucket"] = pd.cut(df_d["kehadiran_persen"], bins=bins, labels=labels, right=False)
    att_dist = []
    for lbl in labels:
        sub = df_d[df_d["att_bucket"] == lbl]
        att_dist.append(
            {
                "bucket": lbl,
                "selesai": int((sub["status_dropout"] == 0).sum()),
                "dropout": int((sub["status_dropout"] == 1).sum()),
            }
        )

    reasons_pareto = (
        df_r["alasan_keluar"].value_counts().reset_index()
    )
    reasons_pareto.columns = ["alasan", "jumlah"]
    reasons_pareto["persen"] = (reasons_pareto["jumlah"] / reasons_pareto["jumlah"].sum() * 100).round(1)

    # Feature importance from Stage 2 GradientBoosting (using anonymized feature names)
    pipe_d2 = joblib.load(MODELS_DIR / "dropout_stage2_ongoing.pkl")
    gb_d2 = pipe_d2.steps[-1][1]
    anon_d2_names = [
        "paket_jp_diambil",
        "bulan_mulai",
        "jenis_kelamin",
        "usia_masuk",
        "pendidikan",
        "jarak_km",
        "program_PKK",
        "program_PKW",
        "program_REGULER",
        "program_HARIAN",
        "zona_A",
        "zona_B",
        "zona_C",
        "zona_D",
        "zona_E",
        "zona_F",
        "zona_lainnya",
        "kehadiran_persen",
    ]
    dropout_fi = [
        {"feature": f, "importance": round(float(imp), 4)}
        for f, imp in sorted(
            zip(anon_d2_names, gb_d2.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )[:8]
    ]

    # --- Pillar 3: Curriculum & Tracer Study analytics ---
    curr_status = (
        df_t.groupby("status_bekerja", as_index=False)
        .agg(
            count=("alumni_id", "count"),
            ukur_pola=("nilai_ukur_pola", "mean"),
            jahit_dasar=("nilai_jahit_dasar", "mean"),
            jahit_lanjut=("nilai_jahit_lanjut", "mean"),
            finishing=("nilai_finishing", "mean"),
            uji_kompetensi=("skor_uji_kompetensi", "mean"),
            waktu_tunggu_bulan=("waktu_tunggu_bulan", "mean"),
        )
    )
    for col in ["ukur_pola", "jahit_dasar", "jahit_lanjut", "finishing", "uji_kompetensi", "waktu_tunggu_bulan"]:
        curr_status[col] = curr_status[col].round(2)
    curr_status["persen"] = (curr_status["count"] / len(df_t) * 100).round(1)

    curr_scatter = df_t[
        ["alumni_id", "skor_uji_kompetensi", "nilai_finishing", "nilai_ukur_pola", "waktu_tunggu_bulan", "status_bekerja"]
    ].to_dict(orient="records")

    # --- Executive KPIs ---
    total_alumni = int(len(df_t))
    absorbed_alumni = int((df_t["status_bekerja"] != "Belum Bekerja").sum())
    kpis = {
        "unique_machines": int(df_m["asset_id"].nunique()),
        "maintenance_logs": int(len(df_m)),
        "machines_at_risk_pct": round(float((df_m["kondisi_3_kelas"] >= 1).mean() * 100), 1),
        "total_students": int(len(df_d)),
        "dropout_count": int(df_d["status_dropout"].sum()),
        "dropout_rate_pct": round(float(df_d["status_dropout"].mean() * 100), 2),
        "avg_att_completed": round(float(df_d.loc[df_d["status_dropout"] == 0, "kehadiran_persen"].mean()), 2),
        "avg_att_dropout": round(float(df_d.loc[df_d["status_dropout"] == 1, "kehadiran_persen"].mean()), 2),
        "total_alumni": total_alumni,
        "absorption_rate_pct": round(absorbed_alumni / total_alumni * 100, 2),
        "employed_count": int((df_t["status_bekerja"] == "Bekerja").sum()),
        "entrepreneur_count": int((df_t["status_bekerja"] == "Buka Usaha").sum()),
        "unemployed_count": int((df_t["status_bekerja"] == "Belum Bekerja").sum()),
        "total_models": 7,
    }

    return {
        "kpis": kpis,
        "maintenance": {
            "by_type": maint_by_type.to_dict(orient="records"),
            "by_year": maint_by_year.to_dict(orient="records"),
            "actions": maint_actions.to_dict(orient="records"),
            "scatter": maint_scatter,
            "feature_importance": maint_fi,
            "metadata": {
                "target_distribution_3class": meta_m["target_distribution_3class"],
                "classification_stage1_baseline": meta_m["classification_stage1_baseline"],
                "classification_stage2_optimized": meta_m["classification_stage2_optimized"],
                "regression_evaluation": meta_m["regression_evaluation"],
            },
        },
        "dropout": {
            "by_program": do_by_prog.to_dict(orient="records"),
            "by_zone": do_by_zone.to_dict(orient="records"),
            "attendance_distribution": att_dist,
            "reasons_pareto": reasons_pareto.to_dict(orient="records"),
            "feature_importance": dropout_fi,
            "metadata": {
                "threshold_pre": meta_d["optimal_decision_threshold_pre"],
                "threshold_ongoing": meta_d["optimal_decision_threshold_ongoing"],
                "performance_summary": meta_d["performance_summary"],
            },
        },
        "curriculum": {
            "by_status": curr_status.to_dict(orient="records"),
            "correlations": meta_c["korelasi_signifikansi"],
            "scatter": curr_scatter,
            "feature_importance_skema_a": [
                {"feature": k, "importance": v}
                for k, v in meta_c["evaluasi_skema_A_4modul"]["feature_importance"].items()
            ],
            "feature_importance_skema_b": [
                {"feature": k, "importance": v}
                for k, v in meta_c["evaluasi_skema_B_komprehensif"]["feature_importance"].items()
            ],
            "metadata": {
                "skema_a": meta_c["evaluasi_skema_A_4modul"],
                "skema_b": meta_c["evaluasi_skema_B_komprehensif"],
                "imputer_train_medians": meta_c["imputer_train_medians"],
            },
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    models_payload = {
        "maintenance_classifier_3class": _export_pipeline(
            MODELS_DIR / "maintenance_classifier_3class.pkl", "random_forest_classifier"
        ),
        "maintenance_classifier_4class": _export_pipeline(
            MODELS_DIR / "maintenance_classifier_4class.pkl", "random_forest_classifier"
        ),
        "maintenance_regressor": _export_pipeline(
            MODELS_DIR / "maintenance_regressor.pkl", "random_forest_regressor"
        ),
        "dropout_stage1_pre": _export_pipeline(
            MODELS_DIR / "dropout_stage1_pre.pkl", "logistic_regression"
        ),
        "dropout_stage2_ongoing": _export_pipeline(
            MODELS_DIR / "dropout_stage2_ongoing.pkl", "gradient_boosting_binary"
        ),
        "curriculum_dt_skema_a": _export_pipeline(
            MODELS_DIR / "curriculum_dt_skema_a.pkl", "decision_tree_classifier"
        ),
        "curriculum_dt_skema_b": _export_pipeline(
            MODELS_DIR / "curriculum_dt_skema_b.pkl", "decision_tree_classifier"
        ),
    }

    bundle = {
        "version": "1.0.0-local-github-pages",
        "institution": "Lembaga Pelatihan Vokasi (Studi Kasus Teranonimisasi)",
        "analytics": _build_analytics_payload(),
        "models": models_payload,
    }

    json_path = OUT_DIR / "dashboard_bundle.json"
    js_path = OUT_DIR / "dashboard_bundle.js"

    json_str = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))
    json_path.write_text(json_str, encoding="utf-8")
    js_path.write_text(f"window.PORTFOLIO_BUNDLE = {json_str};\n", encoding="utf-8")

    size_kb = round(json_path.stat().st_size / 1024, 1)
    print(f"SUCCESS: Exported local analytics & 7 ML pipelines -> {json_path.name} ({size_kb} KB)")


if __name__ == "__main__":
    main()
