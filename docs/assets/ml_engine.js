/**
 * Client-Side ML Inference Engine (100% Local — Zero Backend Required).
 *
 * Executes all 7 serialized scikit-learn pipelines directly in the browser
 * (or Node.js for automated parity testing against Python `src/inference.py`).
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.VocationalMLEngine = factory();
  }
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const MACHINE_TYPES = [
    "BUTTERFLY_MANUAL",
    "HIGH_SPEED",
    "MESIN_JAHIT_LAINNYA",
    "NECI",
    "OBRAS",
    "OVERDECK",
    "PORTABLE",
    "SINGER",
  ];

  const PROGRAMS = ["PKK", "PKW", "REGULER", "HARIAN"];
  const PROGRAM_DEFAULT_JP = { PKK: 200, PKW: 200, REGULER: 150, HARIAN: 51 };
  const ZONES = ["Zone A", "Zone B", "Zone C", "Zone D", "Zone E", "Zone F", "Other Zones"];
  const EDUCATION_MAP = {
    "SD/Sederajat": 0,
    "SMP/Sederajat": 1,
    "SMA/SMK/Sederajat": 2,
    "Perguruan Tinggi": 3,
  };
  const GENDER_MAP = { P: 0, L: 1 };

  function round(val, digits) {
    const f = Math.pow(10, digits);
    return Math.round(val * f) / f;
  }

  function sigmoid(z) {
    if (z >= 0) {
      const ez = Math.exp(-z);
      return 1 / (1 + ez);
    }
    const ez = Math.exp(z);
    return ez / (1 + ez);
  }

  function applyPreprocess(xRaw, modelSpec) {
    const x = xRaw.slice();
    if (modelSpec.imputer_medians) {
      for (let i = 0; i < x.length; i++) {
        if (x[i] === null || x[i] === undefined || Number.isNaN(x[i])) {
          x[i] = modelSpec.imputer_medians[i];
        }
      }
    }
    if (modelSpec.scaler_mean && modelSpec.scaler_scale) {
      for (let i = 0; i < x.length; i++) {
        x[i] = (x[i] - modelSpec.scaler_mean[i]) / modelSpec.scaler_scale[i];
      }
    }
    return x;
  }

  function evalTree(tree, x) {
    let node = 0;
    while (tree.l[node] !== -1) {
      const feat = tree.f[node];
      const thr = tree.t[node];
      node = x[feat] <= thr ? tree.l[node] : tree.r[node];
    }
    return tree.v[node];
  }

  function predictProba(modelSpec, xRaw) {
    const x = applyPreprocess(xRaw, modelSpec);

    if (modelSpec.kind === "logistic_regression") {
      let z = modelSpec.intercept;
      for (let i = 0; i < x.length; i++) {
        z += modelSpec.coef[i] * x[i];
      }
      const p1 = sigmoid(z);
      return [1 - p1, p1];
    }

    if (modelSpec.kind === "decision_tree_classifier") {
      return evalTree(modelSpec.tree, x).slice();
    }

    if (modelSpec.kind === "random_forest_classifier") {
      const nClasses = modelSpec.n_classes;
      const sums = new Array(nClasses).fill(0);
      const nTrees = modelSpec.trees.length;
      for (let t = 0; t < nTrees; t++) {
        const probs = evalTree(modelSpec.trees[t], x);
        for (let c = 0; c < nClasses; c++) {
          sums[c] += probs[c];
        }
      }
      return sums.map((s) => s / nTrees);
    }

    if (modelSpec.kind === "gradient_boosting_binary") {
      let raw = modelSpec.init_raw;
      const lr = modelSpec.learning_rate;
      for (let t = 0; t < modelSpec.trees.length; t++) {
        raw += lr * evalTree(modelSpec.trees[t], x);
      }
      const p1 = sigmoid(raw);
      return [1 - p1, p1];
    }

    throw new Error("Unsupported classifier kind: " + modelSpec.kind);
  }

  function predictRegressor(modelSpec, xRaw) {
    const x = applyPreprocess(xRaw, modelSpec);
    let total = 0;
    for (let t = 0; t < modelSpec.trees.length; t++) {
      total += evalTree(modelSpec.trees[t], x);
    }
    return total / modelSpec.trees.length;
  }

  function argmax(arr) {
    let best = 0;
    for (let i = 1; i < arr.length; i++) {
      if (arr[i] > arr[best]) best = i;
    }
    return best;
  }

  function createEngine(models) {
    return {
      predictMaintenance({ usia_mesin_hari, akumulasi_jp, riwayat_servis_ke, jenis_mesin }) {
        const rasio = akumulasi_jp / (usia_mesin_hari + 1.0);
        const jpPerServis = akumulasi_jp / Math.max(Math.round(riwayat_servis_ke), 1);
        const oneHot = MACHINE_TYPES.map((mt) => (mt === jenis_mesin ? 1.0 : 0.0));

        const xClf = [usia_mesin_hari, akumulasi_jp, riwayat_servis_ke, rasio, jpPerServis, ...oneHot];
        const proba3 = predictProba(models.maintenance_classifier_3class, xClf);
        const pred3 = argmax(proba3);

        const proba4 = predictProba(models.maintenance_classifier_4class, xClf);
        const pred4 = argmax(proba4);

        const xReg = [usia_mesin_hari, riwayat_servis_ke, ...oneHot];
        const estJp = predictRegressor(models.maintenance_regressor, xReg);

        const labels3 = ["Baik (Normal)", "Rusak Ringan (Waspada)", "Rusak Berat / Afkir (Bahaya)"];
        const labels4 = ["Baik", "Rusak Ringan", "Rusak Berat", "Rusak Total"];
        const actions3 = [
          "Pertahankan perawatan rutin (pembersihan debu & pelumasan berkala).",
          "Jadwalkan inspeksi teknisi & servis preventif sebelum kelas intensif berikutnya.",
          "Hentikan pemakaian intensif; lakukan overhaul komponen atau pertimbangkan peremajaan aset.",
        ];

        return {
          derived_features: {
            rasio_beban_harian: round(rasio, 4),
            jp_per_servis: round(jpPerServis, 2),
          },
          classification_3class: {
            code: pred3,
            label: labels3[pred3],
            probabilities: {
              Baik: round(proba3[0], 4),
              "Rusak Ringan": round(proba3[1], 4),
              "Rusak Berat / Afkir": round(proba3[2], 4),
            },
          },
          classification_4class: {
            code: pred4,
            label: labels4[pred4],
          },
          estimated_akumulasi_jp: round(estJp, 2),
          recommendation: actions3[pred3],
        };
      },

      predictDropout({
        jenis_program,
        bulan_mulai,
        jenis_kelamin,
        usia_masuk,
        pendidikan,
        jarak_km,
        zona,
        kehadiran_persen = null,
        paket_jp_diambil = null,
      }) {
        const jp = paket_jp_diambil !== null ? paket_jp_diambil : PROGRAM_DEFAULT_JP[jenis_program];
        const progOH = PROGRAMS.map((p) => (p === jenis_program ? 1.0 : 0.0));
        const zoneOH = ZONES.map((z) => (z === zona ? 1.0 : 0.0));

        const base17 = [
          jp,
          bulan_mulai,
          GENDER_MAP[jenis_kelamin],
          usia_masuk,
          EDUCATION_MAP[pendidikan],
          jarak_km,
          ...progOH,
          ...zoneOH,
        ];

        const probPre = predictProba(models.dropout_stage1_pre, base17)[1];
        const thrPre = 0.25;
        const predPre = probPre >= thrPre ? 1 : 0;

        const out = {
          stage1_pre_enrollment: {
            probability_dropout: round(probPre, 4),
            threshold: thrPre,
            predicted_code: predPre,
            predicted_label: predPre === 1 ? "KELUAR (Berisiko Drop-Out)" : "SELESAI (Risiko Rendah)",
          },
        };

        if (kehadiran_persen !== null && kehadiran_persen !== undefined && !Number.isNaN(kehadiran_persen)) {
          const base18 = [...base17, kehadiran_persen];
          const probOng = predictProba(models.dropout_stage2_ongoing, base18)[1];
          const thrOng = 0.5;
          const predOng = probOng >= thrOng ? 1 : 0;
          out.stage2_ongoing_monitoring = {
            probability_dropout: round(probOng, 4),
            threshold: thrOng,
            predicted_code: predOng,
            predicted_label: predOng === 1 ? "KELUAR (Berisiko Drop-Out)" : "SELESAI (Aman)",
          };
          if (predOng === 1) {
            out.recommendation =
              "Intervensi segera: lakukan konseling personal, evaluasi kendala kehadiran/transportasi, dan tawarkan fleksibilitas jadwal praktik.";
          } else if (predPre === 1) {
            out.recommendation =
              "Status saat ini terkendali, namun profil awal berisiko; pantau presensi mingguan secara berkala.";
          } else {
            out.recommendation = "Peserta berada pada jalur penyelesaian yang stabil; pertahankan monitoring standar.";
          }
        } else {
          out.recommendation =
            predPre === 1
              ? "Berikan orientasi intensif & check-in minggu pertama karena profil awal masuk kategori risiko."
              : "Profil awal stabil; ikuti alur orientasi standar.";
        }

        return out;
      },

      predictCurriculum({
        nilai_ukur_pola,
        nilai_jahit_dasar,
        nilai_jahit_lanjut,
        nilai_finishing,
        skor_uji_kompetensi = null,
      }) {
        const xA = [nilai_ukur_pola, nilai_jahit_dasar, nilai_jahit_lanjut, nilai_finishing];
        const probaA = predictProba(models.curriculum_dt_skema_a, xA);
        const predA = argmax(probaA);

        const xB = [...xA, skor_uji_kompetensi];
        const probaB = predictProba(models.curriculum_dt_skema_b, xB);
        const predB = argmax(probaB);

        const labels = [
          "Bekerja (Terserap Industri)",
          "Buka Usaha (Wirausaha Mandiri)",
          "Belum Bekerja (Perlu Pendampingan)",
        ];
        const actions = [
          "Arahkan ke kanal penyaluran kerja mitra garmen/konveksi dan perkuat portofolio kualitas jahitan.",
          "Berikan pendampingan bisnis mikro, penetapan harga jasa, dan pemasaran digital untuk usaha mandiri.",
          "Prioritaskan remedial pada modul Finishing & Ukur Pola sebelum uji kompetensi ulang.",
        ];

        const primaryCode = skor_uji_kompetensi !== null && !Number.isNaN(skor_uji_kompetensi) ? predB : predA;

        return {
          skema_a_4modul: {
            code: predA,
            label: labels[predA],
            probabilities: {
              Bekerja: round(probaA[0], 4),
              "Buka Usaha": round(probaA[1], 4),
              "Belum Bekerja": round(probaA[2], 4),
            },
          },
          skema_b_komprehensif: {
            code: predB,
            label: labels[predB],
            probabilities: {
              Bekerja: round(probaB[0], 4),
              "Buka Usaha": round(probaB[1], 4),
              "Belum Bekerja": round(probaB[2], 4),
            },
          },
          recommendation: actions[primaryCode],
        };
      },
    };
  }

  return {
    MACHINE_TYPES,
    PROGRAMS,
    PROGRAM_DEFAULT_JP,
    ZONES,
    createEngine,
  };
});
