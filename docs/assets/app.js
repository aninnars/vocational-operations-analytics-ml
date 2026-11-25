/**
 * Main Application Controller for Vocational Operations Modern Analytics Dashboard.
 * Consumes 100% local `window.PORTFOLIO_BUNDLE` (data + 7 exported ML pipelines).
 */
(function () {
  "use strict";

  const BUNDLE = window.PORTFOLIO_BUNDLE;
  if (!BUNDLE) {
    document.body.innerHTML =
      '<div style="padding:40px;color:#f43f5e;font-family:sans-serif;">Gagal memuat data lokal (docs/data/dashboard_bundle.js). Jalankan <code>python src/export_web_artifacts.py</code>.</div>';
    return;
  }

  const ANALYTICS = BUNDLE.analytics;
  const ENGINE = window.VocationalMLEngine.createEngine(BUNDLE.models);
  const CHARTS = window.ModernCharts;

  const VIEW_META = {
    overview: {
      title: "Ringkasan Eksekutif Operasional",
      sub: "Studi Kasus Teranonimisasi Lembaga Pelatihan Vokasi — Integrasi Business Intelligence & 7 Pipeline ML",
    },
    maintenance: {
      title: "Pilar 1: Predictive Maintenance & IoT Proxy",
      sub: "Estimasi beban jam pelajaran mesin tanpa sensor fisik & klasifikasi prioritas perawatan preventif",
    },
    dropout: {
      title: "Pilar 2: Dual-Stage Early Warning System Drop-Out",
      sub: "Deteksi risiko berhenti sejak Hari-1 pendaftaran hingga pemantauan kehadiran aktif (imbalance 4,01%)",
    },
    curriculum: {
      title: "Pilar 3: Evaluasi Kurikulum & Tracer Study Alumni",
      sub: "Korelasi empiris modul praktik terhadap serapan kerja, wirausaha, dan kecepatan waktu tunggu",
    },
    modellab: {
      title: "Model Lab, Feature Importance, & Arsitektur",
      sub: "Evaluasi saintifik 7 pipeline scikit-learn, transparansi batas peran tim, dan jaminan privasi lokal",
    },
  };

  // -------------------------------------------------------------------------
  // NAVIGATION
  // -------------------------------------------------------------------------
  function switchView(viewId) {
    document.querySelectorAll(".view-panel").forEach((el) => {
      el.classList.toggle("active", el.id === "view-" + viewId);
    });
    document.querySelectorAll("#mainNav .nav-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.view === viewId);
    });
    const meta = VIEW_META[viewId] || VIEW_META.overview;
    document.getElementById("topbarTitle").textContent = meta.title;
    document.getElementById("topbarSubtitle").textContent = meta.sub;
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  document.querySelectorAll("#mainNav .nav-btn").forEach((btn) => {
    btn.addEventListener("click", () => switchView(btn.dataset.view));
  });
  document.querySelectorAll("[data-jump]").forEach((btn) => {
    btn.addEventListener("click", () => switchView(btn.dataset.jump));
  });

  // -------------------------------------------------------------------------
  // VIEW 1: EXECUTIVE OVERVIEW
  // -------------------------------------------------------------------------
  function renderOverview() {
    const k = ANALYTICS.kpis;
    const kpiData = [
      {
        label: "Aset Mesin Dipantau",
        value: `${k.unique_machines} Unit`,
        meta: `${k.maintenance_logs} log servis · ${k.machines_at_risk_pct}% butuh perhatian`,
        color: "#38bdf8",
      },
      {
        label: "Peserta Dianalisis",
        value: `${k.total_students} Siswa`,
        meta: `Tingkat Drop-Out ${k.dropout_rate_pct}% (${k.dropout_count} kasus)`,
        color: "#f59e0b",
      },
      {
        label: "Serapan Karir Alumni",
        value: `${k.absorption_rate_pct}%`,
        meta: `${k.employed_count} Bekerja + ${k.entrepreneur_count} Wirausaha (n=${k.total_alumni})`,
        color: "#10b981",
      },
      {
        label: "Pipeline ML Lokal",
        value: `${k.total_models} Model`,
        meta: "100% client-side parity · 0 API cloud",
        color: "#818cf8",
      },
    ];

    const grid = document.getElementById("execKpiGrid");
    grid.innerHTML = kpiData
      .map(
        (item) => `
      <div class="card kpi-card" style="--kpi-color:${item.color}">
        <div class="kpi-label">${item.label}</div>
        <div class="kpi-value">${item.value}</div>
        <div class="kpi-meta">${item.meta}</div>
      </div>`
      )
      .join("");

    // Maintenance yearly trend
    const yrs = ANALYTICS.maintenance.by_year;
    CHARTS.renderGroupedBar(
      document.getElementById("chartExecMaintTrend"),
      yrs.map((d) => String(d.tahun_servis)),
      [
        { name: "Baik (Normal)", color: "#10b981", values: yrs.map((d) => d.baik) },
        { name: "Rusak Ringan", color: "#f59e0b", values: yrs.map((d) => d.waspada) },
        { name: "Rusak Berat/Afkir", color: "#f43f5e", values: yrs.map((d) => d.bahaya) },
      ],
      { height: 230, valueSuffix: " log" }
    );

    // Alumni career donut
    CHARTS.renderDonut(
      document.getElementById("chartExecAlumniDonut"),
      [
        { label: "Bekerja (Industri)", value: k.employed_count, color: "#10b981" },
        { label: "Buka Usaha (Mandiri)", value: k.entrepreneur_count, color: "#38bdf8" },
        { label: "Belum Bekerja", value: k.unemployed_count, color: "#f43f5e" },
      ],
      `${k.absorption_rate_pct}%`,
      "Terserap"
    );
  }

  // -------------------------------------------------------------------------
  // VIEW 2: PREDICTIVE MAINTENANCE
  // -------------------------------------------------------------------------
  function renderMaintenanceCharts(selectedType) {
    const allPts = ANALYTICS.maintenance.scatter;
    const filtered = selectedType === "ALL" ? allPts : allPts.filter((p) => p.jenis_mesin === selectedType);

    const uniqueCount = new Set(filtered.map((p) => p.asset_id)).size;
    const atRisk = filtered.filter((p) => p.kondisi_3_kelas >= 1).length;
    document.getElementById("maintFilterMeta").textContent =
      `Menampilkan ${filtered.length} log dari ${uniqueCount} unit unik (${atRisk} log berisiko rusak)`;

    const colorMap = { 0: "#10b981", 1: "#f59e0b", 2: "#f43f5e" };
    const lblMap = { 0: "Baik (Normal)", 1: "Rusak Ringan", 2: "Rusak Berat / Afkir" };

    CHARTS.renderScatter(
      document.getElementById("chartMaintScatter"),
      filtered.map((p) => ({
        x: p.usia_mesin_hari,
        y: p.akumulasi_jp,
        color: colorMap[p.kondisi_3_kelas],
        tooltip: `<strong>${p.asset_id}</strong> (${p.jenis_mesin})<br/>Kondisi: <b>${lblMap[p.kondisi_3_kelas]}</b><br/>Usia: ${p.usia_mesin_hari} hari · Beban: ${p.akumulasi_jp} JP<br/>Servis ke-${p.riwayat_servis_ke} · ${p.tindakan_servis}`,
      })),
      {
        xLabel: "Usia Pakai Mesin (Hari)",
        yLabel: "Akumulasi Beban (JP)",
        legend: [
          { label: "Baik (Normal)", color: "#10b981" },
          { label: "Rusak Ringan (Waspada)", color: "#f59e0b" },
          { label: "Rusak Berat / Afkir", color: "#f43f5e" },
        ],
      }
    );

    CHARTS.renderHorizontalBar(
      document.getElementById("chartMaintByType"),
      ANALYTICS.maintenance.by_type.map((d) => ({
        label: d.jenis_mesin,
        value: d.total_logs,
        color: d.jenis_mesin === selectedType ? "#f59e0b" : "#38bdf8",
        tooltip: `${d.total_logs} log (${d.unique_assets} unit) · Baik: ${d.baik}, Ringan: ${d.rusak_ringan}, Berat: ${d.rusak_berat}`,
      })),
      { valueSuffix: " log" }
    );

    CHARTS.renderHorizontalBar(
      document.getElementById("chartMaintActions"),
      ANALYTICS.maintenance.actions.map((d) => ({
        label: d.tindakan,
        value: d.jumlah,
        color: "#818cf8",
        tooltip: `Frekuensi tindakan: ${d.jumlah} kali`,
      })),
      { valueSuffix: "x" }
    );
  }

  function initMaintenance() {
    const filterSel = document.getElementById("maintTypeFilter");
    const simSel = document.getElementById("simMaintType");
    const types = window.VocationalMLEngine.MACHINE_TYPES;

    filterSel.innerHTML =
      `<option value="ALL">Semua Tipe Mesin (8 Kategori)</option>` +
      types.map((t) => `<option value="${t}">${t}</option>`).join("");
    simSel.innerHTML = types.map((t) => `<option value="${t}" ${t === "HIGH_SPEED" ? "selected" : ""}>${t}</option>`).join("");

    filterSel.addEventListener("change", () => renderMaintenanceCharts(filterSel.value));
    renderMaintenanceCharts("ALL");

    function updateSim() {
      const jenis_mesin = simSel.value;
      const usia_mesin_hari = Number(document.getElementById("simMaintUsia").value);
      const akumulasi_jp = Number(document.getElementById("simMaintJp").value);
      const riwayat_servis_ke = Number(document.getElementById("simMaintServis").value);

      document.getElementById("valMaintUsia").textContent = `${usia_mesin_hari} hari`;
      document.getElementById("valMaintJp").textContent = `${akumulasi_jp} JP`;
      document.getElementById("valMaintServis").textContent = `${riwayat_servis_ke} kali`;

      const res = ENGINE.predictMaintenance({ usia_mesin_hari, akumulasi_jp, riwayat_servis_ke, jenis_mesin });
      const statusCls = ["status-good", "status-warn", "status-danger"][res.classification_3class.code];
      const probs = res.classification_3class.probabilities;

      document.getElementById("maintSimOutput").innerHTML = `
        <div class="result-banner ${statusCls}">
          <div class="result-kicker">Prediksi Utama (Random Forest 3-Kelas)</div>
          <div class="result-headline">${res.classification_3class.label}</div>
          <div style="font-size:12px;color:var(--text-secondary);">Pembanding 4-Kelas: <strong>${res.classification_4class.label}</strong></div>
        </div>
        <div class="prob-list">
          ${[
            ["Baik (Normal)", probs["Baik"], "#10b981"],
            ["Rusak Ringan (Waspada)", probs["Rusak Ringan"], "#f59e0b"],
            ["Rusak Berat / Afkir (Bahaya)", probs["Rusak Berat / Afkir"], "#f43f5e"],
          ]
            .map(
              ([lbl, p, clr]) => `
            <div class="prob-item">
              <div class="prob-head"><span>${lbl}</span><strong>${(p * 100).toFixed(1)}%</strong></div>
              <div class="prob-track"><div class="prob-fill" style="width:${p * 100}%;background:${clr};"></div></div>
            </div>`
            )
            .join("")}
        </div>
        <div class="metric-strip">
          <div class="mini-metric"><div class="lbl">Estimasi Beban (Regresor)</div><div class="num">${res.estimated_akumulasi_jp} JP</div></div>
          <div class="mini-metric"><div class="lbl">Rasio Beban Harian</div><div class="num">${res.derived_features.rasio_beban_harian} JP/hr</div></div>
          <div class="mini-metric"><div class="lbl">JP per Servis</div><div class="num">${res.derived_features.jp_per_servis} JP</div></div>
        </div>
        <div class="action-box">
          <strong>Rekomendasi Aksi Teknisi</strong>
          ${res.recommendation}
        </div>`;
    }

    ["simMaintType", "simMaintUsia", "simMaintJp", "simMaintServis"].forEach((id) => {
      document.getElementById(id).addEventListener("input", updateSim);
    });

    const presets = {
      normal: { type: "PORTABLE", usia: 180, jp: 650, servis: 1 },
      waspada: { type: "HIGH_SPEED", usia: 500, jp: 2200, servis: 1 },
      bahaya: { type: "BUTTERFLY_MANUAL", usia: 1850, jp: 5800, servis: 5 },
    };
    document.querySelectorAll("[data-mpreset]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const p = presets[btn.dataset.mpreset];
        simSel.value = p.type;
        document.getElementById("simMaintUsia").value = p.usia;
        document.getElementById("simMaintJp").value = p.jp;
        document.getElementById("simMaintServis").value = p.servis;
        updateSim();
      });
    });

    updateSim();
  }

  // -------------------------------------------------------------------------
  // VIEW 3: DROP-OUT EWS
  // -------------------------------------------------------------------------
  function initDropout() {
    const att = ANALYTICS.dropout.attendance_distribution;
    CHARTS.renderGroupedBar(
      document.getElementById("chartDoAttendance"),
      att.map((d) => d.bucket),
      [
        { name: "Selesai", color: "#10b981", values: att.map((d) => d.selesai) },
        { name: "Drop-Out", color: "#f43f5e", values: att.map((d) => d.dropout) },
      ],
      { height: 225, valueSuffix: " siswa" }
    );

    CHARTS.renderHorizontalBar(
      document.getElementById("chartDoProgram"),
      ANALYTICS.dropout.by_program.map((d) => ({
        label: `${d.jenis_program} (n=${d.total})`,
        value: d.dropout_rate,
        color: d.dropout_rate >= 6 ? "#f43f5e" : d.dropout_rate >= 4 ? "#f59e0b" : "#10b981",
        tooltip: `${d.dropout} dari ${d.total} peserta keluar (${d.dropout_rate}%) · Rata-rata hadir ${d.avg_attendance}%`,
      })),
      { valueSuffix: "%", maxVal: 10 }
    );

    CHARTS.renderHorizontalBar(
      document.getElementById("chartDoReasons"),
      ANALYTICS.dropout.reasons_pareto.map((d) => ({
        label: d.alasan,
        value: d.jumlah,
        color: "#f59e0b",
        tooltip: `${d.jumlah} kasus (${d.persen}% dari total drop-out)`,
      })),
      { valueSuffix: " kasus" }
    );

    CHARTS.renderHorizontalBar(
      document.getElementById("chartDoZone"),
      ANALYTICS.dropout.by_zone.map((d) => ({
        label: d.zona,
        value: d.total,
        color: "#38bdf8",
        tooltip: `${d.total} peserta · Drop-out: ${d.dropout} (${d.dropout_rate}%) · Rata-rata jarak ${d.avg_jarak_km} km`,
      })),
      { valueSuffix: " siswa" }
    );

    function updateSim() {
      const jenis_program = document.getElementById("simDoProgram").value;
      const zona = document.getElementById("simDoZone").value;
      const pendidikan = document.getElementById("simDoEdu").value;
      const jenis_kelamin = document.getElementById("simDoGender").value;
      const usia_masuk = Number(document.getElementById("simDoUsia").value);
      const bulan_mulai = Number(document.getElementById("simDoBulan").value);
      const jarak_km = Number(document.getElementById("simDoJarak").value);
      const kehadiran_persen = Number(document.getElementById("simDoAtt").value);

      document.getElementById("valDoUsia").textContent = `${usia_masuk} th`;
      document.getElementById("valDoBulan").textContent = `Bulan ${bulan_mulai}`;
      document.getElementById("valDoJarak").textContent = `${jarak_km.toFixed(1)} km`;
      document.getElementById("valDoAtt").textContent = `${kehadiran_persen}%`;

      const res = ENGINE.predictDropout({
        jenis_program,
        bulan_mulai,
        jenis_kelamin,
        usia_masuk,
        pendidikan,
        jarak_km,
        zona,
        kehadiran_persen,
      });

      const s1 = res.stage1_pre_enrollment;
      const s2 = res.stage2_ongoing_monitoring;

      CHARTS.renderRiskGauge(
        document.getElementById("gaugeDoStage1"),
        s1.probability_dropout,
        s1.threshold,
        "Stage 1 · Skrining Hari-1 (LogReg)"
      );
      CHARTS.renderRiskGauge(
        document.getElementById("gaugeDoStage2"),
        s2.probability_dropout,
        s2.threshold,
        "Stage 2 · Monitoring Berjalan (GradBoost)"
      );

      document.getElementById("doSimOutput").innerHTML = `
        <div class="metric-strip" style="grid-template-columns:repeat(2,minmax(0,1fr));">
          <div class="mini-metric">
            <div class="lbl">Status Stage 1 (Pra-Daftar)</div>
            <div class="num" style="font-size:13.5px;color:${s1.predicted_code ? "#f43f5e" : "#10b981"};">${s1.predicted_label}</div>
          </div>
          <div class="mini-metric">
            <div class="lbl">Status Stage 2 (Berjalan)</div>
            <div class="num" style="font-size:13.5px;color:${s2.predicted_code ? "#f43f5e" : "#10b981"};">${s2.predicted_label}</div>
          </div>
        </div>
        <div class="action-box">
          <strong>Rekomendasi Intervensi Akademik</strong>
          ${res.recommendation}
        </div>`;
    }

    [
      "simDoProgram",
      "simDoZone",
      "simDoEdu",
      "simDoGender",
      "simDoUsia",
      "simDoBulan",
      "simDoJarak",
      "simDoAtt",
    ].forEach((id) => document.getElementById(id).addEventListener("input", updateSim));

    const presets = {
      aman: { prog: "PKK", zone: "Zone A", edu: "SMA/SMK/Sederajat", jk: "P", usia: 22, bln: 8, jarak: 4, att: 96 },
      ref: { prog: "REGULER", zone: "Zone A", edu: "SMA/SMK/Sederajat", jk: "P", usia: 26, bln: 3, jarak: 7, att: 62 },
      waspada: { prog: "PKW", zone: "Zone B", edu: "SMP/Sederajat", jk: "P", usia: 28, bln: 4, jarak: 15, att: 94 },
    };
    document.querySelectorAll("[data-dpreset]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const p = presets[btn.dataset.dpreset];
        document.getElementById("simDoProgram").value = p.prog;
        document.getElementById("simDoZone").value = p.zone;
        document.getElementById("simDoEdu").value = p.edu;
        document.getElementById("simDoGender").value = p.jk;
        document.getElementById("simDoUsia").value = p.usia;
        document.getElementById("simDoBulan").value = p.bln;
        document.getElementById("simDoJarak").value = p.jarak;
        document.getElementById("simDoAtt").value = p.att;
        updateSim();
      });
    });

    updateSim();
  }

  // -------------------------------------------------------------------------
  // VIEW 4: CURRICULUM & TRACER STUDY
  // -------------------------------------------------------------------------
  function initCurriculum() {
    const byStatus = {};
    ANALYTICS.curriculum.by_status.forEach((r) => (byStatus[r.status_bekerja] = r));
    const modules = [
      ["Ukur Pola", "ukur_pola"],
      ["Jahit Dasar", "jahit_dasar"],
      ["Jahit Lanjut", "jahit_lanjut"],
      ["Finishing", "finishing"],
      ["Uji Komp.", "uji_kompetensi"],
    ];

    CHARTS.renderGroupedBar(
      document.getElementById("chartCurrProfile"),
      modules.map((m) => m[0]),
      [
        { name: "Bekerja (4,54 bln)", color: "#10b981", values: modules.map((m) => byStatus["Bekerja"][m[1]]) },
        { name: "Buka Usaha (5,49 bln)", color: "#38bdf8", values: modules.map((m) => byStatus["Buka Usaha"][m[1]]) },
        { name: "Belum Bekerja (18,15 bln)", color: "#f43f5e", values: modules.map((m) => byStatus["Belum Bekerja"][m[1]]) },
      ],
      { height: 235, maxVal: 100, valueSuffix: " pts" }
    );

    const corrs = ANALYTICS.curriculum.correlations;
    const corrItems = [];
    corrs.forEach((c) => {
      corrItems.push({
        label: `${c.Modul} (Serapan)`,
        value: c["r_Terserap (All)"],
        displayValue: `+${c["r_Terserap (All)"].toFixed(3)}`,
        color: "#10b981",
        tooltip: `Pearson r terhadap serapan karir: <b>+${c["r_Terserap (All)"]}</b> (p = ${c.p_r_Terserap})`,
      });
      corrItems.push({
        label: `${c.Modul} (Tunggu)`,
        value: c["r_WaktuTunggu (Emp)"],
        displayValue: c["r_WaktuTunggu (Emp)"].toFixed(3),
        color: c["r_WaktuTunggu (Emp)"] < -0.2 ? "#38bdf8" : "#64748b",
        tooltip: `Pearson r terhadap waktu tunggu kerja: <b>${c["r_WaktuTunggu (Emp)"]}</b> (nilai negatif = semakin cepat dapat kerja)`,
      });
    });
    CHARTS.renderHorizontalBar(document.getElementById("chartCurrCorr"), corrItems, {
      diverging: true,
      maxVal: 1.0,
    });

    const statusColor = { Bekerja: "#10b981", "Buka Usaha": "#38bdf8", "Belum Bekerja": "#f43f5e" };
    CHARTS.renderScatter(
      document.getElementById("chartCurrScatter"),
      ANALYTICS.curriculum.scatter.map((p) => ({
        x: p.skor_uji_kompetensi,
        y: p.waktu_tunggu_bulan,
        color: statusColor[p.status_bekerja],
        tooltip: `<strong>${p.alumni_id}</strong> · <b>${p.status_bekerja}</b><br/>Skor Uji Komp: ${p.skor_uji_kompetensi} · Finishing: ${p.nilai_finishing}<br/>Waktu Tunggu: ${p.waktu_tunggu_bulan} bulan`,
      })),
      {
        minX: 60,
        minY: 0,
        xLabel: "Skor Uji Kompetensi",
        yLabel: "Waktu Tunggu (Bulan)",
        legend: [
          { label: "Bekerja (Industri)", color: "#10b981" },
          { label: "Buka Usaha (Wirausaha)", color: "#38bdf8" },
          { label: "Belum Bekerja", color: "#f43f5e" },
        ],
      }
    );

    const nullChk = document.getElementById("simCurrLanjutNull");
    const lanjutSlider = document.getElementById("simCurrLanjut");

    function updateSim() {
      const pola = Number(document.getElementById("simCurrPola").value);
      const dasar = Number(document.getElementById("simCurrDasar").value);
      const isNull = nullChk.checked;
      lanjutSlider.disabled = isNull;
      const lanjut = isNull ? null : Number(lanjutSlider.value);
      const finish = Number(document.getElementById("simCurrFinish").value);
      const uji = Number(document.getElementById("simCurrUji").value);

      document.getElementById("valCurrPola").textContent = pola.toFixed(1);
      document.getElementById("valCurrDasar").textContent = dasar.toFixed(1);
      document.getElementById("valCurrLanjut").textContent = isNull ? "Missing (Imputer → 82.75)" : lanjut.toFixed(1);
      document.getElementById("valCurrFinish").textContent = finish.toFixed(1);
      document.getElementById("valCurrUji").textContent = uji.toFixed(1);

      const res = ENGINE.predictCurriculum({
        nilai_ukur_pola: pola,
        nilai_jahit_dasar: dasar,
        nilai_jahit_lanjut: lanjut,
        nilai_finishing: finish,
        skor_uji_kompetensi: uji,
      });

      const clsMap = { 0: "status-good", 1: "status-warn", 2: "status-danger" };
      const a = res.skema_a_4modul;
      const b = res.skema_b_komprehensif;

      document.getElementById("currSimOutput").innerHTML = `
        <div class="grid-2" style="margin-bottom:12px;">
          <div class="result-banner ${clsMap[a.code]}" style="margin:0;">
            <div class="result-kicker">Skema A (4 Modul Praktik)</div>
            <div class="result-headline" style="font-size:17px;">${a.label}</div>
            <div style="font-size:11.5px;color:var(--text-secondary);font-family:var(--font-mono);">
              Bekerja ${(a.probabilities["Bekerja"] * 100).toFixed(0)}% · Usaha ${(a.probabilities["Buka Usaha"] * 100).toFixed(0)}% · Belum ${(a.probabilities["Belum Bekerja"] * 100).toFixed(0)}%
            </div>
          </div>
          <div class="result-banner ${clsMap[b.code]}" style="margin:0;">
            <div class="result-kicker">Skema B (4 Modul + Uji Komp.)</div>
            <div class="result-headline" style="font-size:17px;">${b.label}</div>
            <div style="font-size:11.5px;color:var(--text-secondary);font-family:var(--font-mono);">
              Bekerja ${(b.probabilities["Bekerja"] * 100).toFixed(0)}% · Usaha ${(b.probabilities["Buka Usaha"] * 100).toFixed(0)}% · Belum ${(b.probabilities["Belum Bekerja"] * 100).toFixed(0)}%
            </div>
          </div>
        </div>
        <div class="action-box">
          <strong>Rekomendasi Penyaluran &amp; Kurikulum</strong>
          ${res.recommendation}
        </div>`;
    }

    ["simCurrPola", "simCurrDasar", "simCurrLanjut", "simCurrFinish", "simCurrUji"].forEach((id) =>
      document.getElementById(id).addEventListener("input", updateSim)
    );
    nullChk.addEventListener("change", updateSim);

    const presets = {
      industri: { pola: 80.5, dasar: 89.0, lanjut: 82.0, isNull: false, finish: 89.5, uji: 86.5 },
      wirausaha: { pola: 87.5, dasar: 82.0, lanjut: 85.0, isNull: true, finish: 81.5, uji: 85.0 },
      remedial: { pola: 70.0, dasar: 72.5, lanjut: 69.0, isNull: false, finish: 68.5, uji: 70.0 },
    };
    document.querySelectorAll("[data-cpreset]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const p = presets[btn.dataset.cpreset];
        document.getElementById("simCurrPola").value = p.pola;
        document.getElementById("simCurrDasar").value = p.dasar;
        lanjutSlider.value = p.lanjut;
        nullChk.checked = p.isNull;
        document.getElementById("simCurrFinish").value = p.finish;
        document.getElementById("simCurrUji").value = p.uji;
        updateSim();
      });
    });

    updateSim();
  }

  // -------------------------------------------------------------------------
  // VIEW 5: MODEL LAB & ARCHITECTURE
  // -------------------------------------------------------------------------
  function initModelLab() {
    CHARTS.renderHorizontalBar(
      document.getElementById("chartFiMaint"),
      ANALYTICS.maintenance.feature_importance.map((d) => ({
        label: d.feature,
        value: Number((d.importance * 100).toFixed(1)),
        color: "#38bdf8",
        tooltip: `Importance: ${(d.importance * 100).toFixed(2)}%`,
      })),
      { valueSuffix: "%" }
    );

    CHARTS.renderHorizontalBar(
      document.getElementById("chartFiDrop"),
      ANALYTICS.dropout.feature_importance.map((d) => ({
        label: d.feature,
        value: Number((d.importance * 100).toFixed(1)),
        color: "#f59e0b",
        tooltip: `Importance: ${(d.importance * 100).toFixed(2)}%`,
      })),
      { valueSuffix: "%" }
    );

    CHARTS.renderHorizontalBar(
      document.getElementById("chartFiCurr"),
      ANALYTICS.curriculum.feature_importance_skema_a.map((d) => ({
        label: d.feature,
        value: Number((d.importance * 100).toFixed(1)),
        color: "#10b981",
        tooltip: `Importance: ${(d.importance * 100).toFixed(2)}%`,
      })),
      { valueSuffix: "%" }
    );
  }

  // Boot all views
  renderOverview();
  initMaintenance();
  initDropout();
  initCurriculum();
  initModelLab();
})();
