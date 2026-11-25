/**
 * Zero-Dependency Interactive SVG Chart Library for Modern Analytics Dashboard.
 * Works 100% offline locally and on GitHub Pages with smooth hover tooltips.
 */
window.ModernCharts = (function () {
  "use strict";

  let tooltipEl = null;

  function ensureTooltip() {
    if (!tooltipEl) {
      tooltipEl = document.createElement("div");
      tooltipEl.className = "chart-tooltip";
      document.body.appendChild(tooltipEl);
    }
    return tooltipEl;
  }

  function showTooltip(evt, html) {
    const tip = ensureTooltip();
    tip.innerHTML = html;
    tip.classList.add("visible");
    const pad = 14;
    let x = evt.clientX + pad;
    let y = evt.clientY + pad;
    if (x + 250 > window.innerWidth) x = evt.clientX - 250;
    if (y + 110 > window.innerHeight) y = evt.clientY - 100;
    tip.style.left = x + "px";
    tip.style.top = y + "px";
  }

  function hideTooltip() {
    if (tooltipEl) tooltipEl.classList.remove("visible");
  }

  function bindTooltip(el, html) {
    el.addEventListener("mousemove", (e) => showTooltip(e, html));
    el.addEventListener("mouseleave", hideTooltip);
  }

  function svgEl(tag, attrs = {}) {
    const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, String(v)));
    return el;
  }

  // 1. Horizontal Ranked Bar Chart (supports positive & diverging negative values)
  function renderHorizontalBar(container, items, opts = {}) {
    container.innerHTML = "";
    const {
      valueSuffix = "",
      maxVal = null,
      diverging = false,
      defaultColor = "#38bdf8",
    } = opts;

    const absMax =
      maxVal ||
      Math.max(...items.map((d) => Math.abs(d.value)), 0.01);

    const wrap = document.createElement("div");
    wrap.style.display = "flex";
    wrap.style.flexDirection = "column";
    wrap.style.gap = "10px";

    items.forEach((item) => {
      const row = document.createElement("div");
      row.style.display = "grid";
      row.style.gridTemplateColumns = "150px 1fr 62px";
      row.style.alignItems = "center";
      row.style.gap = "10px";
      row.style.fontSize = "12px";

      const label = document.createElement("div");
      label.textContent = item.label;
      label.style.color = "#cbd5e1";
      label.style.whiteSpace = "nowrap";
      label.style.overflow = "hidden";
      label.style.textOverflow = "ellipsis";
      label.title = item.label;

      const track = document.createElement("div");
      track.style.position = "relative";
      track.style.height = "12px";
      track.style.borderRadius = "999px";
      track.style.background = "rgba(148,163,184,0.12)";
      track.style.overflow = "hidden";

      const bar = document.createElement("div");
      const color = item.color || defaultColor;
      bar.style.position = "absolute";
      bar.style.top = "0";
      bar.style.bottom = "0";
      bar.style.borderRadius = "999px";
      bar.style.background = color;

      if (diverging) {
        const halfPct = (Math.abs(item.value) / absMax) * 50;
        if (item.value >= 0) {
          bar.style.left = "50%";
          bar.style.width = halfPct + "%";
        } else {
          bar.style.left = 50 - halfPct + "%";
          bar.style.width = halfPct + "%";
        }
        const mid = document.createElement("div");
        mid.style.position = "absolute";
        mid.style.left = "50%";
        mid.style.top = "0";
        mid.style.bottom = "0";
        mid.style.width = "1px";
        mid.style.background = "rgba(241,245,249,0.35)";
        track.appendChild(mid);
      } else {
        bar.style.left = "0";
        bar.style.width = Math.min(100, (Math.abs(item.value) / absMax) * 100) + "%";
      }
      track.appendChild(bar);

      const val = document.createElement("div");
      val.textContent = (item.displayValue !== undefined ? item.displayValue : item.value) + valueSuffix;
      val.style.fontFamily = "var(--font-mono)";
      val.style.textAlign = "right";
      val.style.color = "#f1f5f9";
      val.style.fontWeight = "600";

      row.append(label, track, val);
      bindTooltip(
        row,
        `<strong>${item.label}</strong><br/>${item.tooltip || val.textContent}`
      );
      wrap.appendChild(row);
    });

    container.appendChild(wrap);
  }

  // 2. Grouped Vertical Bar Chart
  function renderGroupedBar(container, categories, series, opts = {}) {
    container.innerHTML = "";
    const width = 560;
    const height = opts.height || 240;
    const pad = { top: 18, right: 16, bottom: 38, left: 42 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;

    const allVals = series.flatMap((s) => s.values);
    const maxV = opts.maxVal || Math.max(...allVals, 1) * 1.12;

    const svg = svgEl("svg", {
      viewBox: `0 0 ${width} ${height}`,
      width: "100%",
      height: height,
    });

    // Horizontal gridlines
    const ticks = 4;
    for (let i = 0; i <= ticks; i++) {
      const val = (maxV / ticks) * i;
      const y = pad.top + plotH - (val / maxV) * plotH;
      svg.appendChild(
        svgEl("line", {
          x1: pad.left,
          x2: width - pad.right,
          y1: y,
          y2: y,
          stroke: "rgba(148,163,184,0.14)",
          "stroke-dasharray": i === 0 ? "0" : "3 4",
        })
      );
      const t = svgEl("text", {
        x: pad.left - 8,
        y: y + 4,
        "text-anchor": "end",
        fill: "#64748b",
        "font-size": 10.5,
        "font-family": "var(--font-mono)",
      });
      t.textContent = Math.round(val);
      svg.appendChild(t);
    }

    const groupW = plotW / categories.length;
    const barGap = 4;
    const barW = Math.min(28, (groupW * 0.72 - barGap * (series.length - 1)) / series.length);

    categories.forEach((cat, ci) => {
      const totalClusterW = series.length * barW + (series.length - 1) * barGap;
      const startX = pad.left + ci * groupW + (groupW - totalClusterW) / 2;

      series.forEach((s, si) => {
        const v = s.values[ci] || 0;
        const h = (v / maxV) * plotH;
        const x = startX + si * (barW + barGap);
        const y = pad.top + plotH - h;

        const rect = svgEl("rect", {
          x,
          y,
          width: barW,
          height: Math.max(h, 1.5),
          rx: 4,
          fill: s.color,
          opacity: 0.9,
          style: "cursor:pointer;transition:opacity 0.15s;",
        });
        rect.addEventListener("mouseenter", () => rect.setAttribute("opacity", "1"));
        rect.addEventListener("mouseleave", () => rect.setAttribute("opacity", "0.9"));
        bindTooltip(
          rect,
          `<strong>${cat}</strong><br/>${s.name}: <b>${v}${opts.valueSuffix || ""}</b>`
        );
        svg.appendChild(rect);
      });

      const lbl = svgEl("text", {
        x: pad.left + ci * groupW + groupW / 2,
        y: height - 14,
        "text-anchor": "middle",
        fill: "#94a3b8",
        "font-size": 11,
      });
      lbl.textContent = cat;
      svg.appendChild(lbl);
    });

    container.appendChild(svg);

    const leg = document.createElement("div");
    leg.className = "chart-legend";
    series.forEach((s) => {
      const item = document.createElement("span");
      item.className = "legend-item";
      item.innerHTML = `<span class="legend-swatch" style="background:${s.color}"></span>${s.name}`;
      leg.appendChild(item);
    });
    container.appendChild(leg);
  }

  // 3. Donut Chart
  function renderDonut(container, segments, centerText, centerSub) {
    container.innerHTML = "";
    const total = segments.reduce((a, b) => a + b.value, 0) || 1;
    const size = 210;
    const r = 74;
    const cx = size / 2;
    const cy = size / 2;
    const circ = 2 * Math.PI * r;

    const wrap = document.createElement("div");
    wrap.style.display = "flex";
    wrap.style.flexWrap = "wrap";
    wrap.style.alignItems = "center";
    wrap.style.justifyContent = "center";
    wrap.style.gap = "20px";

    const svg = svgEl("svg", { viewBox: `0 0 ${size} ${size}`, width: size, height: size });
    let offset = 0;

    segments.forEach((seg) => {
      const frac = seg.value / total;
      const dash = frac * circ;
      const circle = svgEl("circle", {
        cx,
        cy,
        r,
        fill: "none",
        stroke: seg.color,
        "stroke-width": 22,
        "stroke-dasharray": `${Math.max(dash - 2, 1)} ${circ}`,
        "stroke-dashoffset": -offset,
        transform: `rotate(-90 ${cx} ${cy})`,
        style: "cursor:pointer;",
      });
      const pct = ((seg.value / total) * 100).toFixed(1);
      bindTooltip(circle, `<strong>${seg.label}</strong><br/>Jumlah: <b>${seg.value}</b> (${pct}%)`);
      svg.appendChild(circle);
      offset += dash;
    });

    const t1 = svgEl("text", {
      x: cx,
      y: cy - 2,
      "text-anchor": "middle",
      fill: "#f1f5f9",
      "font-size": 22,
      "font-weight": 700,
      "font-family": "var(--font-mono)",
    });
    t1.textContent = centerText;
    const t2 = svgEl("text", {
      x: cx,
      y: cy + 16,
      "text-anchor": "middle",
      fill: "#94a3b8",
      "font-size": 11,
    });
    t2.textContent = centerSub;
    svg.append(t1, t2);

    const list = document.createElement("div");
    list.style.display = "flex";
    list.style.flexDirection = "column";
    list.style.gap = "10px";
    list.style.minWidth = "170px";

    segments.forEach((seg) => {
      const pct = ((seg.value / total) * 100).toFixed(1);
      const row = document.createElement("div");
      row.style.display = "flex";
      row.style.justifyContent = "space-between";
      row.style.alignItems = "center";
      row.style.gap = "12px";
      row.style.fontSize = "12.5px";
      row.innerHTML = `<span class="legend-item"><span class="legend-swatch" style="background:${seg.color}"></span>${seg.label}</span><strong style="font-family:var(--font-mono)">${seg.value} (${pct}%)</strong>`;
      list.appendChild(row);
    });

    wrap.append(svg, list);
    container.appendChild(wrap);
  }

  // 4. Interactive XY Scatter Chart
  function renderScatter(container, points, opts = {}) {
    container.innerHTML = "";
    const width = 560;
    const height = opts.height || 260;
    const pad = { top: 18, right: 20, bottom: 40, left: 52 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;

    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    const minX = opts.minX !== undefined ? opts.minX : Math.min(...xs, 0);
    const maxX = Math.max(...xs, minX + 1) * 1.05;
    const minY = opts.minY !== undefined ? opts.minY : Math.min(...ys, 0);
    const maxY = Math.max(...ys, minY + 1) * 1.08;

    const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%", height });

    // Grid
    for (let i = 0; i <= 4; i++) {
      const yv = minY + ((maxY - minY) / 4) * i;
      const y = pad.top + plotH - ((yv - minY) / (maxY - minY)) * plotH;
      svg.appendChild(
        svgEl("line", {
          x1: pad.left,
          x2: width - pad.right,
          y1: y,
          y2: y,
          stroke: "rgba(148,163,184,0.12)",
          "stroke-dasharray": "3 4",
        })
      );
      const yt = svgEl("text", {
        x: pad.left - 8,
        y: y + 4,
        "text-anchor": "end",
        fill: "#64748b",
        "font-size": 10,
        "font-family": "var(--font-mono)",
      });
      yt.textContent = Math.round(yv);
      svg.appendChild(yt);

      const xv = minX + ((maxX - minX) / 4) * i;
      const x = pad.left + ((xv - minX) / (maxX - minX)) * plotW;
      const xt = svgEl("text", {
        x,
        y: height - 22,
        "text-anchor": "middle",
        fill: "#64748b",
        "font-size": 10,
        "font-family": "var(--font-mono)",
      });
      xt.textContent = Math.round(xv);
      svg.appendChild(xt);
    }

    // Axis titles
    const xLbl = svgEl("text", {
      x: pad.left + plotW / 2,
      y: height - 4,
      "text-anchor": "middle",
      fill: "#94a3b8",
      "font-size": 11,
    });
    xLbl.textContent = opts.xLabel || "";
    svg.appendChild(xLbl);

    const yLbl = svgEl("text", {
      x: 13,
      y: pad.top + plotH / 2,
      "text-anchor": "middle",
      fill: "#94a3b8",
      "font-size": 11,
      transform: `rotate(-90 13 ${pad.top + plotH / 2})`,
    });
    yLbl.textContent = opts.yLabel || "";
    svg.appendChild(yLbl);

    // Points
    points.forEach((p) => {
      const cx = pad.left + ((p.x - minX) / (maxX - minX)) * plotW;
      const cy = pad.top + plotH - ((p.y - minY) / (maxY - minY)) * plotH;
      const dot = svgEl("circle", {
        cx,
        cy,
        r: 4.5,
        fill: p.color,
        "fill-opacity": 0.82,
        stroke: "#070b14",
        "stroke-width": 1,
        style: "cursor:pointer;",
      });
      dot.addEventListener("mouseenter", () => dot.setAttribute("r", "7"));
      dot.addEventListener("mouseleave", () => dot.setAttribute("r", "4.5"));
      bindTooltip(dot, p.tooltip);
      svg.appendChild(dot);
    });

    container.appendChild(svg);

    if (opts.legend) {
      const leg = document.createElement("div");
      leg.className = "chart-legend";
      opts.legend.forEach((l) => {
        const sp = document.createElement("span");
        sp.className = "legend-item";
        sp.innerHTML = `<span class="legend-swatch" style="background:${l.color}"></span>${l.label}`;
        leg.appendChild(sp);
      });
      container.appendChild(leg);
    }
  }

  // 5. Semi-Circular Probability Gauge with Decision Threshold Marker
  function renderRiskGauge(container, prob, threshold, title) {
    container.innerHTML = "";
    const pct = Math.max(0, Math.min(1, prob));
    const isRisk = pct >= threshold;
    const color = isRisk ? "#f43f5e" : "#10b981";
    const w = 180;
    const h = 108;
    const r = 68;
    const cx = w / 2;
    const cy = 88;
    const semiCirc = Math.PI * r;

    const svg = svgEl("svg", { viewBox: `0 0 ${w} ${h}`, width: w, height: h });

    // Background arc
    svg.appendChild(
      svgEl("path", {
        d: `M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`,
        fill: "none",
        stroke: "rgba(148,163,184,0.16)",
        "stroke-width": 13,
        "stroke-linecap": "round",
      })
    );

    // Value arc
    svg.appendChild(
      svgEl("path", {
        d: `M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`,
        fill: "none",
        stroke: color,
        "stroke-width": 13,
        "stroke-linecap": "round",
        "stroke-dasharray": `${pct * semiCirc} ${semiCirc}`,
      })
    );

    // Threshold tick mark
    const angle = Math.PI * (1 - threshold);
    const tx1 = cx + (r - 11) * Math.cos(angle);
    const ty1 = cy - (r - 11) * Math.sin(angle);
    const tx2 = cx + (r + 11) * Math.cos(angle);
    const ty2 = cy - (r + 11) * Math.sin(angle);
    svg.appendChild(
      svgEl("line", {
        x1: tx1,
        y1: ty1,
        x2: tx2,
        y2: ty2,
        stroke: "#f59e0b",
        "stroke-width": 2.5,
      })
    );

    const valTxt = svgEl("text", {
      x: cx,
      y: cy - 14,
      "text-anchor": "middle",
      fill: color,
      "font-size": 22,
      "font-weight": 700,
      "font-family": "var(--font-mono)",
    });
    valTxt.textContent = (pct * 100).toFixed(1) + "%";

    const thrTxt = svgEl("text", {
      x: cx,
      y: cy + 4,
      "text-anchor": "middle",
      fill: "#94a3b8",
      "font-size": 10.5,
      "font-family": "var(--font-mono)",
    });
    thrTxt.textContent = `Threshold: ${(threshold * 100).toFixed(0)}%`;

    svg.append(valTxt, thrTxt);

    const head = document.createElement("div");
    head.style.fontSize = "12px";
    head.style.fontWeight = "600";
    head.style.color = "#cbd5e1";
    head.style.marginBottom = "4px";
    head.textContent = title;

    const badge = document.createElement("div");
    badge.style.fontSize = "11.5px";
    badge.style.fontWeight = "700";
    badge.style.color = color;
    badge.style.marginTop = "2px";
    badge.textContent = isRisk ? "BERISIKO DROP-OUT" : "AMAN / TERKENDALI";

    container.append(head, svg, badge);
  }

  return {
    renderHorizontalBar,
    renderGroupedBar,
    renderDonut,
    renderScatter,
    renderRiskGauge,
  };
})();
