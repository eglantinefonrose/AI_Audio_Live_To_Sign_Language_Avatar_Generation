"""Graphique des performances : une barre par run d'évaluation (eval/results/*.json).

Ordonnée : taux d'exactitude (ou F1, ordre, couverture LSFB, au choix dans la page).
Abscisse : le modèle (et la méthode, pour distinguer deux runs du même modèle), les
runs étant groupés par dataset d'éval (on ne compare que des runs de même dataset).

Sortie : `eval/graphique_resultats.html`, page autonome (SVG, sans dépendance) avec
infobulle au survol, filtres par dataset et tableau de détail.

    uv run python -m src.eval.graphique [--ouvrir] [--dataset a77bfbad8a13 ...]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = HERE / "graphique_resultats.html"


def short_method(r: dict) -> str:
    """Méthode lisible sous le nom du modèle : prompt + options."""
    parts = ["règles seules" if r.get("prompt_mode") == "rules-only" else "prod"]
    method = r.get("method") or "aucune"
    if method != "aucune":
        parts += method.split("+")
    return " + ".join(parts)


ABBREV = {"few-shot-cible": "FSC", "post-traitement": "PP", "vocabulaire-lsfb": "VOC",
          "indices-lsfb": "IND", "few-shot-dynamique": "FSD"}


def code_method(r: dict) -> str:
    """Code court de la méthode, affiché sous le nom du modèle (légende dans la page)."""
    parts = ["règles" if r.get("prompt_mode") == "rules-only" else "prod"]
    method = r.get("method") or "aucune"
    if method != "aucune":
        parts += [next((v for k, v in ABBREV.items() if m.startswith(k)), m) for m in method.split("+")]
    return "·".join(parts)


def load_runs(datasets: list[str] | None) -> list[dict]:
    runs = []
    for f in sorted(RESULTS.glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        if datasets and r.get("dataset_sha") not in datasets:
            continue
        m = r.get("metrics", {})
        cov = m.get("couverture_lsfb")
        if cov is None and r.get("per_sentence"):      # anciens records : recalculée
            try:
                from src.lsf_translator import lsfb
                ok = tot = 0
                for s in r["per_sentence"]:
                    a, b = lsfb.coverage(s["pred"])
                    ok, tot = ok + a, tot + b
                cov = ok / tot if tot else None
            except Exception:
                cov = None
        runs.append({
            "fichier": f.name, "label": r.get("label", ""), "modele": r.get("model", "?"),
            "methode": short_method(r), "code": code_method(r), "dataset": r.get("dataset_sha", "?"),
            "n": r.get("dataset_n") or m.get("n"), "date": r.get("timestamp", "")[:16].replace("T", " "),
            "exact": m.get("exact"), "f1": m.get("f1"), "ordre": m.get("order"), "couverture": cov,
        })
    # Groupes par dataset (dans l'ordre d'apparition), runs chronologiques dans un groupe.
    first = {}
    for i, r in enumerate(sorted(runs, key=lambda r: r["date"])):
        first.setdefault(r["dataset"], i)
    return sorted(runs, key=lambda r: (first[r["dataset"]], r["date"]))


PAGE = r"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Performances des runs</title>
<style>
:root { --surface:#fcfcfb; --page:#f4f3f0; --ink:#0b0b0b; --ink2:#52514e; --muted:#8a8984; --grid:#e6e5e1;
        --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100; --s5:#e87ba4; --s6:#008300; --s7:#6250d6; --s8:#e34948; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --surface:#1a1a19; --page:#121211; --ink:#f0efec;
        --ink2:#c3c2b7; --muted:#8f8e88; --grid:#2e2e2c;
        --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#e66767; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--page); color:var(--ink); font:15px/1.45 system-ui, sans-serif; }
main { max-width:1200px; margin:0 auto; padding:20px 16px 40px; }
h1 { font-size:20px; margin:0 0 4px; }
.sub { color:var(--ink2); margin:0 0 16px; font-size:14px; }
.controls { display:flex; flex-wrap:wrap; gap:8px 18px; align-items:center; margin-bottom:12px; font-size:14px; }
.controls label { display:inline-flex; gap:6px; align-items:center; cursor:pointer; }
.sw { width:12px; height:12px; border-radius:3px; display:inline-block; }
select { font:inherit; padding:4px 8px; border-radius:6px; border:1px solid var(--grid); background:var(--surface); color:var(--ink); }
.card { background:var(--surface); border:1px solid var(--grid); border-radius:12px; padding:12px; }
.scroll { overflow-x:auto; }
svg text { fill:var(--ink2); font-size:11px; }
svg .axis-title { font-size:12px; fill:var(--ink2); }
svg .value { fill:var(--ink); font-size:11px; font-weight:600; }
svg .group { fill:var(--ink); font-size:12px; font-weight:600; }
.bar { cursor:default; }
.bar:hover, .bar.focus { filter:brightness(1.12); }
#tip { position:fixed; pointer-events:none; background:var(--surface); color:var(--ink); border:1px solid var(--grid);
       border-radius:8px; padding:8px 10px; font-size:13px; box-shadow:0 4px 16px rgba(0,0,0,.15); max-width:340px; display:none; z-index:5; }
#tip b { display:block; margin-bottom:2px; }
#tip .k { color:var(--ink2); }
details { margin-top:16px; }
table { border-collapse:collapse; width:100%; font-size:13px; }
th, td { text-align:left; padding:5px 8px; border-bottom:1px solid var(--grid); }
th { color:var(--ink2); font-weight:600; }
td.num { text-align:right; font-variant-numeric:tabular-nums; }
</style></head><body><main>
<h1>Performances des runs d'évaluation</h1>
<p class="sub">Une barre par run, groupées par dataset d'éval : ne comparer que des runs de même dataset.
Survol d'une barre : détail du run. Meilleur run de chaque dataset annoté.<br>
Sous chaque modèle : prompt (<b>prod</b> = règles + exemples, <b>règles</b> = règles seules) et options :
<b>FSC</b> few-shot ciblé · <b>PP</b> post-traitement · <b>VOC</b> vocabulaire LSFB · <b>IND</b> indications LSFB.</p>
<div class="controls">
  <label>Mesure <select id="metric">
    <option value="exact">Taux d'exactitude</option><option value="f1">F1</option>
    <option value="ordre">Ordre</option><option value="couverture">Couverture LSFB</option>
  </select></label>
  <span id="filters"></span>
</div>
<div class="card scroll"><svg id="chart" role="img" aria-label="Graphique en barres des performances par run"></svg></div>
<details><summary>Tableau des runs</summary><div class="scroll"><table id="table"></table></div></details>
<div id="tip"></div>
</main>
<script>
const RUNS = __RUNS__;
const METRICS = {exact:"Taux d'exactitude", f1:"F1", ordre:"Ordre", couverture:"Couverture LSFB"};
const datasets = [...new Set(RUNS.map(r => r.dataset))];
const color = d => `var(--s${datasets.indexOf(d) % 8 + 1})`;   // ordre fixe, jamais réattribué
const visible = new Set(datasets);
const $ = id => document.getElementById(id);
const pct = v => v == null ? "—" : (v * 100).toFixed(0) + " %";
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

$("filters").innerHTML = datasets.map(d => {
  const n = RUNS.filter(r => r.dataset === d); const ph = n[0].n ? ` · ${n[0].n} phrases` : "";
  return `<label><input type="checkbox" data-d="${d}" checked><span class="sw" style="background:${color(d)}"></span>${d}${ph} (${n.length} runs)</label>`;
}).join("");
$("filters").addEventListener("change", e => { const d = e.target.dataset.d;
  e.target.checked ? visible.add(d) : visible.delete(d); draw(); });
$("metric").addEventListener("change", draw);

function draw() {
  const key = $("metric").value;
  const runs = RUNS.filter(r => visible.has(r.dataset));
  const BW = 24, GAP = 2, GROUP_GAP = 20, LEFT = 52, TOP = 24, H = 300, LABEL_H = 215;
  // positions : barres d'un même dataset accolées, espace entre datasets
  let x = LEFT, prev = null; const pos = [];
  runs.forEach(r => { if (prev !== null && r.dataset !== prev) x += GROUP_GAP; pos.push(x); x += BW + GAP; prev = r.dataset; });
  const W = Math.max(x + 16, 400), y = v => TOP + H - (v || 0) * H;
  let s = "";
  for (const t of [0, .25, .5, .75, 1]) {
    s += `<line x1="${LEFT}" x2="${W - 8}" y1="${y(t)}" y2="${y(t)}" stroke="var(--grid)" stroke-width="1"/>`;
    s += `<text x="${LEFT - 8}" y="${y(t) + 4}" text-anchor="end">${t * 100} %</text>`;
  }
  s += `<text class="axis-title" transform="translate(14 ${TOP + H / 2}) rotate(-90)" text-anchor="middle">${METRICS[key]}</text>`;
  // meilleur run par dataset (étiquette directe sélective)
  const best = {};
  runs.forEach((r, i) => { if (r[key] != null && (best[r.dataset] == null || r[key] > runs[best[r.dataset]][key])) best[r.dataset] = i; });
  runs.forEach((r, i) => {
    const v = r[key], h = v == null ? 0 : Math.max(1, v * H), x0 = pos[i], top = TOP + H - h, rad = Math.min(4, h);
    // barre : coins arrondis en haut seulement, ancrée sur la ligne de base
    s += `<path class="bar" data-i="${i}" fill="${color(r.dataset)}" d="M${x0},${TOP + H} V${top + rad} Q${x0},${top} ${x0 + rad},${top} H${x0 + BW - rad} Q${x0 + BW},${top} ${x0 + BW},${top + rad} V${TOP + H} Z"/>`;
    if (best[r.dataset] === i) s += `<text class="value" x="${x0 + BW / 2}" y="${top - 6}" text-anchor="middle">${pct(v)}</text>`;
    // étiquette d'abscisse : « modèle · méthode » sur une ligne, inclinée
    const lx = x0 + BW / 2 + 4, ly = TOP + H + 8;
    s += `<text transform="translate(${lx} ${ly}) rotate(-62)" text-anchor="end"><tspan style="fill:var(--ink)">${esc(r.modele)}</tspan> · ${esc(r.code)}</text>`;
  });
  // nom du dataset sous chaque groupe
  let i = 0;
  while (i < runs.length) {
    let j = i; while (j + 1 < runs.length && runs[j + 1].dataset === runs[i].dataset) j++;
    const cx = (pos[i] + pos[j] + BW) / 2, yy = TOP + H + LABEL_H;
    s += `<line x1="${pos[i]}" x2="${pos[j] + BW}" y1="${yy - 16}" y2="${yy - 16}" stroke="${color(runs[i].dataset)}" stroke-width="3" stroke-linecap="round"/>`;
    s += `<text class="group" x="${cx}" y="${yy}" text-anchor="middle">dataset ${runs[i].dataset}</text>`;
    i = j + 1;
  }
  const svg = $("chart");
  svg.setAttribute("viewBox", `0 0 ${W} ${TOP + H + LABEL_H + 12}`);
  svg.setAttribute("width", W); svg.setAttribute("height", TOP + H + LABEL_H + 12);
  svg.innerHTML = s;
  svg.querySelectorAll(".bar").forEach(b => {
    b.addEventListener("mousemove", e => tip(e, runs[+b.dataset.i]));
    b.addEventListener("mouseleave", () => { $("tip").style.display = "none"; });
  });
  $("table").innerHTML = "<tr><th>Date</th><th>Dataset</th><th>Modèle</th><th>Méthode</th><th>Label</th>"
    + "<th>Exact</th><th>F1</th><th>Ordre</th><th>Couv. LSFB</th></tr>"
    + runs.map(r => `<tr><td>${r.date}</td><td>${r.dataset}</td><td>${esc(r.modele)}</td><td>${esc(r.methode)}</td>`
      + `<td>${esc(r.label)}</td><td class="num">${pct(r.exact)}</td><td class="num">${r.f1?.toFixed(3) ?? "—"}</td>`
      + `<td class="num">${r.ordre?.toFixed(3) ?? "—"}</td><td class="num">${pct(r.couverture)}</td></tr>`).join("");
}

function tip(e, r) {
  const t = $("tip");
  t.innerHTML = `<b>${esc(r.modele)}</b><div class="k">${esc(r.label)}</div>`
    + `<div><span class="k">Dataset</span> ${r.dataset}${r.n ? ` (${r.n} phrases)` : ""} · <span class="k">${r.date}</span></div>`
    + `<div><span class="k">Méthode</span> ${esc(r.methode)}</div>`
    + `<div><span class="k">Exact</span> ${pct(r.exact)} · <span class="k">F1</span> ${r.f1?.toFixed(3) ?? "—"}`
    + ` · <span class="k">Ordre</span> ${r.ordre?.toFixed(3) ?? "—"} · <span class="k">Couv.</span> ${pct(r.couverture)}</div>`;
  t.style.display = "block";
  const w = t.offsetWidth, h = t.offsetHeight;
  t.style.left = Math.min(e.clientX + 14, innerWidth - w - 8) + "px";
  t.style.top = (e.clientY - h - 12 < 8 ? e.clientY + 16 : e.clientY - h - 12) + "px";
}
draw();
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", nargs="*", help="ne garder que ces datasets d'éval (hash)")
    ap.add_argument("--sortie", default=str(OUT), help="fichier HTML produit")
    ap.add_argument("--ouvrir", action="store_true", help="ouvrir la page dans le navigateur")
    args = ap.parse_args()
    runs = load_runs(args.dataset)
    if not runs:
        sys.exit("Aucun run à afficher.")
    out = Path(args.sortie)
    out.write_text(PAGE.replace("__RUNS__", json.dumps(runs, ensure_ascii=False).replace("</", "<\\/")),
                   encoding="utf-8")
    print(f"{len(runs)} runs, {len({r['dataset'] for r in runs})} datasets → {out}")
    if args.ouvrir:
        opener = {"darwin": ["open"], "win32": ["cmd", "/c", "start", ""]}.get(sys.platform, ["xdg-open"])
        subprocess.run([*opener, str(out)], check=False)


if __name__ == "__main__":
    main()
