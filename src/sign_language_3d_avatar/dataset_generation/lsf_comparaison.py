"""Page d'étiquetage LSFB → LSF : pour chaque glose, regarder le clip LSFB et dire si
c'est le même signe en LSF de France (identique / variante / différent / incertain).

`lsfb_mapping.csv` ne rapproche que les **mots** (même étiquette française) : un statut
« exact » ne dit pas que le **geste** est le même en LSF. Cette page sert à le noter vite :
un signe à la fois, clip en boucle, verdict au clavier (1 à 4) et passage automatique au
suivant. Les gloses les plus fréquentes dans les datasets du projet viennent en premier,
et les formes directionnelles d'un verbe (AIDER, AIDER-À-MOI…) ne font qu'une carte.

Sorties :
- `lsfb_lsf_comparaison.csv` (versionné) : une ligne par glose, colonnes `verdict`,
  `note`, `date` ; les verdicts déjà présents sont conservés à chaque relance ;
- `data/lsfb/comparaison.html` (ignoré par git, à côté des vidéos) : la page. Les
  réponses restent dans le navigateur entre deux visites ; « Exporter le CSV » les
  rapatrie, à copier à la place de `lsfb_lsf_comparaison.csv`.

Les clips LSFB sont en MPEG-4 Part 2, que les navigateurs ne lisent pas : le script en
fait des copies H.264 (ffmpeg) dans `data/lsfb/videos_h264/`, sans toucher aux originaux.

Prérequis : une vidéo LSFB par glose, téléchargée avec
    uv run python src/sign_language_3d_avatar/dataset_generation/lsfb_subset.py download \\
        --words-file src/sign_language_3d_avatar/dataset_generation/lsfb_words.txt \\
        --videos --landmarks pose --max-par-glose 1

Usage :
    uv run python src/sign_language_3d_avatar/dataset_generation/lsf_comparaison.py [--ouvrir]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent   # src/sign_language_3d_avatar/
SRC = BASE.parent                                # src/ (eval/, finetune/)
GEN = BASE / "dataset_generation"
MAPPING = GEN / "lsfb_mapping.csv"
CSV_PATH = GEN / "lsfb_lsf_comparaison.csv"
DATA = GEN / "data" / "lsfb"
MANIFEST = DATA / "manifest.csv"
HTML_PATH = DATA / "comparaison.html"
H264 = DATA / "videos_h264"   # copies lisibles par les navigateurs
FIELDS = ["glose", "etiquette_lsfb", "statut", "frequence", "verdict", "note", "date"]
DIRECTIONAL_RE = re.compile(r"^(.+)-À-(MOI|TOI|LUI|NOUS|VOUS|EUX)$")
# Gloses dont le clip LSFB ne correspond pas directement à l'usage dans le projet.
SPECIAL_NOTES = {
    "QUESTION": "Dans le projet, QUESTION est surtout le marqueur de question fermée, porté par le "
                "visage en LSF ; le clip LSFB montre le nom « question ». Juge le signe du nom.",
}


# ------------------------------------------------------------------- données
def gloss_frequencies() -> Counter:
    """Occurrences de chaque glose dans l'éval et le dataset d'entraînement le plus récent."""
    freq: Counter = Counter()
    sources = [SRC / "eval" / "dataset.jsonl"]
    pairs = list((SRC / "finetune" / "data").glob("*/pairs.jsonl"))
    if pairs:   # le plus récent (date de création du manifeste)
        def created(p: Path) -> str:
            m = p.parent / "manifest.json"
            return json.loads(m.read_text(encoding="utf-8")).get("created", "") if m.exists() else ""
        sources.append(max(pairs, key=created))
    for path in sources:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                freq.update(json.loads(line)["glosses"])
    return freq


def load_rows() -> list[dict]:
    with open(MAPPING, newline="", encoding="utf-8") as f:
        mapping = {r["glose_projet"]: r for r in csv.DictReader(f)}
    videos: dict[str, str] = {}
    if MANIFEST.exists():
        with open(MANIFEST, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                v = next((p for p in r["fichiers"].split() if p.endswith(".mp4")), None)
                if v and (DATA / v).exists():
                    videos.setdefault(r["glose_projet"], v)
    freq = gloss_frequencies()
    rows = []
    for gloss, m in mapping.items():
        if gloss not in videos:     # sans clip LSFB, rien à juger
            continue
        rows.append({
            "glose": gloss, "etiquette_lsfb": m["etiquette_lsfb"], "statut": m["statut"],
            "frequence": freq.get(gloss, 0), "instances": m.get("instances", ""),
            "autres": m.get("autres_etiquettes", ""), "note_mapping": m.get("note", ""),
            "partage": [g for g, o in mapping.items()
                        if g != gloss and o["etiquette_lsfb"] == m["etiquette_lsfb"]],
            "video": videos[gloss],
        })
    rows.sort(key=lambda r: (-r["frequence"], r["glose"]))
    return rows


def group_cards(rows: list[dict], old: dict[str, dict]) -> list[dict]:
    """Une carte par signe à juger : les formes directionnelles d'un verbe (AIDER,
    AIDER-À-MOI, AIDER-À-TOI) montrent toutes la forme de base, un seul verdict suffit."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        m = DIRECTIONAL_RE.match(r["glose"])
        groups.setdefault(m.group(1) if m else r["glose"], []).append(r)
    cards = []
    for key, members in groups.items():
        main = next((r for r in members if r["glose"] == key), members[0])
        names = [r["glose"] for r in members]
        # Verdict de la carte : le plus récent parmi ceux déjà saisis pour ses gloses.
        prev = max((old[g] for g in names if old.get(g, {}).get("verdict")),
                   key=lambda o: o.get("date", ""), default={})
        others = sorted({g for r in members for g in r["partage"]} - set(names))
        cards.append(main | {
            "glose": key, "membres": names, "frequence": sum(r["frequence"] for r in members),
            "lignes": [{k: r[k] for k in FIELDS[:4]} for r in members],
            "partage": ", ".join(others), "note_mapping": SPECIAL_NOTES.get(key, main["note_mapping"]),
            "verdict": prev.get("verdict", ""), "note": prev.get("note", ""), "date": prev.get("date", ""),
        })
    cards.sort(key=lambda c: (-c["frequence"], c["glose"]))
    return cards


def read_csv() -> dict[str, dict]:
    if not CSV_PATH.exists():
        return {}
    with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
        return {r["glose"]: r for r in csv.DictReader(f)}


def write_csv(rows: list[dict], old: dict[str, dict]) -> None:
    """Réécrit le CSV dans l'ordre de la page, sans perdre aucun verdict (même pour une
    glose qui ne serait plus dans la page)."""
    out = []
    for r in rows:
        prev = old.get(r["glose"], {})
        out.append({k: r[k] for k in FIELDS[:4]} | {k: prev.get(k, "") for k in FIELDS[4:]})
    listed = {r["glose"] for r in rows}
    out += [{k: v.get(k, "") for k in FIELDS} for g, v in old.items() if g not in listed]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out)


def to_h264(rows: list[dict]) -> int:
    """Copies H.264 des clips (les originaux LSFB sont en MPEG-4 Part 2, illisibles dans
    Chrome et Safari). Seules les copies manquantes sont produites. Renvoie le nombre de
    clips restés sans copie lisible."""
    if not shutil.which("ffmpeg"):
        print("⚠️  ffmpeg introuvable (brew install ffmpeg) : les clips ne seront pas lisibles.")
        return len({r["video"] for r in rows})
    H264.mkdir(parents=True, exist_ok=True)

    def convert(rel: str) -> bool:
        out = H264 / Path(rel).name
        if out.exists():
            return True
        tmp = out.with_name(out.stem + ".part.mp4")
        ok = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(DATA / rel), "-an",
                             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
                             "-crf", "20", "-movflags", "+faststart", str(tmp)]).returncode == 0
        if ok:
            tmp.rename(out)
        else:
            tmp.unlink(missing_ok=True)
        return ok

    todo = sorted({r["video"] for r in rows})
    with ThreadPoolExecutor(4) as ex:
        results = dict(zip(todo, ex.map(convert, todo)))
    for r in rows:
        if results.get(r["video"]):
            r["video"] = f"{H264.name}/{Path(r['video']).name}"
    return sum(not ok for ok in results.values())


# ---------------------------------------------------------------------- page
PAGE = r"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Étiquetage LSFB → LSF</title>
<style>
:root { --bg:#f6f5f2; --card:#fff; --ink:#1d1d1f; --mute:#6b6b70; --line:#e2e0da; --accent:#2457c5;
        --ok:#1e6b3a; --mid:#a15c00; --bad:#b3261e; --none:#5f5f66; }
@media (prefers-color-scheme: dark) { :root { --bg:#17171a; --card:#222226; --ink:#ececf0; --mute:#a0a0a8;
        --line:#34343a; --accent:#8ab4ff; --ok:#81c995; --mid:#ffb74d; --bad:#ff8a80; --none:#b0b0b8; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.45 system-ui, sans-serif; }
header { position:sticky; top:0; z-index:1; background:var(--bg); display:flex; flex-wrap:wrap;
         gap:8px 16px; align-items:center; padding:10px 16px; border-bottom:1px solid var(--line); }
header h1 { font-size:17px; margin:0; }
#stats { color:var(--mute); font-size:14px; margin-right:auto; }
#stats b { color:var(--ink); }
button, select, input, textarea { font:inherit; font-size:14px; border-radius:6px; border:1px solid var(--line);
         background:var(--card); color:var(--ink); }
button, select, input { padding:5px 10px; }
button { cursor:pointer; }
#exportBtn.due { border-color:var(--bad); color:var(--bad); font-weight:600; }
main { max-width:760px; margin:0 auto; padding:16px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px;
        display:flex; flex-direction:column; gap:10px; }
#gloss { font-size:30px; font-weight:700; margin:0; }
#forms { font-size:14px; color:var(--mute); }
video { width:100%; max-height:55vh; border-radius:8px; background:#000; }
.row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
.meta { font-size:13px; color:var(--mute); }
.meta b { color:var(--ink); font-weight:600; }
.warn { font-size:13px; color:var(--mid); }
.question { font-size:16px; font-weight:600; margin-top:4px; }
.verdicts { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
.verdicts button { padding:12px; text-align:left; font-size:15px; }
.verdicts button small { display:block; color:var(--mute); font-size:12px; }
.verdicts button kbd { float:right; }
.verdicts button.on { outline:2px solid var(--accent); }
[data-v=identique] b { color:var(--ok); } [data-v=variante] b { color:var(--mid); }
[data-v="différent"] b { color:var(--bad); } [data-v=incertain] b { color:var(--none); }
.v-identique { color:var(--ok); } .v-variante { color:var(--mid); } .v-différent { color:var(--bad); }
.v-incertain { color:var(--none); }
kbd { font:12px ui-monospace, monospace; border:1px solid var(--line); border-bottom-width:2px;
      border-radius:4px; padding:0 5px; color:var(--mute); }
textarea { width:100%; padding:6px; background:var(--bg); resize:vertical; }
.help { font-size:12px; color:var(--mute); }
@media (max-width: 520px) { .verdicts { grid-template-columns:1fr; } }
</style></head><body>
<header>
  <h1>LSFB → LSF</h1>
  <span id="stats"></span>
  <label>Afficher <select id="filter">
    <option value="todo">à étiqueter</option><option value="all">toutes</option>
    <option value="identique">identique</option><option value="variante">variante</option>
    <option value="différent">différent</option><option value="incertain">incertain</option>
  </select></label>
  <input id="jump" list="glosses" placeholder="Aller à une glose…" size="16">
  <datalist id="glosses"></datalist>
  <button id="exportBtn" title="Télécharge le CSV à copier dans le projet">Exporter le CSV</button>
</header>
<main><section class="card">
  <div class="row" style="justify-content:space-between">
    <h2 id="gloss"></h2><span id="pos" class="meta"></span>
  </div>
  <div id="forms"></div>
  <video id="clip" autoplay loop muted playsinline></video>
  <div class="row">
    <button id="prev">← Précédent <kbd>←</kbd></button>
    <button id="replay">Rejouer <kbd>Espace</kbd></button>
    <button id="slow"><span id="speed">Ralenti</span> <kbd>L</kbd></button>
    <button id="next">Suivant <kbd>→</kbd></button>
  </div>
  <div id="meta" class="meta"></div>
  <div id="warn" class="warn"></div>
  <div class="question">Est-ce le même signe en LSF de France ? <span id="current"></span></div>
  <div class="verdicts">
    <button data-v="identique"><b>Identique</b> <kbd>1</kbd><small>même signe en LSF</small></button>
    <button data-v="variante"><b>Variante</b> <kbd>2</kbd><small>même signe, un détail diffère</small></button>
    <button data-v="différent"><b>Différent</b> <kbd>3</kbd><small>autre signe en LSF</small></button>
    <button data-v="incertain"><b>Je ne sais pas</b> <kbd>4</kbd><small>à revoir plus tard</small></button>
  </div>
  <textarea id="note" rows="2" placeholder="Note facultative (N pour écrire, Échap pour revenir)"></textarea>
  <div class="help"><kbd>1</kbd>–<kbd>4</kbd> répondre et passer au suivant · <kbd>0</kbd> effacer la réponse ·
    <kbd>←</kbd> <kbd>→</kbd> naviguer · <kbd>Espace</kbd> rejouer · <kbd>L</kbd> ralenti · <kbd>N</kbd> note.
    Les réponses sont gardées dans ce navigateur : exporte le CSV régulièrement.</div>
</section></main>
<script>
const SIGNS = __SIGNS__;
const KEY = "lsfb-lsf-etiquetage";
const EXPORT_EVERY = 25;
let store = { verdicts: {}, sinceExport: 0 };
try { store = Object.assign(store, JSON.parse(localStorage.getItem(KEY) || "{}")); } catch (e) {}
// Le CSV du projet fait foi s'il est plus récent que ce que le navigateur a gardé.
for (const s of SIGNS) {
  const local = store.verdicts[s.glose];
  if (s.verdict && (!local || (s.date || "") > (local.date || "")))
    store.verdicts[s.glose] = { verdict: s.verdict, note: s.note || "", date: s.date || "" };
}
function save() { try { localStorage.setItem(KEY, JSON.stringify(store)); } catch (e) {} }

const $ = id => document.getElementById(id);
const clip = $("clip");
let order = [], idx = 0, rate = 1;

const v = g => store.verdicts[g] || {};
function visible() {
  const f = $("filter").value;
  return SIGNS.map((s, i) => i).filter(i => {
    const vd = v(SIGNS[i].glose).verdict;
    return f === "all" || (f === "todo" ? !vd : vd === f);
  });
}
function stats() {
  const c = { identique: 0, variante: 0, "différent": 0, incertain: 0 };
  for (const s of SIGNS) { const vd = v(s.glose).verdict; if (vd in c) c[vd]++; }
  const n = Object.values(c).reduce((a, b) => a + b, 0);
  $("stats").innerHTML = `<b>${n}</b> / ${SIGNS.length} · ` +
    `<span class="v-identique">${c.identique} identiques</span> · <span class="v-variante">${c.variante} variantes</span> · ` +
    `<span class="v-différent">${c["différent"]} différents</span> · <span class="v-incertain">${c.incertain} incertains</span>`;
  const due = store.sinceExport >= EXPORT_EVERY;
  $("exportBtn").classList.toggle("due", due);
  $("exportBtn").textContent = due ? `Exporter le CSV (${store.sinceExport} réponses non exportées)` : "Exporter le CSV";
}
function show() {
  order = visible();
  if (!order.length) {
    $("gloss").textContent = "Rien à afficher";
    $("forms").textContent = $("filter").value === "todo" ? "Toutes les gloses sont étiquetées : pense à exporter le CSV." : "";
    for (const id of ["meta", "warn", "current", "pos"]) $(id).textContent = "";
    clip.removeAttribute("src"); stats(); return;
  }
  idx = Math.max(0, Math.min(idx, order.length - 1));
  const s = SIGNS[order[idx]], cur = v(s.glose);
  $("gloss").textContent = s.glose;
  $("forms").textContent = s.membres.length > 1 ? `Même verdict pour : ${s.membres.join(", ")}` : "";
  clip.src = s.video; clip.playbackRate = rate; clip.play().catch(() => {});
  const bits = [`<b>${s.frequence}</b> occurrence(s) dans les datasets`, `étiquette LSFB : <b>${s.etiquette_lsfb}</b>`];
  if (s.autres) bits.push(`autres étiquettes LSFB : ${s.autres}`);
  $("meta").innerHTML = bits.join(" · ");
  const warns = [];
  if (s.note_mapping) warns.push(s.note_mapping);
  if (s.partage) warns.push(`même clip que ${s.partage} : vérifie que c'est bien le sens de « ${s.glose} »`);
  $("warn").textContent = warns.join(" · ");
  $("current").innerHTML = cur.verdict ? `— déjà répondu : <span class="v-${cur.verdict}">${cur.verdict}</span>` : "";
  document.querySelectorAll(".verdicts button").forEach(b => b.classList.toggle("on", b.dataset.v === cur.verdict));
  $("note").value = cur.note || "";
  $("pos").textContent = `${idx + 1} / ${order.length}`;
  stats();
}
function localNow() {   // heure locale, au format AAAA-MM-JJTHH:MM:SS
  const d = new Date(), p = n => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}
const current = () => order.length ? SIGNS[order[idx]] : null;
function setVerdict(vd) {
  const s = current(); if (!s) return;
  if (vd) store.verdicts[s.glose] = { verdict: vd, note: $("note").value.trim(), date: localNow() };
  else delete store.verdicts[s.glose];
  store.sinceExport++; save();
  if (!(vd && $("filter").value === "todo")) idx += vd ? 1 : 0;   // en « à étiqueter », la suivante prend la place
  show();
}
function move(d) { idx = Math.max(0, Math.min(idx + d, order.length - 1)); show(); }

document.querySelectorAll(".verdicts button").forEach(b => b.addEventListener("click", () => setVerdict(b.dataset.v)));
$("prev").onclick = () => move(-1);
$("next").onclick = () => move(1);
$("replay").onclick = () => { clip.currentTime = 0; clip.play(); };
$("slow").onclick = () => { rate = rate === 1 ? 0.5 : 1; clip.playbackRate = rate; $("speed").textContent = rate === 1 ? "Ralenti" : "Vitesse normale"; };
$("filter").onchange = () => { idx = 0; show(); };
$("note").addEventListener("input", () => {
  const s = current(); if (!s || !store.verdicts[s.glose]) return;
  store.verdicts[s.glose].note = $("note").value.trim(); save();
});
$("jump").addEventListener("change", () => {
  const g = $("jump").value.trim().toUpperCase(); $("jump").value = "";
  if (!SIGNS.some(s => s.glose === g)) return;
  $("filter").value = "all"; order = visible();
  idx = order.findIndex(i => SIGNS[i].glose === g); show(); $("jump").blur();
});
$("glosses").innerHTML = SIGNS.map(s => `<option value="${s.glose}">`).join("");
$("exportBtn").onclick = () => {
  const cols = ["glose", "etiquette_lsfb", "statut", "frequence", "verdict", "note", "date"];
  const esc = x => '"' + String(x ?? "").replace(/"/g, '""') + '"';
  // Une ligne par glose du projet : le verdict d'une carte vaut pour toutes ses formes.
  const lines = [cols.join(",")].concat(SIGNS.flatMap(s => {
    const cur = v(s.glose);
    return s.lignes.map(l => [l.glose, l.etiquette_lsfb, l.statut, l.frequence, cur.verdict, cur.note, cur.date]
      .map(esc).join(","));
  }));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob(["﻿" + lines.join("\n") + "\n"], { type: "text/csv" }));
  a.download = "lsfb_lsf_comparaison.csv"; a.click();
  store.sinceExport = 0; save(); stats();
};
document.addEventListener("keydown", e => {
  if (["TEXTAREA", "INPUT", "SELECT"].includes(document.activeElement.tagName)) {
    if (e.key === "Escape") document.activeElement.blur();
    return;
  }
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key.toLowerCase();
  const map = { "1": "identique", "2": "variante", "3": "différent", "4": "incertain" };
  if (map[k]) setVerdict(map[k]);
  else if (k === "0") setVerdict(null);
  else if (k === "arrowright") move(1);
  else if (k === "arrowleft") move(-1);
  else if (k === " ") { clip.currentTime = 0; clip.play(); }
  else if (k === "l") $("slow").click();
  else if (k === "n") $("note").focus();
  else return;
  e.preventDefault();
});
show();
</script></body></html>
"""


def write_html(cards: list[dict]) -> None:
    payload = json.dumps(cards, ensure_ascii=False).replace("</", "<\\/")
    HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
    HTML_PATH.write_text(PAGE.replace("__SIGNS__", payload), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ouvrir", action="store_true", help="ouvrir la page dans le navigateur")
    args = ap.parse_args()
    if not MANIFEST.exists():
        sys.exit("Aucune vidéo LSFB : lance d'abord lsfb_subset.py download --videos (voir l'aide).")
    rows = load_rows()
    failed = to_h264(rows)
    if failed:
        print(f"⚠️  {failed} clip(s) non convertis en H.264 : ils risquent de ne pas se lire.")
    old = read_csv()
    write_csv(rows, old)
    cards = group_cards(rows, old)
    write_html(cards)
    done = Counter(c["verdict"] or "à étiqueter" for c in cards)
    print(f"{len(rows)} gloses avec clip LSFB, {len(cards)} signes à juger (directionnels regroupés) ; "
          + ", ".join(f"{k} {n}" for k, n in done.most_common()))
    print(f"CSV  : {CSV_PATH.relative_to(SRC.parent)}")
    print(f"Page : {HTML_PATH.relative_to(SRC.parent)}")
    if args.ouvrir:
        opener = {"darwin": ["open"], "win32": ["cmd", "/c", "start", ""]}.get(sys.platform, ["xdg-open"])
        subprocess.run([*opener, str(HTML_PATH)], check=False)


if __name__ == "__main__":
    main()
