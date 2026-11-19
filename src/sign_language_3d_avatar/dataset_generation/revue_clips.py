"""Page de relecture des clips : pour les signes qui font échouer le plus de phrases du banc,
choisir à l'œil, parmi tous les exemples LSFB-ISOL téléchargés, celui que l'avatar copiera.

Pourquoi à l'œil : le choix automatique (`select_examples.py`) repose sur un reconnaisseur
simple qui se laisse piéger (deux clips atypiques qui se ressemblent se classent 1ers l'un
grâce à l'autre). Le corriger avec les scores de l'évaluation reviendrait à régler sur
l'évaluation ; un choix humain n'utilise pas ces scores. La page n'en montre donc aucun :
seulement les phrases que le signe fait échouer (pour savoir quoi relire), le rendu actuel de
l'avatar dans ces phrases (`demo_phrase_blender.py`, `--sans-rendu` pour s'en passer) et le clip
actuel. Les exemples des autres variantes LSFB du signe (`QUAND.1` pour `QUAND.8`…) sont montrés
aussi : en choisir un remplace la variante pour toutes les gloses du projet qui utilisaient le
signe (colonne `remplace` de `choix_clips.csv`, reprise par `lsfb_subset.py map`).

Signes relus : ceux qui font échouer au moins une phrase jouable au seuil 80 %, dans le
dernier record d'évaluation du dictionnaire actuel (`evaluation/results/*__mannequin.json`,
ou `--record`), les
plus bloquants d'abord (`--max`, 20 par défaut). Les lettres de l'alphabet manuel (`FS:N`…)
n'y figurent pas : elles servent à épeler les termes techniques et noms propres. Elles se
relisent à part avec `--lettres` : les lettres qui font échouer des phrases (N, P, H…),
pour choisir la forme de main la plus nette. Les signes jugés corrects à l'œil
(`evaluation/signes_verifies.csv`) n'y figurent pas non plus. L'export garde aussi les réponses déjà dans
`choix_clips.csv` pour les signes absents de la page.

Avec `--clips-a-changer` : les signes de la liste « Clips sources à changer » du rapport
d'évaluation (clip copié mal reconnu alors qu'un clip d'un autre signeur l'est bien). Ils ne
font pas forcément échouer de phrase : la page montre l'avatar dans les phrases du banc qui
les utilisent, ou le signe seul. Le clip bien reconnu n'est pas signalé : le choix reste à
l'œil, sans les scores (page `data/lsfb/revue_clips_a_changer.html`).

Réponses (clavier) : 1-9 ou clic = ce clip, G = garder le clip actuel, A = le clip est bon mais
l'avatar le joue mal (à décrire dans la note), V = aucun ne convient (autre signe à envisager,
voir la note), 0 = effacer. « Exporter le CSV » produit
`choix_clips.csv` (versionné, identifiants seulement), à copier dans `dataset_generation/` ;
`select_examples.py` reprend ensuite chaque clip choisi (critère « relecture »), après
`lsfb_subset.py map` si une variante a changé.

Les vidéos (CC BY-NC-SA 4.0) sont téléchargées dans `data/lsfb/videos/` puis copiées en
H.264 ; la page est `data/lsfb/revue_clips.html` (ignorée par git), les rendus dans `data/lsfb/revue_clips/`.

Usage :
    uv run python src/sign_language_3d_avatar/dataset_generation/revue_clips.py [--ouvrir] [--max 20] [--record chemin.json] [--lettres] [--sans-rendu]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import urllib.parse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lsf_comparaison import DATA, GEN, SRC, to_h264  # noqa: E402
from lsfb_subset import SOURCE, fetch  # noqa: E402

sys.path.insert(0, str(GEN.parent / "evaluation"))
from evaluate_avatar import INDEX, RESULTS, blocking_signs, sha, verified_signs  # noqa: E402
from clean_signs import is_letter  # noqa: E402

CSV_PATH = GEN / "choix_clips.csv"
HTML_PATH = DATA / "revue_clips.html"
HTML_LETTERS = DATA / "revue_lettres.html"
HTML_CHANGE = DATA / "revue_clips_a_changer.html"
SELECTION = GEN / "selection.csv"
RENDUS = DATA / "revue_clips"
BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"
DEMO = GEN.parent / "avatar_animation" / "demo_phrase_blender.py"
FIELDS = ["etiquette_lsfb", "instance", "verdict", "note", "date", "remplace"]
MAX_CLIPS = 20
MAX_VARIANTE = 10                  # exemples montrés par autre variante


def latest_record() -> Path:
    """Le plus récent record du mannequin qui a évalué le dictionnaire actuel."""
    current = sha(INDEX.read_bytes())
    for path in sorted(RESULTS.glob("*__mannequin.json"), reverse=True):
        if json.loads(path.read_text(encoding="utf-8"))["dictionnaire"] == current:
            return path
    sys.exit(f"Aucun record pour le dictionnaire actuel ({current}) : lance evaluate_avatar.py --save.")


def read_csv() -> dict[str, dict]:
    """Réponses par signe relu : une variante choisie est rangée sous le signe qu'elle remplace."""
    if not CSV_PATH.exists():
        return {}
    with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
        return {r.get("remplace") or r["etiquette_lsfb"]: r for r in csv.DictReader(f)}


def render(phrases: list[str]) -> dict[str, str]:
    """Rendu actuel de l'avatar pour chaque phrase (refait si le dictionnaire a changé) :
    phrase -> chemin relatif à la page, horodaté pour que le navigateur ne garde pas l'ancien."""
    RENDUS.mkdir(exist_ok=True)
    out = {p: RENDUS / f"{hashlib.md5(p.encode()).hexdigest()[:10]}.mp4" for p in phrases}
    todo = [p for p, f in out.items() if not f.exists() or f.stat().st_mtime < INDEX.stat().st_mtime]
    if todo:
        print(f"Rendu de l'avatar pour {len(todo)} phrase(s)…", flush=True)

        def run(p: str) -> None:
            subprocess.run([BLENDER, "-b", "--factory-startup", "--python", str(DEMO), "--", "--gloses", p,
                            "--sortie", str(out[p]), "--angle", "15"], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with ThreadPoolExecutor(2) as ex:
            list(ex.map(run, todo))
    return {p: f"{f.relative_to(DATA)}?v={int(f.stat().st_mtime)}" for p, f in out.items() if f.exists()}


def videos(ids: list[str]) -> dict[str, str]:
    """Clips téléchargés si besoin puis copiés en H.264 : identifiant -> chemin relatif à la page."""
    todo = [i for i in ids if not (DATA / "videos" / f"{i}.mp4").exists()]
    if todo:
        print(f"Téléchargement de {len(todo)} clips LSFB-ISOL (CC BY-NC-SA 4.0)…", flush=True)

        def get(i: str) -> None:
            try:
                fetch(f"{SOURCE}/{urllib.parse.quote(f'videos/{i}.mp4')}", DATA / "videos" / f"{i}.mp4")
            except Exception:  # noqa: BLE001 — clip indisponible : absent de la page
                pass
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(get, todo))
    rows = [{"video": f"videos/{i}.mp4"} for i in ids if (DATA / "videos" / f"{i}.mp4").exists()]
    if rows:
        to_h264(rows)
    return {Path(r["video"]).stem: r["video"] for r in rows if r["video"].startswith("videos_h264/")}


PAGE = r"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITRE__</title>
<style>
:root { --bg:#f6f5f2; --card:#fff; --ink:#1d1d1f; --mute:#6b6b70; --line:#e2e0da; --accent:#2457c5;
        --ok:#1e6b3a; --mid:#a15c00; --bad:#b3261e; }
@media (prefers-color-scheme: dark) { :root { --bg:#17171a; --card:#222226; --ink:#ececf0; --mute:#a0a0a8;
        --line:#34343a; --accent:#8ab4ff; --ok:#81c995; --mid:#ffb74d; --bad:#ff8a80; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.45 system-ui, sans-serif; }
header { position:sticky; top:0; z-index:1; background:var(--bg); display:flex; flex-wrap:wrap;
         gap:8px 16px; align-items:center; padding:10px 16px; border-bottom:1px solid var(--line); }
header h1 { font-size:17px; margin:0; }
#stats { color:var(--mute); font-size:14px; margin-right:auto; }
button, input { font:inherit; font-size:14px; border-radius:6px; border:1px solid var(--line);
         background:var(--card); color:var(--ink); padding:5px 10px; }
button { cursor:pointer; }
main { max-width:1200px; margin:0 auto; padding:16px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px;
        display:flex; flex-direction:column; gap:10px; }
#label { font-size:28px; font-weight:700; margin:0; }
.meta { font-size:13px; color:var(--mute); }
#phrases { margin:0; padding-left:18px; }
#grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(170px, 1fr)); gap:8px; }
.clip { border:1px solid var(--line); border-radius:8px; padding:4px; cursor:pointer; display:flex;
        flex-direction:column; gap:3px; }
.clip.current { border-color:var(--mid); border-width:2px; }
.clip.on { outline:3px solid var(--ok); border-color:var(--ok); }
.clip video { width:100%; aspect-ratio:4/3; object-fit:contain; border-radius:5px; background:#000; }
.clip .cap { font-size:12px; color:var(--mute); display:flex; justify-content:space-between; }
.group { grid-column:1/-1; font-size:13px; font-weight:600; margin-top:6px; }
#phrases { display:grid; grid-template-columns:repeat(auto-fill, minmax(260px, 1fr)); gap:10px; list-style:none; padding:0; }
#phrases video { display:block; width:100%; aspect-ratio:1; object-fit:contain; margin-top:4px; border-radius:6px; background:#000; }
.clip.current .cap b { color:var(--mid); }
.row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
.row button.on { outline:2px solid var(--accent); }
kbd { font:12px ui-monospace, monospace; border:1px solid var(--line); border-bottom-width:2px;
      border-radius:4px; padding:0 5px; color:var(--mute); }
textarea { width:100%; padding:6px; font:inherit; font-size:14px; border-radius:6px; border:1px solid var(--line);
           background:var(--bg); color:var(--ink); resize:vertical; }
.help { font-size:12px; color:var(--mute); }
.v-choisi, .v-garder { color:var(--ok); } .v-variante { color:var(--bad); }
</style></head><body>
<header>
  <h1>__TITRE__</h1>
  <span id="stats"></span>
  <button id="slow">Ralenti <kbd>L</kbd></button>
  <button id="exportBtn">Exporter le CSV</button>
</header>
<main><section class="card">
  <div class="row" style="justify-content:space-between"><h2 id="label"></h2><span id="pos" class="meta"></span></div>
  <div class="meta" id="info"></div>
  <div class="meta">__PHRASES__</div>
  <ul id="phrases" class="meta"></ul>
  <div><b>Quel exemple montre le mieux ce signe ?</b> Le plus net, le plus typique, celui qu'un signeur
    reconnaîtrait sans contexte (pour une lettre : la forme de main la plus lisible, tenue nettement). <span id="current"></span></div>
  <div id="grid"></div>
  <div class="row">
    <button data-v="garder">Garder le clip actuel <kbd>G</kbd></button>
    <button data-v="avatar">Le clip est bon, l'avatar le joue mal <kbd>A</kbd></button>
    <button data-v="variante">Aucun ne convient : autre signe <kbd>V</kbd></button>
    <button id="prev">← Précédent</button><button id="next">Suivant →</button>
  </div>
  <textarea id="note" rows="2" placeholder="Note (ex. : la main devrait monter plus haut ; MODELE.EXEMPLE serait le bon signe) — N pour écrire, Échap pour revenir"></textarea>
  <div class="help"><kbd>1</kbd>–<kbd>9</kbd> ou clic : ce clip (un clip d'une autre variante remplace la variante) ·
    <kbd>G</kbd> garder l'actuel (cadre orange) · <kbd>A</kbd> l'avatar joue mal le clip · <kbd>V</kbd> autre signe · <kbd>0</kbd> effacer · <kbd>←</kbd> <kbd>→</kbd> naviguer · <kbd>L</kbd> ralenti.
    Les réponses restent dans ce navigateur : exporte le CSV et copie-le dans <code>dataset_generation/choix_clips.csv</code>.</div>
</section></main>
<script>
const SIGNS = __SIGNS__;
const AUTRES = __AUTRES__;
const DEPUIS = "__DEPUIS__";  // réponses plus anciennes que l'évaluation : déjà essayées, le signe bloque encore   // réponses du CSV pour des signes absents de cette page, recopiées à l'export
const KEY = "lsfb-revue-clips";
let store = { answers: {} };
try { store = Object.assign(store, JSON.parse(localStorage.getItem(KEY) || "{}")); } catch (e) {}
for (const s of SIGNS) {           // le CSV du projet fait foi s'il est plus récent
  const local = store.answers[s.etiquette];
  if (s.reponse && (!local || (s.reponse.date || "") > (local.date || ""))) store.answers[s.etiquette] = s.reponse;
}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(store)); } catch (e) {} };
const $ = id => document.getElementById(id);
const esc = t => String(t).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
let idx = 0, rate = 1;
const ans = l => store.answers[l] || {};
function stats() {
  const n = SIGNS.filter(s => ans(s.etiquette).verdict && (ans(s.etiquette).date || "") > DEPUIS).length;
  $("stats").innerHTML = `<b>${n}</b> / ${SIGNS.length} signes relus depuis l'évaluation du ${DEPUIS.slice(0, 10)}`;
}
function show() {
  const s = SIGNS[idx], a = ans(s.etiquette);
  $("label").textContent = s.nom || s.etiquette;
  $("pos").textContent = `${idx + 1} / ${SIGNS.length}`;
  $("info").innerHTML = (s.nom ? `étiquette LSFB ${esc(s.etiquette)} · ` : "") + `${s.nom ? "mots épelés avec cette lettre" : "gloses du projet"} : <b>${esc(s.gloses)}</b> · ` +
    (s.changer ? `${s.phrases.length} phrase(s) du banc` : `fait échouer <b>${s.phrases.length}</b> phrase(s), dont ${s.seul} à lui seul`) +
    ` · ${s.clips.length} exemple(s)` +
    (s.limite ? ` · <b class="v-variante">mesure peu fiable</b> : aucun clip réel de ce signe n'est reconnu non plus par l'évaluation` : "") +
    (s.variantes.length ? ` · autres variantes LSFB : ${s.variantes.map(esc).join(", ")}` : "");
  $("phrases").innerHTML = s.phrases.map(p => `<li>${esc(p.texte)}${p.video ?
    `<video autoplay loop muted playsinline preload="auto" src="${p.video}"></video>` : ""}</li>`).join("");
  $("grid").innerHTML = s.clips.map((c, k) => (k === 0 || c.etiquette !== s.clips[k - 1].etiquette ?
      `<div class="group">${c.etiquette === s.etiquette ? `Exemples de ${esc(s.etiquette)} (signe actuel)`
        : `Autre variante : ${esc(c.etiquette)}`}</div>` : "") + `
    <div class="clip${c.id === s.actuel ? " current" : ""}${a.verdict === "choisi" && a.instance === c.id ? " on" : ""}" data-k="${k}">
      <video autoplay loop muted playsinline preload="auto" src="${c.video}"></video>
      <div class="cap"><span>${k < 9 ? `<kbd>${k + 1}</kbd> ` : ""}${esc(c.signeur)}</span>${c.id === s.actuel ? "<b>actuel</b>" : ""}</div>
    </div>`).join("");
  document.querySelectorAll(".clip").forEach(el => {
    el.querySelector("video").playbackRate = rate;
    el.onclick = () => pick(s.clips[+el.dataset.k].id);
  });
  const lab = a.verdict === "choisi" ? `clip de ${esc((s.clips.find(c => c.id === a.instance) || {}).signeur || a.instance)}`
              + (a.etiquette && a.etiquette !== s.etiquette ? ` (variante ${esc(a.etiquette)})` : "")
            : a.verdict === "garder" ? "garder l'actuel" : a.verdict === "avatar" ? "l'avatar joue mal le clip"
            : a.verdict === "variante" ? "autre signe" : "";
  const old = a.verdict && (a.date || "") <= DEPUIS;
  $("current").innerHTML = a.verdict ? `— ${old ? "réponse précédente (le signe bloque encore)" : "réponse"} : <span class="v-${a.verdict}">${lab}</span>` : "";
  document.querySelectorAll("[data-v]").forEach(b => b.classList.toggle("on", b.dataset.v === a.verdict));
  $("note").value = a.note || "";
  stats();
}
function localNow() {
  const d = new Date(), p = n => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}
function answer(verdict, instance = "", etiquette = "") {
  const s = SIGNS[idx];
  // garder / avatar / variante : le clip actuel reste celui que l'avatar copie
  if (verdict && !instance) instance = s.actuel || "";
  if (verdict) store.answers[s.etiquette] = { verdict, instance, etiquette, note: $("note").value.trim(), date: localNow() };
  else delete store.answers[s.etiquette];
  save();
  if (verdict && idx < SIGNS.length - 1) idx++;
  show();
}
const pick = id => {
  const s = SIGNS[idx], c = s.clips.find(c => c.id === id);
  if (id === s.actuel) answer("garder");
  else answer("choisi", id, c.etiquette === s.etiquette ? "" : c.etiquette);
};
const move = d => { idx = Math.max(0, Math.min(idx + d, SIGNS.length - 1)); show(); };
document.querySelectorAll("[data-v]").forEach(b => b.onclick = () => answer(b.dataset.v));
$("prev").onclick = () => move(-1);
$("next").onclick = () => move(1);
$("slow").onclick = () => { rate = rate === 1 ? 0.5 : 1; document.querySelectorAll("video").forEach(v => v.playbackRate = rate);
  $("slow").firstChild.textContent = rate === 1 ? "Ralenti " : "Vitesse normale "; };
$("note").addEventListener("input", () => {
  const a = store.answers[SIGNS[idx].etiquette]; if (a) { a.note = $("note").value.trim(); save(); }
});
$("exportBtn").onclick = () => {
  const cols = ["etiquette_lsfb", "instance", "verdict", "note", "date", "remplace"];
  const q = x => '"' + String(x ?? "").replace(/"/g, '""') + '"';
  const lines = [cols.join(",")];
  for (const [lab, a] of Object.entries(store.answers))   // variante : rangée sous la nouvelle étiquette
    lines.push((a.etiquette ? [a.etiquette, a.instance, a.verdict, a.note, a.date, lab]
                            : [lab, a.instance, a.verdict, a.note, a.date, ""]).map(q).join(","));
  for (const r of AUTRES)
    if (!store.answers[r.remplace || r.etiquette_lsfb]) lines.push(cols.map(k => q(r[k])).join(","));
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob(["﻿" + lines.join("\n") + "\n"], { type: "text/csv" }));
  link.download = "choix_clips.csv"; link.click();
};
document.addEventListener("keydown", e => {
  if (["TEXTAREA", "INPUT"].includes(document.activeElement.tagName)) {
    if (e.key === "Escape") document.activeElement.blur();
    return;
  }
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key.toLowerCase(), s = SIGNS[idx];
  if (/^[1-9]$/.test(k) && s.clips[+k - 1]) pick(s.clips[+k - 1].id);
  else if (k === "g") answer("garder");
  else if (k === "v") answer("variante");
  else if (k === "a") answer("avatar");
  else if (k === "0") answer(null);
  else if (k === "arrowright") move(1);
  else if (k === "arrowleft") move(-1);
  else if (k === "l") $("slow").click();
  else if (k === "n") $("note").focus();
  else return;
  e.preventDefault();
});
show();
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", type=Path, help="record d'évaluation (par défaut : le plus récent du mannequin)")
    ap.add_argument("--max", type=int, default=20, help="nombre de signes à relire (les plus bloquants)")
    ap.add_argument("--ouvrir", action="store_true", help="ouvrir la page dans le navigateur")
    ap.add_argument("--sans-rendu", action="store_true", help="ne pas rendre les phrases avec l'avatar (Blender)")
    ap.add_argument("--clips-a-changer", action="store_true",
                    help="relire les signes de la liste « Clips sources à changer » du rapport d'évaluation")
    ap.add_argument("--lettres", action="store_true",
                    help="relire les lettres de l'alphabet manuel qui font échouer des phrases (FS:N…)")
    args = ap.parse_args()
    record = args.record or latest_record()
    rec = json.loads(record.read_text(encoding="utf-8"))
    verified = verified_signs(json.loads(INDEX.read_text(encoding="utf-8")))
    blocking = [b for b in blocking_signs(rec["par_phrase"], 0.8) if b["etiquette"] not in verified]
    # Hors de portée du reconnaisseur (aucun clip réel ne passe non plus) : en dernier, signalés.
    limit = {b["etiquette"] for b in rec["metrics"]["phrases"].get("signes_bloquants", {}).get("80%", []) if b.get("limite")}
    blocking.sort(key=lambda b: b["etiquette"] in limit)
    letters = [b["etiquette"] for b in blocking if is_letter(b["etiquette"])]
    worst = [b for b in blocking if is_letter(b["etiquette"]) == args.lettres][:args.max]
    if args.clips_a_changer:     # même critère que le rapport (evaluate_avatar.report)
        worst = [{"etiquette": s["etiquette"], "seul": 0} for s in sorted(
            (s for s in rec["signes"] if not is_letter(s["etiquette"]) and s["rang_clip_source"] is not None
             and s["rang_clip_source"] > 50 and (s["rang_autre_signeur"] or 999) <= 10 and not s.get("clip_relu")),
            key=lambda s: -s["rang_clip_source"])][:args.max]
    with open(SELECTION, newline="", encoding="utf-8") as f:
        selection = {r["etiquette_lsfb"]: r for r in csv.DictReader(f)}
    by_label: dict[str, list[dict]] = defaultdict(list)
    with open(DATA / "instances.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by_label[r["sign"]].append(r)
    old = read_csv()
    signs = []
    for w in worst:
        lab = w["etiquette"]
        clips = [r for r in sorted(by_label[lab], key=lambda r: r["id"])
                 if (DATA / "poses" / "pose" / f"{r['id']}.npy").exists()][:MAX_CLIPS]
        base = lab.split(".")[0]
        variants = [l for l, v in sorted(by_label.items()) if l != lab and l.split(".")[0] == base and len(v) >= 2]
        for v in variants:   # poses pas forcément téléchargées : `lsfb_subset.py` les prendra si la variante est choisie
            clips += sorted(by_label[v], key=lambda r: r["id"])[:MAX_VARIANTE]
        failing = [" ".join(p["gloses"]) + (f" — dans {', '.join(dict.fromkeys(s['glose'] for s in p['signes'] if s['etiquette'] == lab))}"
                                             if args.lettres else "")
                   for p in rec["par_phrase"] if not p["absents"]
                   and any(s["etiquette"] == lab and s["centile"] is not None
                           and (args.clips_a_changer or s["centile"] < 0.8) for s in p["signes"])]
        if args.clips_a_changer and not failing:      # absent du banc : le signe seul
            failing = selection.get(lab, {}).get("gloses_projet", lab).split()[:1]
        prev = old.get(lab)
        signs.append({"etiquette": lab, "limite": lab in limit, "changer": args.clips_a_changer, "gloses": " ".join(dict.fromkeys(selection.get(lab, {}).get("gloses_projet", "").split())),
                      # FS:H = la lettre H de l'alphabet manuel (« FS » : fingerspelling), pas F-S-H
                      "nom": f"Lettre {lab.split(':', 1)[1]} (une seule lettre de l'alphabet manuel)" if is_letter(lab) else "",
                      "actuel": selection.get(lab, {}).get("instance", ""), "seul": w["seul"], "phrases": failing,
                      "variantes": [f"{v} ({len(by_label[v])})" for v in variants],
                      "clips": [{"id": r["id"], "signeur": r["signer"], "etiquette": r["sign"]} for r in clips],
                      "reponse": {k: prev[k] for k in ("instance", "verdict", "note", "date")}
                      | {"etiquette": prev["etiquette_lsfb"] if prev.get("remplace") else ""} if prev else None})
    if not signs:
        sys.exit(f"Aucun signe à relire dans {record.name} : plus aucun ne fait échouer de phrase "
                 f"(hors lettres et signes jugés corrects à l'œil).")
    ready = videos([c["id"] for s in signs for c in s["clips"]])
    rendered = {} if args.sans_rendu or args.lettres else render(
        list(dict.fromkeys(p for s in signs for p in s["phrases"])))
    for s in signs:
        s["clips"] = [c | {"video": ready[c["id"]]} for c in s["clips"] if c["id"] in ready]
        s["phrases"] = [{"texte": p, "video": rendered.get(p, "")} for p in s["phrases"]]
    dump = json.dumps(signs, ensure_ascii=False).replace("</", "<\\/")
    others = [r for lab, r in old.items() if lab not in {s["etiquette"] for s in signs}]   # clés : signe relu
    path = HTML_LETTERS if args.lettres else HTML_CHANGE if args.clips_a_changer else HTML_PATH
    title = ("Relecture des lettres" if args.lettres else "Clips sources à revoir" if args.clips_a_changer
             else "Relecture des clips")
    heading = ("Phrases du banc qui utilisent ce signe (ou le signe seul), jouées par l'avatar avec le clip actuel :"
               if args.clips_a_changer else "Phrases du banc que ce signe fait échouer, jouées par l'avatar avec le clip actuel :")
    path.write_text(PAGE.replace("__SIGNS__", dump).replace("__AUTRES__", json.dumps(others, ensure_ascii=False).replace("</", "<\\/"))
                    .replace("__TITRE__", title).replace("__PHRASES__", heading).replace("__DEPUIS__", rec["timestamp"]), encoding="utf-8")
    what = "lettres" if args.lettres else "signes"
    print(f"{len(signs)} {what} à relire (record {record.name}), {sum(len(s['clips']) for s in signs)} clips"
          + (f" ; lettres épelées écartées : {', '.join(letters)}" if letters and not args.lettres else ""))
    print(f"Page : {path.relative_to(SRC.parent)} — export à copier dans {CSV_PATH.relative_to(SRC.parent)}")
    if args.ouvrir:
        opener = {"darwin": ["open"], "win32": ["cmd", "/c", "start", ""]}.get(sys.platform, ["xdg-open"])
        subprocess.run([*opener, str(path)], check=False)


if __name__ == "__main__":
    main()
