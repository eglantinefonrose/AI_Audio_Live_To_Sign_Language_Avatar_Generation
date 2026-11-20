"""Page de vérification des lettres : l'avatar signe chaque lettre de l'alphabet manuel, et
on dit à l'œil si la lettre est correcte.

Pour chaque lettre : l'avatar en pied et en gros plan sur la main (rendus par
`avatar_animation/demo_lettres_blender.py`, lettre jouée comme dans un mot épelé), le clip
LSFB-ISOL qu'il copie et, s'il y en a une, l'image de référence de `data/alphabet/` dont
viennent les doigts (`lettres_images.py`). Aucun score d'évaluation n'est montré.

Une réponse ne vaut que pour la main jugée : si la source des doigts d'une lettre change
(nouvelle image ou vidéo de référence), sa réponse est écartée et la lettre est à revoir ; si
seul le rendu change (réglage appliqué, vidéo entière…, colonne `rendu` : empreinte du
`.json` de la lettre), le verdict est écarté et la note gardée.

Réglage des doigts : quand la mesure se trompe, choisir pour chaque doigt « tendu »,
« plié » ou « crochet » (« mesuré » : tel que mesuré). Changer un réglage efface le verdict
de la lettre ; après export, `lettres_images.py` reconstruit ces doigts, puis on revérifie.
Lettre jouée d'après la vidéo entière : « rotation max » plafonne la rotation du poignet
(degrés, par rapport à la lettre tenue ; la rotation mesurée est affichée). Toute lettre :
« paume » impose la direction de la paume (vers le bas, le haut, l'avant ou soi).

Réponses (clavier) : C = correcte, D = douteuse, I = incorrecte, 0 = effacer, N = note.
« Exporter le CSV » produit `verif_lettres.csv` (lettre, verdict, note, image de référence
utilisée, état imposé de chaque doigt, date), à copier dans `dataset_generation/`.

La page et les vidéos sont dans `data/lsfb/` (ignoré par git : clips CC BY-NC-SA 4.0).

Usage :
    uv run python src/sign_language_3d_avatar/dataset_generation/verif_lettres.py [--lettres N H P] [--ouvrir]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import string
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lsf_comparaison import DATA, GEN, SRC  # noqa: E402
from revue_clips import videos  # noqa: E402

BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"
DEMO = GEN.parent / "avatar_animation" / "demo_lettres_blender.py"
RENDUS = DATA / "avatar_lettres"
ALPHABET = GEN / "data" / "alphabet"
HTML_PATH = DATA / "verif_lettres.html"
CSV_PATH = GEN / "verif_lettres.csv"
SELECTION = GEN / "selection.csv"
DOIGTS = ["pouce", "index", "majeur", "annulaire", "auriculaire"]
REGLAGES = ["rotation_max", "paume"]
FIELDS = ["lettre", "verdict", "note", "image_reference", *DOIGTS, *REGLAGES, "rendu", "date"]

PAGE = r"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vérification des lettres</title>
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
button { font:inherit; font-size:14px; border-radius:6px; border:1px solid var(--line);
         background:var(--card); color:var(--ink); padding:5px 10px; cursor:pointer; }
main { max-width:1200px; margin:0 auto; padding:16px; display:flex; flex-direction:column; gap:12px; }
#strip { display:flex; flex-wrap:wrap; gap:4px; }
#strip button { width:34px; padding:4px 0; font-weight:600; }
#strip button.here { outline:2px solid var(--accent); }
#strip .v-correcte { background:color-mix(in srgb, var(--ok) 22%, var(--card)); }
#strip .v-douteuse { background:color-mix(in srgb, var(--mid) 25%, var(--card)); }
#tuning input { width:150px; font:inherit; font-size:14px; border-radius:6px; border:1px solid var(--line);
         background:var(--card); color:var(--ink); padding:3px 6px; }
#fingers select, #tuning select { font:inherit; font-size:14px; border-radius:6px; border:1px solid var(--line);
         background:var(--card); color:var(--ink); padding:3px 6px; }
#fingers label, #tuning label { display:flex; flex-direction:column; font-size:12px; color:var(--mute); gap:2px; }
#fingers select.set, #tuning .set { border-color:var(--accent); outline:1px solid var(--accent); }
#pending { color:var(--mid); font-size:13px; }
#strip .v-incorrecte { background:color-mix(in srgb, var(--bad) 25%, var(--card)); }
.card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px;
        display:flex; flex-direction:column; gap:12px; }
#label { font-size:28px; font-weight:700; margin:0; }
.meta { font-size:13px; color:var(--mute); }
#views { display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:10px; }
figure { margin:0; display:flex; flex-direction:column; gap:4px; }
figure video, figure img { width:100%; aspect-ratio:1; object-fit:contain; border-radius:8px;
        background:#000; }
figure img { background:#fff; }
figcaption { font-size:12px; color:var(--mute); }
.row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
.row button.on { outline:2px solid var(--accent); }
[data-v="correcte"] { color:var(--ok); } [data-v="douteuse"] { color:var(--mid); } [data-v="incorrecte"] { color:var(--bad); }
kbd { font:12px ui-monospace, monospace; border:1px solid var(--line); border-bottom-width:2px;
      border-radius:4px; padding:0 5px; color:var(--mute); }
textarea { width:100%; padding:6px; font:inherit; font-size:14px; border-radius:6px; border:1px solid var(--line);
           background:var(--bg); color:var(--ink); resize:vertical; }
.help { font-size:12px; color:var(--mute); }
</style></head><body>
<header>
  <h1>Vérification des lettres</h1>
  <span id="stats"></span>
  <button id="slow">Ralenti <kbd>L</kbd></button>
  <button id="exportBtn">Exporter le CSV</button>
</header>
<main>
  <div id="strip"></div>
  <section class="card">
    <div class="row" style="justify-content:space-between"><h2 id="label"></h2><span id="pos" class="meta"></span></div>
    <div class="meta" id="info"></div>
    <div id="views"></div>
    <div><b>L'avatar signe-t-il correctement cette lettre ?</b> Regardez surtout la forme de la main
      (doigts tendus ou pliés, pouce) et son orientation, dans le gros plan.</div>
    <div class="row">
      <button data-v="correcte">Correcte <kbd>C</kbd></button>
      <button data-v="douteuse">Douteuse <kbd>D</kbd></button>
      <button data-v="incorrecte">Incorrecte <kbd>I</kbd></button>
      <button id="prev">← Précédente</button><button id="next">Suivante →</button>
    </div>
    <div><b>Un doigt est faux ?</b> Imposez son état (« mesuré » : tel que mesuré dans la vidéo ou l'image) :</div>
    <div class="row" id="fingers"></div>
    <div class="row" id="tuning"></div>
    <div id="pending"></div>
    <textarea id="note" rows="2" placeholder="Note (ex. : le pouce devrait être replié) — N pour écrire, Échap pour revenir"></textarea>
    <div class="help"><kbd>C</kbd> correcte · <kbd>D</kbd> douteuse · <kbd>I</kbd> incorrecte · <kbd>0</kbd> effacer ·
      <kbd>←</kbd> <kbd>→</kbd> naviguer · <kbd>L</kbd> ralenti. Les réponses restent dans ce navigateur :
      exportez le CSV et copiez-le dans <code>dataset_generation/verif_lettres.csv</code>.</div>
  </section>
</main>
<script>
const LETTERS = __LETTRES__;
const KEY = "lsfb-verif-lettres";
let store = { answers: {} };
try { store = Object.assign(store, JSON.parse(localStorage.getItem(KEY) || "{}")); } catch (e) {}
const DOIGTS = ["pouce", "index", "majeur", "annulaire", "auriculaire"];
const ETATS = [["", "mesuré"], ["tendu", "tendu"], ["plié", "plié"], ["crochet", "crochet"]];
const REGLAGES = ["rotation_max", "paume"];
const PAUMES = [["", "mesurée"], ["bas", "vers le bas"], ["haut", "vers le haut"], ["avant", "vers l'avant"], ["soi", "vers soi"]];
const sameFingers = (a, b) => DOIGTS.every(d => ((a || {})[d] || "") === ((b || {})[d] || ""));
const sameTuning = (a, b) => REGLAGES.every(k => String((a || {})[k] ?? "") === String((b || {})[k] ?? ""));
for (const l of LETTERS) {           // le CSV du projet fait foi s'il est au moins aussi récent
  let local = store.answers[l.lettre];
  if (local && (local.image_reference || "") !== (l.image || "")) { delete store.answers[l.lettre]; local = null; }  // main changée depuis
  if (l.reponse && (!local || (l.reponse.date || "") >= (local.date || ""))) store.answers[l.lettre] = local = l.reponse;
  if (local && local.verdict && !sameFingers(local.doigts, l.doigts)) delete local.verdict;  // jugé avec d'autres doigts
  if (local && local.verdict && !sameTuning(local.reglages, l.reglages)) delete local.verdict;
  if (local && local.verdict && (local.rendu || "") !== l.rendu) delete local.verdict;       // rendu refait depuis
}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(store)); } catch (e) {} };
const $ = id => document.getElementById(id);
const esc = t => String(t).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
let idx = 0, rate = 1;
const ans = l => store.answers[l] || {};
function strip() {
  $("strip").innerHTML = LETTERS.map((l, k) =>
    `<button class="${k === idx ? "here " : ""}v-${ans(l.lettre).verdict || ""}" data-k="${k}">${l.lettre}</button>`).join("");
  document.querySelectorAll("#strip button").forEach(b => b.onclick = () => { idx = +b.dataset.k; show(); });
  const n = LETTERS.filter(l => ans(l.lettre).verdict).length;
  const bad = LETTERS.filter(l => ans(l.lettre).verdict === "incorrecte").length;
  $("stats").innerHTML = `<b>${n}</b> / ${LETTERS.length} lettres vérifiées` + (bad ? ` · ${bad} incorrecte(s)` : "");
}
function show() {
  const l = LETTERS[idx], a = ans(l.lettre);
  $("label").textContent = `Lettre ${l.lettre}`;
  $("pos").textContent = `${idx + 1} / ${LETTERS.length}`;
  $("info").innerHTML = (l.image ? `doigts d'après l'image de référence <b>${esc(l.image)}</b>, placés comme dans le clip`
                                 : "forme de main copiée du clip LSFB")
    + (l.signeur ? ` · clip de ${esc(l.signeur)}` : "");
  const views = [[l.main, "Avatar — gros plan sur la main"], [l.pied, "Avatar — en pied"]];
  if (l.clip) views.push([l.clip, "Clip LSFB copié (vrai signeur)"]);
  $("views").innerHTML = views.map(([src, cap]) =>
      `<figure><video autoplay loop muted playsinline preload="auto" src="${src}"></video><figcaption>${cap}</figcaption></figure>`).join("")
    + (!l.image ? "" : /\.(mp4|mov|m4v)$/i.test(l.image)
      ? `<figure><video autoplay loop muted playsinline preload="auto" src="${l.image_src}"></video><figcaption>Votre vidéo de référence (${esc(l.image)})</figcaption></figure>`
      : `<figure><img src="${l.image_src}" alt="Image de référence ${l.lettre}"><figcaption>Image de référence</figcaption></figure>`);
  document.querySelectorAll("video").forEach(v => v.playbackRate = rate);
  document.querySelectorAll("[data-v]").forEach(b => b.classList.toggle("on", b.dataset.v === a.verdict));
  const want = a.doigts || {};
  $("fingers").innerHTML = DOIGTS.map(d => `<label>${d}<select data-d="${d}" class="${want[d] ? "set" : ""}">` +
    ETATS.map(([v, t]) => `<option value="${v}"${(want[d] || "") === v ? " selected" : ""}>${t}</option>`).join("") +
    `</select></label>`).join("");
  document.querySelectorAll("#fingers select").forEach(sel => sel.onchange = () => setFinger(sel.dataset.d, sel.value));
  const tune = a.reglages || {};
  $("tuning").innerHTML = (l.rotation_mesuree != null
      ? `<label>rotation max (°)<input type="number" min="0" max="180" step="5" data-k="rotation_max"
           class="${tune.rotation_max ? "set" : ""}" value="${esc(tune.rotation_max || "")}"
           placeholder="mesurée : ${l.rotation_mesuree}°"></label>` : "")
    + `<label>paume<select data-k="paume" class="${tune.paume ? "set" : ""}">` +
      PAUMES.map(([v, t]) => `<option value="${v}"${(tune.paume || "") === v ? " selected" : ""}>${t}</option>`).join("") +
      `</select></label>`;
  document.querySelectorAll("#tuning [data-k]").forEach(el => el.onchange = () => setTuning(el.dataset.k, el.value.trim()));
  const applied = DOIGTS.filter(d => (l.doigts || {})[d]).map(d => `${d} ${l.doigts[d]}`)
    .concat(l.reglages.rotation_max ? [`rotation max ${l.reglages.rotation_max}°`] : [])
    .concat(l.reglages.paume ? [`paume ${l.reglages.paume}`] : []);
  $("pending").innerHTML = (applied.length ? `Réglages déjà appliqués : ${esc(applied.join(", "))}. ` : "")
    + (l.rotation_mesuree != null ? `Lettre jouée d'après la vidéo entière (rotation mesurée : ${l.rotation_mesuree}°). ` : "")
    + (sameFingers(want, l.doigts) && sameTuning(tune, l.reglages) ? "" : "<b>Réglage pas encore appliqué</b> : exportez le CSV et envoyez-le pour que l'avatar soit refait.");
  $("note").value = a.note || "";
  strip();
}
function localNow() {
  const d = new Date(), p = n => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}
function setFinger(d, v) {
  const l = LETTERS[idx];
  const a = store.answers[l.lettre] || { image_reference: l.image || "", doigts: {} };
  a.doigts = Object.assign({}, a.doigts, { [d]: v });
  delete a.verdict;                  // la main va changer : à revoir une fois refaite
  a.note = $("note").value.trim(); a.date = localNow();
  store.answers[l.lettre] = a;
  save(); show();
}
function setTuning(k, v) {
  const l = LETTERS[idx];
  const a = store.answers[l.lettre] || { image_reference: l.image || "", doigts: {} };
  a.reglages = Object.assign({}, a.reglages, { [k]: v });
  delete a.verdict;
  a.note = $("note").value.trim(); a.date = localNow();
  store.answers[l.lettre] = a;
  save(); show();
}
function answer(verdict) {
  const l = LETTERS[idx];
  const doigts = (store.answers[l.lettre] || {}).doigts || {};
  const reglages = (store.answers[l.lettre] || {}).reglages || {};
  if (verdict) store.answers[l.lettre] = { verdict, note: $("note").value.trim(), image_reference: l.image || "", doigts,
                                           reglages, rendu: l.rendu, date: localNow() };
  else if (DOIGTS.some(d => doigts[d]) || REGLAGES.some(k => reglages[k]) || $("note").value.trim()) {
    if (store.answers[l.lettre]) delete store.answers[l.lettre].verdict;
  }
  else delete store.answers[l.lettre];
  save();
  if (verdict && idx < LETTERS.length - 1) idx++;
  show();
}
const move = d => { idx = Math.max(0, Math.min(idx + d, LETTERS.length - 1)); show(); };
document.querySelectorAll("[data-v]").forEach(b => b.onclick = () => answer(b.dataset.v));
$("prev").onclick = () => move(-1);
$("next").onclick = () => move(1);
$("slow").onclick = () => { rate = rate === 1 ? 0.25 : 1; document.querySelectorAll("video").forEach(v => v.playbackRate = rate);
  $("slow").firstChild.textContent = rate === 1 ? "Ralenti " : "Vitesse normale "; };
$("note").addEventListener("input", () => {
  const a = store.answers[LETTERS[idx].lettre]; if (a) { a.note = $("note").value.trim(); save(); }
});
$("exportBtn").onclick = () => {
  const cols = ["lettre", "verdict", "note", "image_reference", ...DOIGTS, ...REGLAGES, "rendu", "date"];
  const q = x => '"' + String(x ?? "").replace(/"/g, '""') + '"';
  const lines = [cols.join(",")];
  for (const l of LETTERS) {
    const a = store.answers[l.lettre];
    if (a) lines.push([l.lettre, a.verdict, a.note, a.image_reference, ...DOIGTS.map(d => (a.doigts || {})[d]),
                       ...REGLAGES.map(k => (a.reglages || {})[k]), a.verdict ? a.rendu : "", a.date].map(q).join(","));
  }
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob(["﻿" + lines.join("\n") + "\n"], { type: "text/csv" }));
  link.download = "verif_lettres.csv"; link.click();
};
document.addEventListener("keydown", e => {
  if (["TEXTAREA", "SELECT", "INPUT"].includes(document.activeElement.tagName)) {
    if (e.key === "Escape") document.activeElement.blur();
    return;
  }
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === "c") answer("correcte");
  else if (k === "d") answer("douteuse");
  else if (k === "i") answer("incorrecte");
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


def reference(letter: str) -> dict:
    """Ce que l'avatar joue pour la lettre, d'après `<LETTRE>.json` : image ou vidéo dont viennent
    les doigts ("" : clip LSFB seul), doigts imposés, réglages appliqués, rotation mesurée (vidéo
    entière) et empreinte du rendu."""
    path = ALPHABET / f"{letter}.json"
    if not path.exists():
        return {"image": "", "doigts": {}, "reglages": {}, "rotation_mesuree": None, "rendu": "clip"}
    d = json.loads(path.read_text(encoding="utf-8"))
    m = d.get("mouvement") or {}
    played = json.dumps([d["points"], m.get("points"), d.get("paume")]
                        + ([d["orientation"]] if d.get("orientation") else []))
    return {"image": d["image"], "doigts": d.get("doigts_imposes", {}),
            "reglages": {k: v for k, v in (("rotation_max", m.get("rotation_max")), ("paume", d.get("paume"))) if v},
            "rotation_mesuree": m.get("rotation_mesuree"),
            "rendu": hashlib.md5(played.encode()).hexdigest()[:10]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lettres", nargs="*", default=list(string.ascii_uppercase))
    ap.add_argument("--ouvrir", action="store_true", help="ouvrir la page dans le navigateur")
    args = ap.parse_args()
    letters = [l.upper() for l in args.lettres]

    print(f"Rendu de {len(letters)} lettre(s) sur l'avatar…", flush=True)
    subprocess.run([BLENDER, "-b", "--factory-startup", "--python", str(DEMO), "--",
                    "--sortie", str(RENDUS), "--lettres", *letters],
                   check=True, stdout=subprocess.DEVNULL)

    with open(SELECTION, newline="", encoding="utf-8") as f:
        selection = {r["etiquette_lsfb"]: r for r in csv.DictReader(f)}
    old = {}
    if CSV_PATH.exists():
        with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
            old = {r["lettre"]: r for r in csv.DictReader(f)}
    clips = videos([selection[f"FS:{l}"]["instance"] for l in letters if f"FS:{l}" in selection])

    page = []
    for l in letters:
        if not (RENDUS / f"{l}.mp4").exists():
            continue
        row = selection.get(f"FS:{l}", {})
        ref = reference(l)
        image = ref["image"]
        prev = old.get(l)
        if prev and prev.get("image_reference", "") != image:     # verdict sur une autre main : à revoir
            prev = None
        answer = None
        if prev:
            answer = {k: prev.get(k, "") for k in ("verdict", "note", "image_reference", "rendu", "date")}
            answer["doigts"] = {d: prev[d] for d in DOIGTS if prev.get(d)}
            answer["reglages"] = {k: prev[k] for k in REGLAGES if prev.get(k)}
        stamp = int((RENDUS / f"{l}_main.mp4").stat().st_mtime)      # le navigateur ne garde pas un ancien rendu
        page.append({"lettre": l, "pied": f"avatar_lettres/{l}.mp4?v={stamp}",
                     "main": f"avatar_lettres/{l}_main.mp4?v={stamp}",
                     "clip": clips.get(row.get("instance", ""), ""), "signeur": row.get("signeur", ""),
                     "image": image,
                     "image_src": f"../alphabet/{image}?v={int((ALPHABET / image).stat().st_mtime)}" if image else "",
                     "doigts": ref["doigts"], "reglages": ref["reglages"],
                     "rotation_mesuree": ref["rotation_mesuree"], "rendu": ref["rendu"], "reponse": answer})
    dump = json.dumps(page, ensure_ascii=False).replace("</", "<\\/")
    HTML_PATH.write_text(PAGE.replace("__LETTRES__", dump), encoding="utf-8")
    with_image = [p["lettre"] for p in page if p["image"]]
    print(f"{len(page)} lettres à vérifier"
          + (f" ; doigts d'après une image : {', '.join(with_image)}" if with_image else ""))
    print(f"Page : {HTML_PATH.relative_to(SRC.parent)} — export à copier dans {CSV_PATH.relative_to(SRC.parent)}")
    if args.ouvrir:
        opener = {"darwin": ["open"], "win32": ["cmd", "/c", "start", ""]}.get(sys.platform, ["xdg-open"])
        subprocess.run([*opener, str(HTML_PATH)], check=False)


if __name__ == "__main__":
    main()
