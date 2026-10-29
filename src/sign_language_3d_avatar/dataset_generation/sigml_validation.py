"""Prépare la relecture des signes SiGML par un locuteur LSF.

Lit tous les `.sigml` de `sigml/` (commentaire de chaque signe = description et
confiance), puis écrit :

- `sigml/validation_pilote.csv` : une ligne par signe, avec des colonnes vides
  `verdict` (ok / à corriger / faux), `correction` et `commentaire` à remplir ;
- `avatar_animation/data/sigml/validation.html` : page locale qui montre chaque
  vidéo à côté de sa description, avec le formulaire de verdict et un bouton
  « Exporter le CSV » (les réponses restent dans le navigateur entre deux visites).

Les vidéos doivent avoir été rendues avant (`avatar_animation/sigml_blender.py`).

    uv run python src/sign_language_3d_avatar/dataset_generation/sigml_validation.py [--ouvrir]
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent   # src/sign_language_3d_avatar/
SIGML_DIR = BASE / "dataset_generation" / "sigml"
VIDEOS = BASE / "avatar_animation" / "data" / "sigml"
CSV_PATH = SIGML_DIR / "validation_pilote.csv"
HTML_PATH = VIDEOS / "validation.html"
FIELDS = ["glose", "fichier", "description", "confiance", "video", "verdict", "correction", "commentaire"]


def read_signs() -> list[dict]:
    """Signes de tous les .sigml, avec le commentaire qui précède chacun."""
    rows = []
    for f in sorted(SIGML_DIR.glob("*.sigml")):
        parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
        root = ET.parse(f, parser).getroot()
        comment = ""
        for node in root:
            if node.tag is ET.Comment:
                comment = " ".join(node.text.split())
            elif node.tag == "hns_sign":
                gloss = node.get("gloss")
                m = re.search(r"Confiance\s*:\s*([^.;]+)", comment)
                description = re.sub(r"^[^:]+:\s*", "", comment.split("Confiance")[0]).strip(" .")
                rows.append({"glose": gloss, "fichier": f.name, "description": description,
                             "confiance": m.group(1).strip() if m else "non indiquée",
                             "video": f"{gloss}.mp4"})
                comment = ""
    return rows


def write_csv(rows: list[dict]) -> None:
    old = {}
    if CSV_PATH.exists():   # on garde les verdicts déjà saisis
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            old = {r["glose"]: r for r in csv.DictReader(f)}
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            prev = old.get(r["glose"], {})
            w.writerow({**r, **{k: prev.get(k, "") for k in ("verdict", "correction", "commentaire")}})


PAGE = """<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Validation des signes</title>
<style>
:root {{ --bg:#f6f5f2; --card:#fff; --ink:#1d1d1f; --mute:#6b6b70; --line:#e2e0da;
        --low:#b3261e; --mid:#a15c00; --high:#1e6b3a; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#17171a; --card:#222226; --ink:#ececf0;
        --mute:#a0a0a8; --line:#34343a; --low:#ff8a80; --mid:#ffb74d; --high:#81c995; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.45 system-ui, sans-serif; }}
header {{ position:sticky; top:0; background:var(--bg); border-bottom:1px solid var(--line);
         padding:12px 16px; display:flex; flex-wrap:wrap; gap:12px; align-items:center; z-index:1; }}
header h1 {{ font-size:18px; margin:0 auto 0 0; }}
button {{ font:inherit; padding:6px 14px; border-radius:6px; border:1px solid var(--line);
         background:var(--card); color:var(--ink); cursor:pointer; }}
.note {{ padding:10px 16px; color:var(--mute); max-width:900px; }}
main {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(300px, 1fr)); gap:16px; padding:16px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px;
        display:flex; flex-direction:column; gap:8px; }}
.card h2 {{ margin:0; font-size:17px; display:flex; justify-content:space-between; gap:8px; }}
.conf {{ font-size:12px; font-weight:600; }}
.conf.faible {{ color:var(--low); }} .conf.moyenne {{ color:var(--mid); }} .conf.bonne {{ color:var(--high); }}
video {{ width:100%; border-radius:6px; background:#ddd; }}
.desc {{ color:var(--mute); font-size:14px; }}
.verdict {{ display:flex; gap:12px; flex-wrap:wrap; font-size:14px; }}
textarea {{ width:100%; font:inherit; font-size:14px; padding:6px; border-radius:6px;
           border:1px solid var(--line); background:var(--bg); color:var(--ink); resize:vertical; }}
.file {{ font-size:12px; color:var(--mute); }}
</style></head><body>
<header><h1>Validation des signes ({n})</h1><span id="count"></span>
<button id="export">Exporter le CSV</button></header>
<p class="note">Brouillons SiGML rédigés par une IA et animés par un interprète maison (pas de visage).
Pour chaque signe : est-ce le bon signe LSF ? Si non, décrivez la correction (configuration,
emplacement, mouvement…). Les réponses sont gardées dans ce navigateur ; exportez le CSV à la fin.</p>
<main>{cards}</main>
<script>
const SIGNS = {signs};
const KEY = "validation-sigml";
let state = {{}};
try {{ state = JSON.parse(localStorage.getItem(KEY) || "{{}}"); }} catch (e) {{}}
for (const s of SIGNS) state[s.glose] = Object.assign(
  {{verdict: s.verdict, correction: s.correction, commentaire: s.commentaire}}, state[s.glose] || {{}});
function save() {{ try {{ localStorage.setItem(KEY, JSON.stringify(state)); }} catch (e) {{}} count(); }}
function count() {{
  const done = SIGNS.filter(s => state[s.glose].verdict).length;
  document.getElementById("count").textContent = done + " / " + SIGNS.length + " relus";
}}
document.querySelectorAll(".card").forEach(card => {{
  const g = card.dataset.glose, st = state[g];
  card.querySelectorAll("input[type=radio]").forEach(r => {{
    r.checked = r.value === st.verdict;
    r.addEventListener("change", () => {{ st.verdict = r.value; save(); }});
  }});
  card.querySelectorAll("textarea").forEach(t => {{
    t.value = st[t.name] || "";
    t.addEventListener("input", () => {{ st[t.name] = t.value; save(); }});
  }});
}});
document.getElementById("export").addEventListener("click", () => {{
  const cols = {fields};
  const esc = v => '"' + String(v ?? "").replace(/"/g, '""') + '"';
  const lines = [cols.join(",")].concat(SIGNS.map(s => cols.map(c =>
    esc(["verdict", "correction", "commentaire"].includes(c) ? state[s.glose][c] : s[c])).join(",")));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob(["\\ufeff" + lines.join("\\n")], {{type: "text/csv"}}));
  a.download = "validation_pilote.csv"; a.click();
}});
count();
</script></body></html>
"""

CARD = """<section class="card" data-glose="{g}">
<h2>{g} <span class="conf {cls}">confiance {conf}</span></h2>
{media}
<div class="desc">{desc}</div>
<div class="file">{file}</div>
<div class="verdict">
<label><input type="radio" name="v-{i}" value="ok"> correct</label>
<label><input type="radio" name="v-{i}" value="à corriger"> à corriger</label>
<label><input type="radio" name="v-{i}" value="faux"> faux / autre signe</label>
</div>
<textarea name="correction" rows="2" placeholder="Correction (forme, emplacement, mouvement…)"></textarea>
<textarea name="commentaire" rows="1" placeholder="Commentaire"></textarea>
</section>"""


def write_html(rows: list[dict]) -> int:
    old = {}
    if CSV_PATH.exists():
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            old = {r["glose"]: r for r in csv.DictReader(f)}
    cards, missing = [], 0
    for i, r in enumerate(rows):
        video = VIDEOS / r["video"]
        if video.exists():
            media = f'<video src="{html.escape(r["video"])}" controls loop muted playsinline preload="metadata"></video>'
        else:
            media, missing = "<p class=\"desc\">(vidéo non rendue)</p>", missing + 1
        cls = next((c for c in ("faible", "moyenne", "bonne") if c in r["confiance"]), "")
        cards.append(CARD.format(g=html.escape(r["glose"]), cls=cls, conf=html.escape(r["confiance"]),
                                 media=media, desc=html.escape(r["description"]),
                                 file=html.escape(r["fichier"]), i=i))
    signs = [{**r, **{k: old.get(r["glose"], {}).get(k, "") for k in ("verdict", "correction", "commentaire")}}
             for r in rows]
    HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
    HTML_PATH.write_text(PAGE.format(n=len(rows), cards="\n".join(cards),
                                     signs=json.dumps(signs, ensure_ascii=False),
                                     fields=json.dumps(FIELDS)), encoding="utf-8")
    return missing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ouvrir", action="store_true", help="ouvrir la page de validation dans le navigateur")
    args = ap.parse_args()
    rows = read_signs()
    write_csv(rows)
    missing = write_html(rows)
    conf = {}
    for r in rows:
        conf[r["confiance"]] = conf.get(r["confiance"], 0) + 1
    print(f"{len(rows)} signes ; confiance : " + ", ".join(f"{k} {v}" for k, v in sorted(conf.items())))
    print(f"CSV  : {CSV_PATH.relative_to(BASE.parent.parent)}")
    print(f"Page : {HTML_PATH.relative_to(BASE.parent.parent)}"
          + (f"  ({missing} vidéo(s) manquante(s) : lancer sigml_blender.py)" if missing else ""))
    if args.ouvrir:
        opener = {"darwin": ["open"], "win32": ["cmd", "/c", "start", ""]}.get(sys.platform, ["xdg-open"])
        subprocess.run([*opener, str(HTML_PATH)], check=False)


if __name__ == "__main__":
    main()
