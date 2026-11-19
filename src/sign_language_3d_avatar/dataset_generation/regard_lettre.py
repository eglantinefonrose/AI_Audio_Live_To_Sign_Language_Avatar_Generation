"""Page pour regarder une lettre de l'alphabet manuel de près (par défaut E, la lettre qui
fait échouer le plus de phrases à l'évaluation).

Sur une seule page :
1. l'avatar (gros plan et en pied), la vidéo de référence filmée (`data/alphabet/`) et le
   clip LSFB dont l'avatar garde le bras ;
2. tous les exemples de la lettre dans LSFB-ISOL, avec la moitié des signeurs à laquelle
   chacun appartient : la moitié « eval » forme la galerie à laquelle l'évaluation compare
   l'avatar, la moitié « choix » sert à choisir le clip copié (`features.signer_half`) ;
3. les phrases du banc où la lettre fait échouer la phrase (dernier record d'évaluation du
   mannequin), avec le rang de la lettre dans chaque mot ; pour celles où elle est seule en
   cause, l'avatar épelle le mot en gros plan (rendu Blender, `demo_phrase_blender.py`).

Rien à exporter : la page sert à regarder. La page et les vidéos sont dans `data/lsfb/`
(ignoré par git : clips CC BY-NC-SA 4.0).

Usage :
    uv run python src/sign_language_3d_avatar/dataset_generation/regard_lettre.py [E] [--sans-rendu] [--ouvrir]
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from revue_clips import latest_record, videos  # noqa: E402
from lsf_comparaison import DATA, GEN, SRC  # noqa: E402
from features import signer_half  # noqa: E402  (evaluation/, ajouté au chemin par revue_clips)

BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"
DEMO = GEN.parent / "avatar_animation" / "demo_phrase_blender.py"
ALPHABET = GEN / "data" / "alphabet"
RENDUS = DATA / "avatar_lettres"
SELECTION = GEN / "selection.csv"
MAX_MOTS = 6                       # mots rendus (phrases où la lettre est seule en cause)

PAGE = r"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITRE__</title>
<style>
:root { --bg:#f6f5f2; --card:#fff; --ink:#1d1d1f; --mute:#6b6b70; --line:#e2e0da; --accent:#2457c5;
        --bad:#b3261e; --ok:#1e6b3a; }
@media (prefers-color-scheme: dark) { :root { --bg:#17171a; --card:#222226; --ink:#ececf0; --mute:#a0a0a8;
        --line:#34343a; --accent:#8ab4ff; --bad:#ff8a80; --ok:#81c995; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 system-ui, sans-serif; }
header { position:sticky; top:0; z-index:1; background:var(--bg); display:flex; flex-wrap:wrap; gap:8px 16px;
         align-items:center; padding:10px 16px; border-bottom:1px solid var(--line); }
header h1 { font-size:17px; margin:0; margin-right:auto; }
button { font:inherit; font-size:14px; border-radius:6px; border:1px solid var(--line); background:var(--card);
         color:var(--ink); padding:5px 10px; cursor:pointer; }
main { max-width:1200px; margin:0 auto; padding:16px; display:flex; flex-direction:column; gap:16px; }
section { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px;
          display:flex; flex-direction:column; gap:10px; }
h2 { font-size:18px; margin:0; }
.lead { color:var(--mute); font-size:14px; margin:0; }
.grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:10px; }
figure { margin:0; display:flex; flex-direction:column; gap:4px; }
figure video { width:100%; aspect-ratio:1; object-fit:contain; border-radius:8px; background:#000; }
figcaption { font-size:13px; color:var(--mute); }
figcaption b { color:var(--ink); }
.tag { display:inline-block; font-size:12px; border-radius:4px; padding:0 6px; margin-right:4px;
       border:1px solid var(--line); }
.tag.eval { border-color:var(--accent); color:var(--accent); }
.tag.copie { border-color:var(--ok); color:var(--ok); }
table { border-collapse:collapse; width:100%; font-size:14px; }
td, th { text-align:left; padding:6px 8px; border-top:1px solid var(--line); vertical-align:top; }
th { color:var(--mute); font-weight:500; }
.seul { color:var(--bad); font-weight:600; }
kbd { font:12px ui-monospace, monospace; border:1px solid var(--line); border-bottom-width:2px; border-radius:4px;
      padding:0 5px; color:var(--mute); }
</style></head><body>
<header><h1>__TITRE__</h1><button id="slow">Ralenti <kbd>L</kbd></button></header>
<main id="main"></main>
<script>
const D = __DONNEES__;
const esc = t => String(t).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fig = (src, cap) => `<figure><video autoplay loop muted playsinline preload="auto" src="${src}"></video>` +
                          `<figcaption>${cap}</figcaption></figure>`;
const L = D.lettre;
let html = `<section><h2>1. L'avatar et votre vidéo</h2>
  <p class="lead">L'avatar prend ${D.source ? `les doigts de <b>${esc(D.source)}</b>` : "la main du clip LSFB"}
  ${D.orientation ? "et l'orientation de votre vidéo" : "et l'orientation du clip LSFB"} ; le bras et la position viennent du clip LSFB.
  Dans un mot épelé, il ne joue que le milieu de la lettre (0,4 s)${D.mouvement ? ", sauf ici : la lettre est jouée en entier (vidéo entière)" : ""}.</p>
  <div class="grid">${fig(D.avatar_main, "<b>Avatar</b> — gros plan")}${fig(D.avatar_pied, "<b>Avatar</b> — en pied")}
  ${D.source_src ? fig(D.source_src, `<b>Votre vidéo</b> (${esc(D.source)})`) : ""}
  ${D.copie_src ? fig(D.copie_src, `<b>Clip LSFB copié</b> (signeur ${esc(D.copie_signeur)}) : bras et position`) : ""}</div></section>`;
html += `<section><h2>2. Les ${L} du dataset LSFB (${D.clips.length} exemples en tout)</h2>
  <p class="lead">L'évaluation reconnaît la lettre de l'avatar en la comparant aux exemples des signeurs
  <span class="tag eval">évaluation</span> : si votre ${L} est juste mais diffère de ces exemples-là, il sera mal reconnu.
  Les autres <span class="tag">choix</span> servent à choisir le clip copié.</p>
  <div class="grid">${D.clips.map(c => fig(c.video,
    `<span class="tag ${c.moitie === "eval" ? "eval" : ""}">${c.moitie === "eval" ? "évaluation" : "choix"}</span>` +
    (c.copie ? `<span class="tag copie">copié par l'avatar</span>` : "") + ` signeur ${esc(c.signeur)}`)).join("")}</div></section>`;
html += `<section><h2>3. Phrases où ${L} fait échouer la phrase (${D.phrases.length})</h2>
  <p class="lead">Rang de la lettre parmi les ${D.galerie} signes de la galerie (1 = reconnue du premier coup) ;
  une phrase échoue quand un signe est hors des 20 % les mieux classés.
  <span class="seul">En rouge</span> : ${L} est le seul signe qui fait échouer la phrase.</p>
  ${D.mots.length ? `<div class="grid">${D.mots.map(m => fig(m.video, `<b>${esc(m.mot)}</b> épelé par l'avatar (gros plan)`)).join("")}</div>` : ""}
  <table><tr><th>Phrase</th><th>Mot épelé</th><th>Rang de ${L}</th></tr>
  ${D.phrases.map(p => `<tr><td class="${p.seul ? "seul" : ""}">${esc(p.phrase)}</td><td>${esc(p.mots.join(", "))}</td>` +
    `<td>${esc(p.rangs.join(", "))}</td></tr>`).join("")}</table></section>`;
document.getElementById("main").innerHTML = html;
let rate = 1;
const slow = () => { rate = rate === 1 ? 0.25 : 1; document.querySelectorAll("video").forEach(v => v.playbackRate = rate);
  document.getElementById("slow").firstChild.textContent = rate === 1 ? "Ralenti " : "Vitesse normale "; };
document.getElementById("slow").onclick = slow;
document.addEventListener("keydown", e => { if (e.key.toLowerCase() === "l" && !e.metaKey && !e.ctrlKey) slow(); });
</script></body></html>
"""


def stamp(path: Path) -> str:
    """Chemin relatif à la page, avec la date du fichier (le navigateur ne garde pas l'ancien)."""
    return f"{path.relative_to(DATA)}?v={int(path.stat().st_mtime)}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lettre", nargs="?", default="E")
    ap.add_argument("--sans-rendu", action="store_true", help="ne pas refaire les mots épelés (Blender)")
    ap.add_argument("--ouvrir", action="store_true", help="ouvrir la page dans le navigateur")
    args = ap.parse_args()
    letter = args.lettre.upper()
    label = f"FS:{letter}"
    out_dir = DATA / f"lettre_{letter}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(SELECTION, newline="", encoding="utf-8") as f:
        copied = {r["etiquette_lsfb"]: r for r in csv.DictReader(f)}.get(label, {})
    with open(DATA / "instances.csv", newline="", encoding="utf-8") as f:
        clips = [r for r in csv.DictReader(f) if r["sign"] == label]
    ready = videos([r["id"] for r in clips])

    ref = ALPHABET / f"{letter}.json"
    d = json.loads(ref.read_text(encoding="utf-8")) if ref.exists() else {}
    source = d.get("image", "")

    rec = json.loads(latest_record().read_text(encoding="utf-8"))
    galerie = rec.get("metrics", {}).get("galerie", {}).get("signes") or len(json.loads(
        (GEN.parent / "avatar_animation" / "data" / "dictionnaire" / "index.json").read_text(encoding="utf-8"))["signes"])
    phrases = []
    for p in rec["par_phrase"]:
        if p["absents"]:
            continue
        mine = [s for s in p["signes"] if s["etiquette"] == label and s["centile"] is not None]
        if not any(s["centile"] < 0.8 for s in mine):
            continue
        others = [s for s in p["signes"] if s["etiquette"] != label and s["centile"] is not None]
        phrases.append({"phrase": " ".join(p["gloses"]), "seul": all(s["centile"] >= 0.8 for s in others),
                        "mots": list(dict.fromkeys(s["glose"] for s in mine)),
                        "rangs": [str(s["rang"]) for s in mine]})
    phrases.sort(key=lambda p: not p["seul"])
    words = list(dict.fromkeys(w for p in phrases if p["seul"] for w in p["mots"]))[:MAX_MOTS]
    if not args.sans_rendu and words:
        print(f"Rendu de {len(words)} mot(s) épelé(s) : {', '.join(words)}…", flush=True)
        for w in words:
            subprocess.run([BLENDER, "-b", "--factory-startup", "--python", str(DEMO), "--", "--gloses", w,
                            "--sortie", str(out_dir / f"{w}.mp4"), "--angle", "15", "--gros-plan"],
                           check=True, stdout=subprocess.DEVNULL)
    mots = [{"mot": w, "video": stamp(out_dir / f"{w}.mp4")} for w in words if (out_dir / f"{w}.mp4").exists()]

    data = {"lettre": letter, "source": source,
            "source_src": f"../alphabet/{source}?v={int((ALPHABET / source).stat().st_mtime)}" if source else "",
            "orientation": d.get("orientation") == "video", "mouvement": "mouvement" in d,
            "avatar_main": stamp(RENDUS / f"{letter}_main.mp4"), "avatar_pied": stamp(RENDUS / f"{letter}.mp4"),
            "copie_src": ready.get(copied.get("instance", ""), ""), "copie_signeur": copied.get("signeur", ""),
            "clips": [{"video": ready[r["id"]], "signeur": r["signer"], "moitie": signer_half(r["signer"]),
                       "copie": r["id"] == copied.get("instance")} for r in clips if r["id"] in ready],
            "galerie": galerie, "phrases": phrases, "mots": mots}
    page = DATA / f"lettre_{letter}.html"
    page.write_text(PAGE.replace("__TITRE__", f"Lettre {letter} de près")
                    .replace("__DONNEES__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")), encoding="utf-8")
    n_eval = sum(c["moitie"] == "eval" for c in data["clips"])
    print(f"Lettre {letter} : {len(data['clips'])} exemples LSFB ({n_eval} dans la galerie d'évaluation), "
          f"{len(phrases)} phrase(s) qui échouent à cause d'elle ({sum(p['seul'] for p in phrases)} seule en cause)")
    print(f"Page : {page.relative_to(SRC.parent)}")
    if args.ouvrir:
        opener = {"darwin": ["open"], "win32": ["cmd", "/c", "start", ""]}.get(sys.platform, ["xdg-open"])
        subprocess.run([*opener, str(page)], check=False)


if __name__ == "__main__":
    main()
