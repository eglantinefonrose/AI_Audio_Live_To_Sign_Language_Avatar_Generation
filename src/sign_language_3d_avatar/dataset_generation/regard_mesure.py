"""Page pour comprendre pourquoi le reconnaisseur de l'évaluation échoue sur un signe : les
points MediaPipe dessinés sur les vidéos LSFB-ISOL.

Par défaut, les signes « hors de portée du reconnaisseur » du dernier record d'évaluation du
mannequin (aucun clip réel du signe n'est reconnu non plus, `evaluate_avatar.beyond_measure`),
ou les étiquettes LSFB données en argument. Pour chaque signe :
1. le clip copié par l'avatar ;
2. les exemples de référence : la moitié « eval » des signeurs, sans le signeur copié, à
   laquelle l'évaluation compare le signe (`features.signer_half`) ;
3. les clips de la galerie que le reconnaisseur trouve les plus proches du clip copié (autres
   signes), avec la distance DTW, pour voir ce qu'il croit reconnaître ;
4. quelques autres exemples du signe (moitié « choix », non utilisés par l'évaluation).

Sur chaque vidéo : corps et mains tels que les lit l'évaluation (`poses/`, trous bouchés).
Points pleins = détectés par MediaPipe sur cette image ; points rouges creux = main non
détectée sur cette image (`poses_raw/` vide), position inventée en bouchant le trou.

Rien à exporter : la page sert à regarder. La page et les vidéos sont dans `data/lsfb/`
(ignoré par git : clips CC BY-NC-SA 4.0).

Usage :
    uv run --with opencv-python python src/sign_language_3d_avatar/dataset_generation/regard_mesure.py [VOTE VOTER …] [--ouvrir]
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from revue_clips import latest_record, videos  # noqa: E402  (ajoute evaluation/ au chemin)
from lsf_comparaison import DATA, GEN, SRC  # noqa: E402
import evaluate_avatar as E  # noqa: E402
from features import dtw_batch, resample, signer_half, vector  # noqa: E402

OUT = DATA / "mesure_mediapipe"
HTML_PATH = DATA / "regard_mesure.html"
SELECTION = GEN / "selection.csv"
N_PROCHES = 3                      # signes les plus proches montrés
N_AUTRES = 6                       # autres exemples (moitié « choix ») montrés
FPS = 50

BODY = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]
HAND = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
COLORS = {"right_hand": (235, 140, 40), "left_hand": (60, 190, 60)}     # BGR : main droite bleue, gauche verte
MISSING = (40, 40, 230)


def load(part: str, kind: str, i: str) -> np.ndarray | None:
    p = DATA / kind / part / f"{i}.npy"
    return np.load(p).astype(np.float32) if p.exists() else None


def overlay(i: str) -> dict | None:
    """Vidéo H.264 du clip avec les points dessinés ; taux de détection des mains."""
    video = DATA / "videos" / f"{i}.mp4"
    pose = load("pose", "poses", i)
    if pose is None or not len(pose) or not video.exists():      # poses vides : clip sans détection
        return None
    hands = {h: load(h, "poses", i) for h in COLORS}
    raw = {h: load(h, "poses_raw", i) for h in COLORS}
    seen = {h: (~np.isnan(raw[h]).any(axis=(1, 2)) if raw[h] is not None else None) for h in COLORS}
    out = OUT / f"{i}.mp4"
    if not out.exists():
        OUT.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video))
        frames = []
        while True:
            ok, img = cap.read()
            if not ok:
                break
            frames.append(img)
        cap.release()
        if not frames:
            return None
        h, w = frames[0].shape[:2]
        tmp = out.with_suffix(".tmp")                # renommé à la fin : pas de vidéo tronquée en cache
        px = lambda p: (int(p[0] * w), int(p[1] * h))
        ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                               "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-vf", "scale=480:-2",
                               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-f", "mp4", str(tmp)],
                              stdin=subprocess.PIPE)
        for t, img in enumerate(frames):
            k = min(t, len(pose) - 1)
            for a, b in BODY:
                cv2.line(img, px(pose[k, a]), px(pose[k, b]), (230, 230, 230), 2, cv2.LINE_AA)
            for part, col in COLORS.items():
                pts = hands[part]
                if pts is None or np.isnan(pts[k]).any():
                    continue
                ok = seen[part] is None or seen[part][k]
                c = col if ok else MISSING
                for a, b in HAND:
                    cv2.line(img, px(pts[k, a]), px(pts[k, b]), c, 2 if ok else 1, cv2.LINE_AA)
                for p in pts[k]:
                    cv2.circle(img, px(p), 3, c, -1 if ok else 1, cv2.LINE_AA)
            cv2.putText(img, f"{t + 1}/{len(frames)}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            ff.stdin.write(img.tobytes())
        ff.stdin.close()
        if ff.wait() == 0:
            tmp.rename(out)
    if not out.exists():
        return None
    rate = lambda s: None if s is None else round(float(s.mean()), 2)
    return {"video": f"{out.relative_to(DATA)}", "images": len(pose),
            "droite": rate(seen["right_hand"]), "gauche": rate(seen["left_hand"])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("signes", nargs="*", help="étiquettes LSFB (par défaut : signes hors de portée du reconnaisseur)")
    ap.add_argument("--ouvrir", action="store_true", help="ouvrir la page dans le navigateur")
    args = ap.parse_args()

    record = latest_record()
    rec = json.loads(record.read_text(encoding="utf-8"))
    blocking = rec["metrics"]["phrases"].get("signes_bloquants", {}).get("80%", [])
    labels = args.signes or [b["etiquette"] for b in blocking if b.get("limite")]
    if not labels:
        sys.exit("Aucun signe hors de portée dans le dernier record : donne les étiquettes en argument.")
    per_sign = {s["etiquette"]: s for s in rec["signes"]}
    index = json.loads(E.INDEX.read_text(encoding="utf-8"))
    with open(SELECTION, newline="", encoding="utf-8") as f:
        selection = {r["etiquette_lsfb"]: r for r in csv.DictReader(f)}
    by_label: dict[str, list[dict]] = defaultdict(list)
    with open(DATA / "instances.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by_label[r["sign"]].append(r)
    # Même galerie que l'évaluation (moitié « eval », lettres comparées à part).
    gallery, _ = E.build_gallery(index, "eval", letters=rec.get("galerie", {}).get("lettres") == "lsfb")

    signs = []
    for lab in labels:
        if lab not in index["signes"]:
            print(f"  {lab} : absent du dictionnaire, ignoré")
            continue
        info, sel = index["signes"][lab], selection.get(lab, {})
        copied, signer = info["instance"], info["signeur"]
        own = np.flatnonzero(gallery.instances == copied)
        query = gallery.seqs[own[0]] if len(own) else resample(vector(E.clean(copied, False)[0])).astype(np.float32)
        keep = gallery.signers != signer
        d = dtw_batch(query, gallery.seqs[keep])
        g_lab, g_inst, g_sign = gallery.labels[keep], gallery.instances[keep], gallery.signers[keep]
        order = np.argsort(d)
        refs = [{"id": str(g_inst[k]), "signeur": str(g_sign[k]), "distance": round(float(d[k]), 3),
                 "place": int(np.flatnonzero(order == k)[0]) + 1} for k in np.flatnonzero(g_lab == lab)]
        near, labs_seen = [], set()
        for k in order:
            if g_lab[k] == lab or g_lab[k] in labs_seen:
                continue
            labs_seen.add(g_lab[k])
            near.append({"id": str(g_inst[k]), "signeur": str(g_sign[k]), "etiquette": str(g_lab[k]),
                         "gloses": selection.get(str(g_lab[k]), {}).get("gloses_projet", ""),
                         "distance": round(float(d[k]), 3)})
            if len(near) == N_PROCHES:
                break
        ref_ids = {r["id"] for r in refs}
        others = [{"id": r["id"], "signeur": r["signer"]} for r in sorted(by_label[lab], key=lambda r: r["id"])
                  if r["id"] != copied and r["id"] not in ref_ids][:N_AUTRES]
        phrases = [" ".join(p["gloses"]) for p in rec["par_phrase"]
                   if any(s["etiquette"] == lab and s["centile"] is not None and s["centile"] < 0.8 for s in p["signes"])]
        s = per_sign.get(lab, {})
        signs.append({"etiquette": lab, "gloses": sel.get("gloses_projet", ""), "phrases": phrases,
                      "rang_copie": s.get("rang_clip_source"), "rang_autre": s.get("rang_autre_signeur"),
                      "classes": len(set(g_lab)),
                      "copie": {"id": copied, "signeur": signer, "moitie": signer_half(signer)},
                      "references": sorted(refs, key=lambda r: r["distance"]), "proches": near,
                      "autres": [o | {"moitie": signer_half(o["signeur"])} for o in others]})

    ids = {c["id"] for s in signs for c in [s["copie"], *s["references"], *s["proches"], *s["autres"]]}
    videos(sorted(ids))                                 # télécharge les vidéos manquantes
    print(f"Dessin des points MediaPipe sur {len(ids)} clips…", flush=True)
    drawn = {i: overlay(i) for i in sorted(ids)}
    for s in signs:
        for c in [s["copie"], *s["references"], *s["proches"], *s["autres"]]:
            c |= drawn.get(c["id"]) or {}
    dump = json.dumps(signs, ensure_ascii=False).replace("</", "<\\/")
    HTML_PATH.write_text(PAGE.replace("__SIGNS__", dump).replace("__RECORD__", record.name), encoding="utf-8")
    print(f"{len(signs)} signe(s) : {', '.join(s['etiquette'] for s in signs)}")
    print(f"Page : {HTML_PATH.relative_to(SRC.parent)}")
    if args.ouvrir:
        opener = {"darwin": ["open"], "win32": ["cmd", "/c", "start", ""]}.get(sys.platform, ["xdg-open"])
        subprocess.run([*opener, str(HTML_PATH)], check=False)


PAGE = r"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Points MediaPipe</title>
<style>
:root { --bg:#f6f5f2; --card:#fff; --ink:#1d1d1f; --mute:#6b6b70; --line:#e2e0da; --accent:#2457c5;
        --ok:#1e6b3a; --mid:#a15c00; --bad:#b3261e; }
@media (prefers-color-scheme: dark) { :root { --bg:#17171a; --card:#222226; --ink:#ececf0; --mute:#a0a0a8;
        --line:#34343a; --accent:#8ab4ff; --ok:#81c995; --mid:#ffb74d; --bad:#ff8a80; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 system-ui, sans-serif; }
header { position:sticky; top:0; z-index:1; background:var(--bg); display:flex; flex-wrap:wrap; gap:8px 16px;
         align-items:center; padding:10px 16px; border-bottom:1px solid var(--line); }
header h1 { font-size:17px; margin:0; margin-right:auto; }
button, select { font:inherit; font-size:14px; border-radius:6px; border:1px solid var(--line); background:var(--card);
         color:var(--ink); padding:5px 10px; cursor:pointer; }
main { max-width:1200px; margin:0 auto; padding:16px; display:flex; flex-direction:column; gap:16px; }
section { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px;
          display:flex; flex-direction:column; gap:10px; }
h2 { font-size:22px; margin:0; }
h3 { font-size:15px; margin:6px 0 0; }
.meta, .help { font-size:13px; color:var(--mute); }
.legend { display:flex; flex-wrap:wrap; gap:6px 18px; font-size:13px; }
.dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:5px; vertical-align:middle; }
.grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(230px, 1fr)); gap:10px; }
.clip { border:1px solid var(--line); border-radius:8px; padding:4px; display:flex; flex-direction:column; gap:3px; }
.clip.copie { border:2px solid var(--mid); }
.clip video { width:100%; aspect-ratio:5/4; object-fit:contain; border-radius:5px; background:#000; }
.cap { font-size:12px; color:var(--mute); }
.cap b { color:var(--ink); }
.low { color:var(--bad); font-weight:600; }
</style></head><body>
<header>
  <h1>Points MediaPipe des signes mal reconnus</h1>
  <select id="pick"></select>
  <button id="slow">Ralenti</button>
</header>
<main>
<section>
  <div class="legend">
    <span><span class="dot" style="background:#ddd;border:1px solid #999"></span>corps</span>
    <span><span class="dot" style="background:#288cec"></span>main droite du signeur</span>
    <span><span class="dot" style="background:#3cbe3c"></span>main gauche du signeur</span>
    <span><span class="dot" style="border:2px solid #e62828"></span>main non détectée sur cette image : position inventée</span>
  </div>
  <div class="help">L'évaluation ne voit pas la vidéo, seulement ces points. Elle compare la suite des points du clip à
    chaque exemple de référence, et calcule une <b>distance</b> : plus elle est petite, plus les clips se ressemblent.
    Le signe est reconnu si un exemple de référence est parmi les plus proches de toute la galerie.
    Pour comprendre un échec, regardez : les points suivent-ils bien les mains ? les références font-elles le même
    geste que le clip copié ? et pourquoi les signes « les plus proches » ressemblent-ils à ce clip ?
    Record : __RECORD__.</div>
</section>
<div id="signs" style="display:contents"></div>
</main>
<script>
const SIGNS = __SIGNS__;
const esc = t => String(t ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pc = x => x == null ? "—" : `<span class="${x < 0.8 ? "low" : ""}">${Math.round(x * 100)} %</span>`;
function clip(c, cls, title) {
  if (!c.video) return `<div class="clip"><div class="cap">${esc(c.id)} : vidéo indisponible</div></div>`;
  return `<div class="clip ${cls}"><video autoplay loop muted playsinline preload="auto" src="${c.video}"></video>
    <div class="cap">${title}<br>signeur ${esc(c.signeur)} · ${c.images} images (${(c.images / 50).toFixed(2)} s) ·
    mains détectées : droite ${pc(c.droite)}, gauche ${pc(c.gauche)}</div></div>`;
}
const box = document.getElementById("signs");
box.innerHTML = SIGNS.map((s, n) => `<section id="s${n}">
  <h2>${esc(s.etiquette)}</h2>
  <div class="meta">gloses du projet : <b>${esc(s.gloses)}</b> · phrase(s) : ${s.phrases.map(esc).join(" ; ") || "—"}<br>
    Rang du clip copié parmi ${s.classes} signes : <b>${s.rang_copie ?? "—"}</b> ·
    rang d'un autre signeur : <b>${s.rang_autre ?? "pas d'autre signeur dans la galerie"}</b> ·
    ${s.references.length} exemple(s) de référence</div>
  <h3>Clip copié par l'avatar</h3>
  <div class="grid">${clip(s.copie, "copie", `<b>copié</b> (moitié ${esc(s.copie.moitie)})`)}</div>
  <h3>Exemples de référence : ce à quoi l'évaluation compare ${esc(s.etiquette)}</h3>
  <div class="grid">${s.references.map(c => clip(c, "", `<b>${esc(s.etiquette)}</b> · distance ${c.distance}
    · ${c.place}<sup>e</sup> clip le plus proche de la galerie`)).join("") || '<div class="meta">aucun</div>'}</div>
  <h3>Ce que le reconnaisseur trouve plus proche (autres signes)</h3>
  <div class="grid">${s.proches.map((c, k) => clip(c, "", `<b>${k + 1}. ${esc(c.etiquette)}</b>${c.gloses ? ` (${esc(c.gloses)})` : ""}
    · distance ${c.distance}`)).join("")}</div>
  <h3>Autres exemples de ${esc(s.etiquette)} (non utilisés par l'évaluation)</h3>
  <div class="grid">${s.autres.map(c => clip(c, "", `moitié ${esc(c.moitie)}`)).join("") || '<div class="meta">aucun</div>'}</div>
</section>`).join("");
const pick = document.getElementById("pick");
pick.innerHTML = SIGNS.map((s, n) => `<option value="s${n}">${esc(s.etiquette)}</option>`).join("");
pick.onchange = () => document.getElementById(pick.value).scrollIntoView({ behavior: "smooth" });
let rate = 1;
document.getElementById("slow").onclick = e => {
  rate = rate === 1 ? 0.25 : 1;
  document.querySelectorAll("video").forEach(v => v.playbackRate = rate);
  e.target.textContent = rate === 1 ? "Ralenti" : "Vitesse normale";
};
</script></body></html>
"""


if __name__ == "__main__":
    main()
