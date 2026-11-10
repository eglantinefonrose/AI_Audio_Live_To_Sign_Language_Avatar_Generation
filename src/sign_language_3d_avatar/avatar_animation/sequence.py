"""Index du dictionnaire de signes LSFB et assemblage d'une phrase (numpy, hors Blender).

`build_index()` écrit `data/dictionnaire/index.json` : pour chaque glose du projet, ce
que l'avatar doit faire.

| type | quand | rendu |
|---|---|---|
| `signe` | statut `exact` ou `directionnel` dans `lsfb_mapping.csv` | clip nettoyé (forme de base pour un directionnel) |
| `pointage` | `MOI`, `TOI`, `LUI`, `NOUS`, `VOUS`, `EUX` | geste de pointage généré (LSFB-ISOL n'en fournit pas) |
| `absent` | autres statuts, ou glose inconnue | mot affiché en sous-titre, mains au repos |

LSFB-ISOL ne contient pas d'alphabet manuel complet (quelques lettres seulement) :
pas d'épellation dans cette version.

`assemble(gloses)` construit la trajectoire d'une phrase : repos → transition → signe →
transition → … → repos. Les transitions interpolent toutes les articulations (courbe
douce, 12 images = 0,24 s) ; seules les directions comptent pour l'avatar, donc des
signeurs de tailles différentes s'enchaînent sans saut.

    uv run python src/sign_language_3d_avatar/avatar_animation/sequence.py index
    uv run python src/sign_language_3d_avatar/avatar_animation/sequence.py plan "MOI MAISON ALLER"
"""
from __future__ import annotations

import csv
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np

from pose3d import FPS, unstack

HERE = Path(__file__).resolve().parent
GEN = HERE.parent / "dataset_generation"
DICT = HERE / "data" / "dictionnaire"
INDEX = DICT / "index.json"
POINTS = ("MOI", "TOI", "LUI", "NOUS", "VOUS", "EUX")
LICENCE = ("Dérivé de LSFB-ISOL, CC BY-NC-SA 4.0 ; Meurant, Laurence (2015), Corpus LSFB, "
           "LSFB-Lab, FRS-FNRS et Université de Namur ; https://lsfb.info.unamur.be/")
REST_FRAMES, TRANSITION, HOLD_MISSING, POINT_FRAMES = 15, 12, 30, 24


# ------------------------------------------------------------------------- index
def _norm(label: str) -> str:
    s = "".join(c for c in unicodedata.normalize("NFD", label.upper()) if unicodedata.category(c) != "Mn")
    return s.replace("'", "-")


def build_index() -> dict:
    with open(GEN / "lsfb_mapping.csv", newline="", encoding="utf-8") as f:
        mapping = list(csv.DictReader(f))
    with open(GEN / "selection.csv", newline="", encoding="utf-8") as f:
        selection = {r["etiquette_lsfb"]: r for r in csv.DictReader(f)}
    signs = {}
    for path in sorted((DICT / "signes").glob("*.npz")):
        meta = json.loads(str(np.load(path)["meta"]))
        signs[meta["etiquette_lsfb"]] = {
            "fichier": f"signes/{path.name}", "images": meta["images"], "instance": meta["instance"],
            "signeur": meta["signeur"], "retourne": meta["retourne"],
            "score_selection": meta["score_selection"],
            "taux_deux_mains": float(selection.get(meta["etiquette_lsfb"], {}).get("taux_deux_mains") or 0),
        }
    glosses = {}
    for r in mapping:
        g, label, status = r["glose_projet"], r["etiquette_lsfb"], r["statut"]
        if g in POINTS:
            glosses[g] = {"type": "pointage", "cible": g}
        elif status in ("exact", "directionnel") and label in signs:
            glosses[g] = {"type": "signe", "signe": label} | ({"forme": "base"} if status == "directionnel" else {})
        else:
            glosses[g] = {"type": "absent", "statut": status}
    index = {"fps": FPS, "licence": LICENCE, "signes": signs, "gloses": glosses,
             "repos": "mains le long du corps", "transition_images": TRANSITION}
    DICT.mkdir(parents=True, exist_ok=True)
    INDEX.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    return index


def load_index() -> dict:
    return json.loads(INDEX.read_text(encoding="utf-8")) if INDEX.exists() else build_index()


def resolve(gloss: str, index: dict) -> dict:
    """Ce que l'avatar fait pour une glose du projet (ou une étiquette LSFB tapée directement)."""
    g = gloss.strip().upper()
    if g in POINTS:
        return {"type": "pointage", "cible": g}
    if g in index["gloses"]:
        return index["gloses"][g]
    by_norm = {_norm(k): k for k in index["signes"]}
    if _norm(g) in by_norm:
        return {"type": "signe", "signe": by_norm[_norm(g)]}
    return {"type": "absent", "statut": "inconnue"}


def load_sign(label: str, index: dict) -> dict[str, np.ndarray]:
    d = np.load(DICT / index["signes"][label]["fichier"])
    return unstack(d["names"], d["joints"].astype(np.float64))


# ------------------------------------------------------------- mains synthétiques
def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) or 1.0)


def _rot(axis, angle) -> np.ndarray:
    """Matrice de rotation (Rodrigues)."""
    a = _unit(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def synth_hand(side: str, wrist, finger_dir, palm_normal, shape: str) -> np.ndarray:
    """21 points MediaPipe d'une main : `relachee` (doigts légèrement fléchis) ou `index`
    (index tendu, autres doigts repliés : pointage)."""
    y = _unit(finger_dir)
    z = _unit(np.asarray(palm_normal, float) - np.dot(palm_normal, y) * y)    # sortie de la paume
    across = np.cross(y, z) * (1 if side == "R" else -1)                      # vers l'index
    curls = {"relachee": {f: (0.25, 0.30, 0.20) for f in ("index", "middle", "ring", "pinky")},
             "index": {"index": (0.0, 0.0, 0.0), "middle": (1.4, 1.6, 1.0),
                       "ring": (1.4, 1.6, 1.0), "pinky": (1.4, 1.6, 1.0)}}[shape]
    pts = np.zeros((21, 3))
    pts[0] = wrist
    layout = {"index": (5, 0.028, (0.045, 0.027, 0.022)), "middle": (9, 0.009, (0.050, 0.030, 0.024)),
              "ring": (13, -0.010, (0.046, 0.028, 0.022)), "pinky": (17, -0.027, (0.036, 0.021, 0.019))}
    for finger, (k, off, lengths) in layout.items():
        p = np.asarray(wrist, float) + y * 0.085 + across * off
        pts[k], angle = p, 0.0
        for seg, (ln, c) in enumerate(zip(lengths, curls[finger])):
            angle += c                                          # flexion cumulée vers la paume
            p = p + (y * np.cos(angle) - z * np.sin(angle)) * ln
            pts[k + seg + 1] = p
    # Pouce : sur le côté de l'index, rentré si pointage.
    tdir = _unit(y * 0.5 + across * 0.8 - z * (0.3 if shape == "index" else 0.1))
    p = np.asarray(wrist, float) + y * 0.02 + across * 0.02
    pts[1] = p
    for k, ln in zip((2, 3, 4), (0.040, 0.032, 0.025)):
        if shape == "index":
            tdir = _unit(tdir + y * 0.2 - across * 0.6)
        p = p + tdir * ln
        pts[k] = p
    return pts


def torso_of(j: dict[str, np.ndarray], f: int = 0) -> dict[str, np.ndarray]:
    keys = ("hip", "chest", "head", "nose", "shoulder.L", "shoulder.R")
    return {k: j[k][f].copy() for k in keys}


def rest_pose(torso: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Bras le long du corps, mains relâchées paumes vers les cuisses (une image)."""
    j = {k: v.copy() for k, v in torso.items()}
    for s, sx in (("L", 1), ("R", -1)):
        sh = torso[f"shoulder.{s}"]
        j[f"elbow.{s}"] = sh + np.array([0.04 * sx, 0.02, -0.29])
        j[f"wrist.{s}"] = j[f"elbow.{s}"] + np.array([0.0, -0.05, -0.25])
        hand = synth_hand(s, j[f"wrist.{s}"], [0.0, 0.0, -1.0], [-sx, 0.0, 0.0], "relachee")
        for k in range(21):
            j[f"hand.{s}.{k}"] = hand[k]
    return j


def pointing(target: str, torso: dict[str, np.ndarray], rest: dict[str, np.ndarray], n: int = POINT_FRAMES):
    """Pointage de la main droite vers MOI/TOI/LUI (fixe) ou NOUS/VOUS/EUX (balayage)."""
    sh, chest = torso["shoulder.R"], torso["chest"]
    # (position du poignet au début, à la fin), (direction de l'index au début, à la fin)
    plans = {
        "MOI": ((chest + [-0.02, -0.22, -0.06],) * 2, ([0.05, 1.0, -0.15],) * 2),
        "TOI": ((sh + [0.10, -0.32, -0.10],) * 2, ([0.05, -1.0, 0.05],) * 2),
        "LUI": ((sh + [-0.05, -0.30, -0.10],) * 2, ([-0.75, -0.65, 0.0],) * 2),
        "NOUS": ((chest + [0.10, -0.20, -0.06], chest + [-0.12, -0.20, -0.06]), ([0.35, 1.0, -0.1], [-0.35, 1.0, -0.1])),
        "VOUS": ((sh + [0.15, -0.32, -0.10], sh + [-0.08, -0.32, -0.10]), ([0.5, -1.0, 0.0], [-0.5, -1.0, 0.0])),
        "EUX": ((sh + [0.0, -0.32, -0.10], sh + [-0.15, -0.26, -0.10]), ([-0.3, -1.0, 0.0], [-1.0, -0.5, 0.0])),
    }
    (w0, w1), (d0, d1) = plans[target]
    frames = []
    for i in range(n):
        t = 0.5 - 0.5 * np.cos(np.pi * i / max(1, n - 1))           # balayage doux
        w = np.asarray(w0, float) * (1 - t) + np.asarray(w1, float) * t
        d = _unit(np.asarray(d0, float) * (1 - t) + np.asarray(d1, float) * t)
        palm = np.array([0.0, 0.0, -1.0]) if abs(d[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        j = {k: v.copy() for k, v in rest.items()}
        j["wrist.R"] = w
        # Coude sous le milieu épaule-poignet : bras levé vers l'avant, pas pendant.
        j["elbow.R"] = sh + 0.29 * _unit((w - sh) * 0.6 + np.array([-0.04, 0.03, -0.20]))
        hand = synth_hand("R", w, d, palm, "index")
        for k in range(21):
            j[f"hand.R.{k}"] = hand[k]
        frames.append(j)
    return {k: np.stack([f[k] for f in frames]) for k in frames[0]}


# ----------------------------------------------------------------- assemblage
def _repeat(j: dict[str, np.ndarray], n: int) -> dict[str, np.ndarray]:
    return {k: np.repeat(v[None], n, axis=0) for k, v in j.items()}


def _blend(a: dict[str, np.ndarray], b: dict[str, np.ndarray], n: int) -> dict[str, np.ndarray]:
    """n images intermédiaires entre la dernière image de a et la première de b."""
    t = (np.arange(1, n + 1) / (n + 1))
    t = (t * t * (3 - 2 * t))[:, None]                               # courbe douce
    return {k: a[k][-1][None] * (1 - t) + b[k][0][None] * t for k in a}


def assemble(glosses: list[str], index: dict | None = None):
    """(articulations (T, 3) de la phrase, chronologie [(début, fin, glose, type, détail)])."""
    index = index or load_index()
    plan = [(g, resolve(g, index)) for g in glosses]
    first = next((p["signe"] for _, p in plan if p["type"] == "signe"), None)
    ref = load_sign(first, index) if first else load_sign(next(iter(index["signes"])), index)
    torso = torso_of(ref)
    rest_one = rest_pose(torso)
    rest = {k: v[None] for k, v in rest_one.items()}
    parts, timeline, t = [_repeat(rest_one, REST_FRAMES)], [], REST_FRAMES
    for gloss, p in plan:
        if p["type"] == "signe":
            clip = load_sign(p["signe"], index)
            detail = p["signe"] + (" (forme de base)" if p.get("forme") == "base" else "")
        elif p["type"] == "pointage":
            clip = pointing(p["cible"], torso, rest_one)
            detail = "pointage généré"
        else:
            clip = _repeat(rest_one, HOLD_MISSING)
            detail = f"pas de signe ({p.get('statut', '?')}) : mot affiché"
        clip = {k: clip[k] for k in rest}                         # mêmes articulations partout
        trans = _blend(parts[-1], clip, TRANSITION)
        parts += [trans, clip]
        t += TRANSITION
        n = len(next(iter(clip.values())))
        timeline.append((t, t + n, gloss, p["type"], detail))
        t += n
    parts += [_blend(parts[-1], rest, TRANSITION), _repeat(rest_one, REST_FRAMES)]
    joints = {k: np.concatenate([part[k] for part in parts]) for k in rest}
    return joints, timeline


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "index":
        idx = build_index()
        types = {}
        for g in idx["gloses"].values():
            types[g["type"]] = types.get(g["type"], 0) + 1
        print(f"{len(idx['signes'])} signes ; gloses du projet : "
              + ", ".join(f"{k} {v}" for k, v in sorted(types.items())) + f" → {INDEX.relative_to(HERE.parent.parent.parent)}")
    elif len(sys.argv) >= 3 and sys.argv[1] == "plan":
        joints, timeline = assemble(sys.argv[2].split())
        total = len(joints["hip"])
        for a, b, g, kind, detail in timeline:
            print(f"  {a / FPS:5.2f}–{b / FPS:5.2f} s  {g:18s} {kind:9s} {detail}")
        print(f"  durée totale : {total / FPS:.2f} s")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
