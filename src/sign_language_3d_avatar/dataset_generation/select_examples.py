"""Choisit automatiquement le meilleur exemple LSFB de chaque signe (sans relecture).

Pour chaque étiquette LSFB utilisée par le projet (statuts `exact` et `directionnel` de
`lsfb_mapping.csv`), tous les exemples téléchargés sont notés sur trois critères :

- **mains détectées** : part des images où MediaPipe a trouvé les deux mains (poses
  brutes, avant l'interpolation de LSFB) ;
- **tremblement** : accélération moyenne des points de la main (poignet + bouts des
  doigts) entre images consécutives détectées, en largeurs d'épaules par image² ;
- **durée** : écart à la durée médiane des exemples du même signe (en log).

Score = 0,5 × mains + 0,3 × (1 − tremblement relatif) + 0,2 × (1 − écart de durée), où
le tremblement est rapporté au pire exemple du même signe et l'écart de durée plafonné
à un facteur 2. Le meilleur score l'emporte (à égalité : identifiant le plus petit).

Sortie : `selection.csv` (versionné : identifiants et scores seulement, pas de données).

Prérequis : poses nettoyées et brutes (mains + corps) de plusieurs exemples par signe :
    uv run python src/sign_language_3d_avatar/dataset_generation/lsfb_subset.py download \\
        --words-file src/sign_language_3d_avatar/dataset_generation/lsfb_words.txt \\
        --max-par-glose 10 --landmarks pose,left_hand,right_hand --raw

Usage :
    uv run python src/sign_language_3d_avatar/dataset_generation/select_examples.py
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

GEN = Path(__file__).resolve().parent
DATA = GEN / "data" / "lsfb"
MAPPING = GEN / "lsfb_mapping.csv"
OUT = GEN / "selection.csv"
STATUSES = {"exact", "directionnel"}
SHOULDER_L, SHOULDER_R = 11, 12
HAND_POINTS = [0, 4, 8, 12, 16, 20]          # poignet + bouts des doigts
WEIGHTS = {"mains": 0.5, "tremblement": 0.3, "duree": 0.2}
FIELDS = ["etiquette_lsfb", "gloses_projet", "instance", "signeur", "candidats", "images",
          "taux_deux_mains", "tremblement", "ecart_duree", "score", "brut"]


def load(kind: str, part: str, instance: str) -> np.ndarray | None:
    path = DATA / kind / part / f"{instance}.npy"
    return np.load(path).astype(np.float32) if path.exists() else None


def measure(instance: str) -> dict | None:
    """Mesures brutes d'un exemple ; None si les poses nettoyées manquent."""
    pose = load("poses", "pose", instance)
    if pose is None or len(pose) < 3:     # quelques exemples LSFB sont vides
        return None
    raw = {s: load("poses_raw", f"{s}_hand", instance) for s in ("left", "right")}
    has_raw = all(h is not None for h in raw.values())
    hands = raw if has_raw else {s: load("poses", f"{s}_hand", instance) for s in ("left", "right")}
    detected = np.ones(len(pose), bool)
    for h in hands.values():
        detected &= ~np.isnan(h).any(axis=(1, 2))
    shoulders = np.linalg.norm(pose[:, SHOULDER_L, :2] - pose[:, SHOULDER_R, :2], axis=1)
    width = float(np.nanmedian(shoulders)) or 1.0
    jitter = []
    for h in hands.values():
        pts = h[:, HAND_POINTS, :2] / width
        acc = pts[2:] - 2 * pts[1:-1] + pts[:-2]          # NaN si une des 3 images manque
        mag = np.linalg.norm(acc, axis=2).mean(axis=1)
        mag = mag[~np.isnan(mag)]
        if len(mag):
            jitter.append(float(mag.mean()))
    return {"images": len(pose), "taux_deux_mains": float(detected.mean()),
            "tremblement": float(np.mean(jitter)) if jitter else math.inf, "brut": has_raw}


def main() -> None:
    with open(MAPPING, newline="", encoding="utf-8") as f:
        mapping = [r for r in csv.DictReader(f) if r["statut"] in STATUSES]
    glosses = defaultdict(list)
    for r in mapping:
        glosses[r["etiquette_lsfb"]].append(r["glose_projet"])
    with open(DATA / "instances.csv", newline="", encoding="utf-8") as f:
        instances = defaultdict(list)
        for r in csv.DictReader(f):
            if r["sign"] in glosses:
                instances[r["sign"]].append(r)

    rows, missing_raw = [], 0
    for label in sorted(glosses):
        cands = []
        for inst in instances[label]:
            m = measure(inst["id"])
            if m:
                cands.append(inst | m)
        # On ne compare que des mesures comparables : brutes si au moins un exemple les a.
        if any(c["brut"] for c in cands):
            cands = [c for c in cands if c["brut"]]
        if not cands:
            continue
        missing_raw += not cands[0]["brut"]
        median = float(np.median([c["images"] for c in cands]))
        worst = max((c["tremblement"] for c in cands if math.isfinite(c["tremblement"])), default=1.0) or 1.0
        for c in cands:
            c["ecart_duree"] = min(1.0, abs(math.log(c["images"] / median)) / math.log(2))
            jit = min(1.0, c["tremblement"] / worst) if math.isfinite(c["tremblement"]) else 1.0
            c["score"] = (WEIGHTS["mains"] * c["taux_deux_mains"] + WEIGHTS["tremblement"] * (1 - jit)
                          + WEIGHTS["duree"] * (1 - c["ecart_duree"]))
        best = max(cands, key=lambda c: (c["score"], [-ord(ch) for ch in c["id"]]))
        rows.append({"etiquette_lsfb": label, "gloses_projet": " ".join(sorted(glosses[label])),
                     "instance": best["id"], "signeur": best["signer"], "candidats": len(cands),
                     "images": best["images"], "taux_deux_mains": f"{best['taux_deux_mains']:.3f}",
                     "tremblement": f"{best['tremblement']:.5f}", "ecart_duree": f"{best['ecart_duree']:.3f}",
                     "score": f"{best['score']:.3f}", "brut": "oui" if best["brut"] else "non"})

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    rates = [float(r["taux_deux_mains"]) for r in rows]
    print(f"{len(rows)} signes choisis sur {len(glosses)} ; mains détectées (médiane) "
          f"{np.median(rates):.0%} ; candidats par signe (médiane) "
          f"{int(np.median([r['candidats'] for r in rows]))}")
    if missing_raw:
        print(f"⚠️  {missing_raw} signe(s) notés sans poses brutes (détection sous-estimée) : "
              "relancer lsfb_subset.py download --raw puis ce script.")
    print(f"→ {OUT.relative_to(GEN.parent.parent.parent)}")


if __name__ == "__main__":
    main()
