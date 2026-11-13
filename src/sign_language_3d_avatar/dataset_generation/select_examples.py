"""Choisit automatiquement le meilleur exemple LSFB de chaque signe (sans relecture).

**Critère principal : le reconnaisseur.** Chaque exemple téléchargé est nettoyé comme le
dictionnaire (`clean_signs.clean`), puis classé par le reconnaisseur de l'évaluation
(`evaluation/features.py`) contre les exemples des **autres** signeurs de la moitié
« choix » (`features.signer_half`), chaque signe étant représenté par son exemple le plus
proche (`K_VOISINS` = 1). Défaut connu : deux clips atypiques qui se ressemblent (OU de
S032 et S044) se classent 1ers l'un grâce à l'autre. La moyenne des 3 plus proches a été
essayée (12/11/2026) : OU et QUAND.8 corrigés, mais 161 clips changés et phrases reconnues
au seuil 90 % de 48 % à 39 % ; à régler par validation croisée dans la moitié « choix ». On garde l'exemple le mieux reconnu (rang centile du
bon signe, arrondi au point), c'est-à-dire le plus typique du signe ; à égalité, ou si la
moitié « choix » n'a pas d'autre exemple du signe, le score technique ci-dessous départage.
L'évaluation de l'avatar n'utilise que l'autre moitié des signeurs : elle ne voit jamais
les exemples qui ont servi au choix.

Un clip choisi à l'œil dans `choix_clips.csv` (page `revue_clips.py`) passe avant tout
critère automatique (critère « relecture »).

Score technique (`--sans-reconnaisseur` : seul critère, comme avant) :

Pour chaque étiquette LSFB utilisée par le projet (statuts `exact`, `directionnel`, `validé`,
et chaque signe ou lettre des statuts `décomposé` et `épeler` de `lsfb_mapping.csv`), tous les exemples téléchargés sont notés sur trois critères :

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

import argparse
import csv
import hashlib
import math
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

GEN = Path(__file__).resolve().parent
sys.path[:0] = [str(GEN.parent / "avatar_animation"), str(GEN.parent / "evaluation")]
from clean_signs import clean, is_letter  # noqa: E402
from features import Gallery, percentile, rank_of, resample, signer_half, vector  # noqa: E402

DATA = GEN / "data" / "lsfb"
MAPPING = GEN / "lsfb_mapping.csv"
OUT = GEN / "selection.csv"
MANUAL = GEN / "choix_clips.csv"          # clips choisis à l'œil (revue_clips.py)
STATUSES = {"exact", "directionnel", "validé", "décomposé", "épeler"}
SHOULDER_L, SHOULDER_R = 11, 12
HAND_POINTS = [0, 4, 8, 12, 16, 20]          # poignet + bouts des doigts
WEIGHTS = {"mains": 0.5, "tremblement": 0.3, "duree": 0.2}
FIELDS = ["etiquette_lsfb", "gloses_projet", "instance", "signeur", "candidats", "images",
          "taux_deux_mains", "tremblement", "ecart_duree", "score", "brut", "centile_reconnu", "critere"]
CACHE = DATA / "cache_selection"
K_VOISINS = 1          # distance au plus proche exemple de chaque signe (voir l'en-tête)


def describe_all(items: list[tuple[str, str]]) -> dict[str, np.ndarray]:
    """Descripteur rééchantillonné de chaque (instance, étiquette), nettoyé comme le
    dictionnaire ; mis en cache (dérivé de LSFB, ignoré par git)."""
    code = b"".join((GEN.parent / p).read_bytes() for p in ("avatar_animation/clean_signs.py", "evaluation/features.py"))
    key = hashlib.sha256("|".join(sorted(f"{i}:{l}" for i, l in items)).encode() + code).hexdigest()[:12]
    path = CACHE / f"descripteurs_{key}.npz"
    if path.exists():
        d = np.load(path)
        return dict(zip(d["ids"], d["seqs"]))
    print(f"Reconnaisseur : {len(items)} exemples à décrire (une fois, mis en cache)…", flush=True)
    out = {}
    for k, (inst, label) in enumerate(items):
        try:
            out[inst] = resample(vector(clean(inst, is_letter(label))[0])).astype(np.float32)
        except Exception:  # noqa: BLE001 — exemple vide ou défectueux : non classé
            pass
        if k % 1000 == 0:
            print(f"  {k}/{len(items)}", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ids=np.array(list(out)), seqs=np.stack(list(out.values())))
    return out


_GALLERY: Gallery | None = None


def _init(labels, signers, ids, seqs) -> None:
    global _GALLERY
    _GALLERY = Gallery(list(labels), list(signers), list(ids), seqs)


def _centile(job: tuple[str, str, str, np.ndarray]) -> tuple[str, float | None]:
    inst, label, signer, seq = job
    return inst, percentile(rank_of(_GALLERY.rank(seq, exclude_signer=signer, k=K_VOISINS), label))


def recognizer_centiles(instances: dict[str, list[dict]]) -> dict[str, float | None]:
    """Rang centile du bon signe pour chaque exemple, contre la moitié « choix » des
    signeurs, sans le signeur de l'exemple (None : pas d'autre exemple du signe)."""
    items = [(r["id"], label) for label, rows in instances.items() for r in rows]
    seqs = describe_all(items)
    pool = [(r["id"], label, r["signer"]) for label, rows in instances.items() for r in rows
            if r["id"] in seqs and signer_half(r["signer"]) == "choix"]
    init = ([l for _, l, _ in pool], [s for _, _, s in pool], [i for i, _, _ in pool],
            np.stack([seqs[i] for i, _, _ in pool]))
    jobs = [(r["id"], label, r["signer"], seqs[r["id"]]) for label, rows in instances.items() for r in rows
            if r["id"] in seqs]
    print(f"Reconnaisseur : {len(jobs)} exemples classés contre {len(pool)} exemples de la moitié « choix »…",
          flush=True)
    with ProcessPoolExecutor(initializer=_init, initargs=init) as ex:
        return dict(ex.map(_centile, jobs, chunksize=32))


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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sans-reconnaisseur", action="store_true", help="score technique seul (ancien choix)")
    args = ap.parse_args()
    with open(MAPPING, newline="", encoding="utf-8") as f:
        mapping = [r for r in csv.DictReader(f) if r["statut"] in STATUSES]
    glosses = defaultdict(list)
    for r in mapping:
        for label in filter(None, r["etiquette_lsfb"].split("+")):     # suites : LUNDI+DERNIER, FS:G+FS:P
            glosses[label].append(r["glose_projet"])
    with open(DATA / "instances.csv", newline="", encoding="utf-8") as f:
        instances = defaultdict(list)
        for r in csv.DictReader(f):
            if r["sign"] in glosses and (DATA / "poses" / "pose" / f"{r['id']}.npy").exists():
                instances[r["sign"]].append(r)
    centiles = {} if args.sans_reconnaisseur else recognizer_centiles(instances)
    manual = {}
    if MANUAL.exists():
        with open(MANUAL, newline="", encoding="utf-8-sig") as f:
            manual = {r["etiquette_lsfb"]: r["instance"] for r in csv.DictReader(f)
                      if r["verdict"] == "choisi" and r["instance"]}

    rows, missing_raw = [], 0
    for label in sorted(glosses):
        cands = []
        for inst in instances[label]:
            m = measure(inst["id"])
            if m:
                cands.append(inst | m)
        chosen = next((c for c in cands if c["id"] == manual.get(label)), None)
        # On ne compare que des mesures comparables : brutes si au moins un exemple les a.
        if any(c["brut"] for c in cands):
            cands = [c for c in cands if c["brut"]]
        if chosen and chosen not in cands:        # choix humain gardé même sans poses brutes
            cands.append(chosen)
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
        for c in cands:
            c["centile"] = centiles.get(c["id"])
        ranked = any(c["centile"] is not None for c in cands)
        best = chosen or max(cands, key=lambda c: (round(c["centile"], 2) if c["centile"] is not None else -1.0,
                                                   c["score"], [-ord(ch) for ch in c["id"]]))
        rows.append({"etiquette_lsfb": label, "gloses_projet": " ".join(sorted(glosses[label])),
                     "instance": best["id"], "signeur": best["signer"], "candidats": len(cands),
                     "images": best["images"], "taux_deux_mains": f"{best['taux_deux_mains']:.3f}",
                     "tremblement": f"{best['tremblement']:.5f}", "ecart_duree": f"{best['ecart_duree']:.3f}",
                     "score": f"{best['score']:.3f}", "brut": "oui" if best["brut"] else "non",
                     "centile_reconnu": "" if best["centile"] is None else f"{best['centile']:.3f}",
                     "critere": "relecture" if chosen else "reconnaisseur" if ranked else "technique"})

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    rates = [float(r["taux_deux_mains"]) for r in rows]
    if centiles:
        by_rec = [float(r["centile_reconnu"]) for r in rows if r["critere"] == "reconnaisseur"]
        by_eye = sum(r["critere"] == "relecture" for r in rows)
        print(f"Reconnaisseur : {len(by_rec)} signes choisis par le reconnaisseur (centile médian du choix "
              f"{np.median(by_rec):.0%}), {by_eye} à l'œil ({MANUAL.name}), {len(rows) - len(by_rec) - by_eye} "
              "au score technique (pas d'autre exemple dans la moitié « choix »)")
    print(f"{len(rows)} signes choisis sur {len(glosses)} ; mains détectées (médiane) "
          f"{np.median(rates):.0%} ; candidats par signe (médiane) "
          f"{int(np.median([r['candidats'] for r in rows]))}")
    if missing_raw:
        print(f"⚠️  {missing_raw} signe(s) notés sans poses brutes (détection sous-estimée) : "
              "relancer lsfb_subset.py download --raw puis ce script.")
    print(f"→ {OUT.relative_to(GEN.parent.parent.parent)}")


if __name__ == "__main__":
    main()
