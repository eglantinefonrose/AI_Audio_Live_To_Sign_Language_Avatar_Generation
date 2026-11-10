"""Nettoie automatiquement l'exemple choisi de chaque signe et le convertit en trajectoire 3D.

Entrée : `dataset_generation/selection.csv` (meilleur exemple par signe, cf.
`select_examples.py`) et les poses LSFB nettoyées (MediaPipe, déjà interpolées et
lissées par LSFB). Pour chaque signe, sans intervention humaine :

1. **droitier** : si la main gauche du signeur bouge nettement plus que la droite, le
   clip est retourné (gauche ↔ droite) pour que la main dominante de l'avatar soit
   toujours la droite ;
2. **sauts** : les images où une main saute loin de ses voisines (> 0,25 largeur
   d'épaules par rapport à la médiane locale) sont remplacées par interpolation ;
3. **lissage léger** : Savitzky-Golay (fenêtre 5, ordre 2) ;
4. **découpe** : les images immobiles au début et à la fin (vitesse des mains < 15 %
   du maximum) sont retirées, avec 2 images de marge ;
5. **3D** : articulations en mètres dans le repère commun (`pose3d.joints`).

Sortie (ignorée par git, dérivée de LSFB CC BY-NC-SA 4.0) :
`data/dictionnaire/signes/<ÉTIQUETTE>.npz` (noms, articulations (T, N, 3), métadonnées).

Usage :
    uv run python src/sign_language_3d_avatar/avatar_animation/clean_signs.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from pose3d import FPS, SHOULDER, WRIST, joints, stack

HERE = Path(__file__).resolve().parent
LSFB = HERE.parent / "dataset_generation" / "data" / "lsfb"
SELECTION = HERE.parent / "dataset_generation" / "selection.csv"
OUT = HERE / "data" / "dictionnaire" / "signes"
ASPECT = 4 / 3                # affichage 4:3 des vidéos LSFB (720×576, pixels non carrés)
# Paires gauche/droite de MediaPipe Pose (pour retourner un clip).
POSE_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18),
              (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]
SG5 = np.array([-3, 12, 17, 12, -3]) / 35          # Savitzky-Golay fenêtre 5, ordre 2
MIN_FRAMES = 8


def safe_name(label: str) -> str:
    return label.replace("/", "_").replace(":", "_")


def interpolate_nan(a: np.ndarray) -> np.ndarray:
    """Comble les NaN le long du temps (linéaire, valeurs extrêmes prolongées)."""
    a = a.copy()
    t = np.arange(len(a))
    flat = a.reshape(len(a), -1)
    for k in range(flat.shape[1]):
        bad = np.isnan(flat[:, k])
        if bad.all():
            flat[:, k] = 0.0
        elif bad.any():
            flat[bad, k] = np.interp(t[bad], t[~bad], flat[~bad, k])
    return flat.reshape(a.shape)


def mirror(pose: np.ndarray, hands: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Retourne le clip : gauche ↔ droite (x -> 1 - x, points pairs échangés)."""
    p = pose.copy()
    for a, b in POSE_PAIRS:
        p[:, [a, b]] = p[:, [b, a]]
    p[..., 0] = 1.0 - p[..., 0]
    h = {"L": hands["R"].copy(), "R": hands["L"].copy()}
    for s in h:
        h[s][..., 0] = 1.0 - h[s][..., 0]
    return p, h


def despike(series: np.ndarray, width: float) -> tuple[np.ndarray, int]:
    """Remplace les images trop loin de la médiane glissante (5 images) par interpolation."""
    T = len(series)
    if T < 5:
        return series, 0
    pad = np.concatenate([series[:1].repeat(2, 0), series, series[-1:].repeat(2, 0)])
    windows = np.stack([pad[i:i + T] for i in range(5)])
    med = np.median(windows, axis=0)
    dist = np.linalg.norm((series - med).reshape(T, -1, series.shape[-1]), axis=-1).max(axis=1)
    bad = dist > 0.25 * width
    if not bad.any() or bad.all():
        return series, 0
    out = series.copy()
    out[bad] = np.nan
    return interpolate_nan(out), int(bad.sum())


def smooth(series: np.ndarray) -> np.ndarray:
    if len(series) < 5:
        return series
    pad = np.concatenate([series[:1].repeat(2, 0), series, series[-1:].repeat(2, 0)])
    return sum(c * pad[i:i + len(series)] for i, c in enumerate(SG5))


def active_range(pose: np.ndarray, hands: dict[str, np.ndarray], width: float) -> tuple[int, int]:
    """Bornes [début, fin) sans les images immobiles au début et à la fin."""
    T = len(pose)
    pts = np.concatenate([pose[:, [WRIST["L"], WRIST["R"]], :2], hands["L"][:, [0, 8], :2],
                          hands["R"][:, [0, 8], :2]], axis=1)
    speed = np.zeros(T)
    speed[1:] = np.linalg.norm(np.diff(pts, axis=0), axis=2).max(axis=1) / width
    if speed.max() <= 0:
        return 0, T
    moving = np.flatnonzero(speed >= 0.15 * speed.max())
    start, end = max(0, moving[0] - 3), min(T, moving[-1] + 3)   # 2 images de marge (+1 : vitesse décalée)
    if end - start < MIN_FRAMES:
        mid = (start + end) // 2
        start, end = max(0, mid - MIN_FRAMES // 2), min(T, mid + MIN_FRAMES // 2)
    return start, end


def clean(instance: str) -> tuple[dict[str, np.ndarray], dict]:
    load = lambda part: np.load(LSFB / "poses" / part / f"{instance}.npy").astype(np.float64)
    pose = interpolate_nan(load("pose"))
    hands = {"L": interpolate_nan(load("left_hand")), "R": interpolate_nan(load("right_hand"))}
    width = float(np.median(np.linalg.norm(pose[:, SHOULDER["L"], :2] - pose[:, SHOULDER["R"], :2], axis=1)))

    # 1. Main dominante : longueur du trajet de chaque poignet.
    path = {s: float(np.linalg.norm(np.diff(hands[s][:, 0, :2], axis=0), axis=1).sum()) for s in "LR"}
    mirrored = path["L"] > 1.3 * path["R"]
    if mirrored:
        pose, hands = mirror(pose, hands)

    # 2-3. Sauts puis lissage.
    spikes = 0
    for s in "LR":
        hands[s], n = despike(hands[s], width)
        spikes += n
        hands[s] = smooth(hands[s])
    pose = smooth(pose)

    # 4. Découpe des immobilités.
    start, end = active_range(pose, hands, width)
    pose, hands = pose[start:end], {s: h[start:end] for s, h in hands.items()}

    # 5. Articulations 3D.
    j = joints(pose, hands, ASPECT)
    meta = {"instance": instance, "retourne": bool(mirrored), "sauts_corriges": spikes,
            "images_source": int(len(load("pose"))), "debut": int(start), "fin": int(end),
            "images": int(end - start), "fps": FPS}
    return j, meta


def main() -> None:
    with open(SELECTION, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    OUT.mkdir(parents=True, exist_ok=True)
    stats = {"retourne": 0, "sauts": 0, "images": [], "erreurs": []}
    for r in rows:
        try:
            j, meta = clean(r["instance"])
        except Exception as e:  # un exemple défectueux ne bloque pas les autres
            stats["erreurs"].append(f"{r['etiquette_lsfb']} : {e}")
            continue
        meta |= {"etiquette_lsfb": r["etiquette_lsfb"], "signeur": r["signeur"],
                 "gloses_projet": r["gloses_projet"].split(), "score_selection": float(r["score"])}
        names, arr = stack(j)
        np.savez_compressed(OUT / f"{safe_name(r['etiquette_lsfb'])}.npz", names=np.array(names),
                            joints=arr.astype(np.float32), meta=json.dumps(meta, ensure_ascii=False))
        stats["retourne"] += meta["retourne"]
        stats["sauts"] += meta["sauts_corriges"]
        stats["images"].append(meta["images"])
    n = len(stats["images"])
    print(f"{n} signes nettoyés → {OUT.relative_to(HERE.parent.parent.parent)}")
    if n:
        print(f"  retournés (gauchers) : {stats['retourne']} ; images corrigées (sauts) : {stats['sauts']} ; "
              f"durée médiane : {np.median(stats['images']) / FPS:.2f} s")
    for e in stats["erreurs"]:
        print(f"  ⚠️  {e}")


if __name__ == "__main__":
    main()
