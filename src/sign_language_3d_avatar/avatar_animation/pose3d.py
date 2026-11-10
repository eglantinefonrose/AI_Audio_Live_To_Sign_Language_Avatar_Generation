"""Points MediaPipe (LSFB) -> articulations 3D en mètres, repère Blender (numpy seul).

Partagé par le retargeting Blender (`retarget_blender.py`, `avatar_rig.py`) et par les
étapes hors Blender (nettoyage, enchaînement de phrases). Repère : Z en haut, le signeur
regarde vers −Y (caméra en −Y), sa gauche est en +X ; bassin à l'origine.
"""
from __future__ import annotations

import numpy as np

FPS = 50                      # LSFB : 50 images/s
SHOULDER_WIDTH = 0.38         # m : échelle commune de tous les signes

# MediaPipe Pose : indices utiles. « L »/« R » = gauche/droite du signeur.
NOSE, EAR_L, EAR_R = 0, 7, 8
SHOULDER = {"L": 11, "R": 12}
ELBOW = {"L": 13, "R": 14}
WRIST = {"L": 15, "R": 16}
HIP_L, HIP_R = 23, 24
# Proportions anthropométriques (÷ largeur d'épaules) pour retrouver la profondeur.
UPPER_ARM, FOREARM = 0.80, 0.68
# MediaPipe Hands : chaînes de doigts (base -> pointe).
FINGERS = {"thumb": (1, 2, 3, 4), "index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
           "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}


def to_blender(xyz: np.ndarray, aspect: float) -> np.ndarray:
    """Image MediaPipe (x à droite, y en bas, z vers la caméra < 0) -> Blender (Z en haut, caméra en -Y)."""
    out = np.empty_like(xyz)
    out[..., 0] = xyz[..., 0] * aspect
    out[..., 1] = xyz[..., 2] * aspect
    out[..., 2] = -xyz[..., 1]
    return out


def lift(parent: np.ndarray, child: np.ndarray, length: float) -> np.ndarray:
    """Recalcule la profondeur de `child` pour que |child - parent| = length (signe pris de MediaPipe)."""
    d = child - parent
    planar = d[:, 0] ** 2 + d[:, 2] ** 2
    depth = np.sqrt(np.clip(length ** 2 - planar, 0.0, None))
    sign = np.where(d[:, 1] < 0, -1.0, 1.0)
    out = child.copy()
    out[:, 1] = parent[:, 1] + sign * depth
    return out


def joints(pose: np.ndarray, hands: dict[str, np.ndarray], aspect: float) -> dict[str, np.ndarray]:
    """Positions 3D (T, 3) de chaque articulation, en mètres, bassin à l'origine."""
    p = to_blender(pose.astype(np.float64), aspect)
    hip = (p[:, HIP_L] + p[:, HIP_R]) / 2
    sw = np.median(np.linalg.norm(p[:, SHOULDER["L"]] - p[:, SHOULDER["R"]], axis=1))
    scale = SHOULDER_WIDTH / sw
    p = (p - hip[:, None]) * scale
    # Profondeur MediaPipe du torse et du visage peu fiable (la tête part en avant) :
    # torse droit (on garde seulement sa rotation), tête au-dessus du torse.
    sh_depth = (p[:, SHOULDER["L"], 1] + p[:, SHOULDER["R"], 1]) / 2
    p[:, :, 1] -= sh_depth[:, None]
    j = {"hip": np.zeros((len(p), 3)), "head": (p[:, EAR_L] + p[:, EAR_R]) / 2}
    j["head"][:, 1] = 0.0
    j["nose"] = lift(j["head"], p[:, NOSE], 0.10)
    for s in "LR":
        j[f"shoulder.{s}"] = p[:, SHOULDER[s]]
        # La profondeur MediaPipe des bras est très bruitée : on la reconstruit
        # à partir de longueurs d'os plausibles (au moins la longueur 2D vue).
        for name, a, b, ratio in (("elbow", SHOULDER[s], ELBOW[s], UPPER_ARM),
                                  ("wrist", ELBOW[s], WRIST[s], FOREARM)):
            parent = j[f"shoulder.{s}"] if name == "elbow" else j[f"elbow.{s}"]
            child = p[:, b] - p[:, a] + parent
            planar = np.linalg.norm((child - parent)[:, [0, 2]], axis=1)
            length = max(ratio * SHOULDER_WIDTH, float(np.percentile(planar, 98)))
            j[f"{name}.{s}"] = lift(parent, child, length)
        # Main : forme 3D de MediaPipe Hands, collée au poignet du bras.
        h = to_blender(hands[s].astype(np.float64), aspect) * scale
        h = h - h[:, :1] + j[f"wrist.{s}"][:, None]
        for k in range(21):
            j[f"hand.{s}.{k}"] = h[:, k]
    j["chest"] = (j["shoulder.L"] + j["shoulder.R"]) / 2
    return j


def stack(j: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    """Dictionnaire d'articulations -> (noms, tableau (T, N, 3)), pour sauvegarder en .npz."""
    names = sorted(j)
    return names, np.stack([j[n] for n in names], axis=1)


def unstack(names, arr: np.ndarray) -> dict[str, np.ndarray]:
    return {str(n): arr[:, i] for i, n in enumerate(names)}
