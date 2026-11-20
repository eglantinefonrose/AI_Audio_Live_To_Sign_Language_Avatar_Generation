"""Nettoie automatiquement l'exemple choisi de chaque signe et le convertit en trajectoire 3D.

Entrée : `dataset_generation/selection.csv` (meilleur exemple par signe, cf.
`select_examples.py`) et les poses LSFB nettoyées (MediaPipe, déjà interpolées et
lissées par LSFB). Pour chaque signe, sans intervention humaine :

1. **droitier** : si la main gauche du signeur bouge nettement plus que la droite, le
   clip est retourné (gauche ↔ droite) pour que la main dominante de l'avatar soit
   toujours la droite. Pour les lettres de l'alphabet manuel (`FS:A`…), presque immobiles,
   c'est la main **levée** qui épelle : l'autre bouge parfois plus en redescendant ;
2. **sauts** : les images où une main saute loin de ses voisines (> 0,25 largeur
   d'épaules par rapport à la médiane locale) sont remplacées par interpolation ;
3. **lissage léger** : Savitzky-Golay (fenêtre 5, ordre 2) ;
4. **découpe** : les images immobiles au début et à la fin (vitesse des mains < 15 %
   du maximum) sont retirées, avec 2 images de marge ;
5. **3D** : articulations en mètres dans le repère commun (`pose3d.joints`) ;
6. **poitrine** : la profondeur reconstruite place parfois la main dans le plan des
   épaules, donc dans le corps. Une main devant le torse est avancée en bloc, poignet
   compris, jusqu'à la surface de la poitrine : une demi-ellipse sous les épaules,
   0,25 largeur d'épaules devant elles au milieu, 0,45 de demi-largeur, arrondie vers
   les clavicules (aucune correction au-dessus des épaules).

Lettre avec une image de référence (`dataset_generation/data/alphabet/<LETTRE>.json`,
extrait par `dataset_generation/lettres_images.py`) : la main droite du clip est remplacée,
image par image, par la forme de main de l'image, recalée sur la paume du clip (poignet,
base du pouce et des doigts) ; le bras, la position et l'orientation restent ceux du clip.
Lettre avec un `mouvement` (vidéo entière, lettre qui bouge : rotation du poignet de K…) :
le clip est rééchantillonné à la durée de la vidéo et chaque image prend la main de l'image
correspondante de la vidéo, orientation comprise : la lettre tenue est posée sur
l'orientation médiane du clip, et la vidéo tourne la main à partir de là.
Lettre avec `orientation: video` (vidéo de face) : l'orientation de la main est celle de la
vidéo elle-même, image par image (même repère caméra que LSFB) ; seuls la taille, la position
et le bras viennent du clip.
Lettre avec une `paume` imposée (bas, haut, avant, soi ; page de vérification) : la main
tourne autour du poignet pour que la paume médiane regarde dans cette direction.

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
ALPHABET = HERE.parent / "dataset_generation" / "data" / "alphabet"
PALM = [0, 1, 2, 5, 9, 13, 17]  # poignet, base du pouce, bases des doigts (points MediaPipe)
SELECTION = HERE.parent / "dataset_generation" / "selection.csv"
OUT = HERE / "data" / "dictionnaire" / "signes"
ASPECT = 4 / 3                # affichage 4:3 des vidéos LSFB (720×576, pixels non carrés)
# Paires gauche/droite de MediaPipe Pose (pour retourner un clip).
POSE_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18),
              (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]
# Direction de la paume (coordonnées image : y vers le bas, z en s'éloignant de la caméra).
FACING = {"bas": (0, 1, 0), "haut": (0, -1, 0), "avant": (0, 0, -1), "soi": (0, 0, 1)}
SG5 = np.array([-3, 12, 17, 12, -3]) / 35          # Savitzky-Golay fenêtre 5, ordre 2
MIN_FRAMES = 8
CHEST_FRONT, CHEST_HALF_WIDTH = 0.25, 0.45         # ÷ largeur d'épaules (≈ 9,5 et 17 cm)


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


def is_letter(label: str) -> bool:
    return label.startswith("FS:")


def letter_shape(label: str) -> tuple[np.ndarray, str, dict | None, str | None, bool] | None:
    """(forme de main, image de référence, mouvement ou None, paume imposée ou None,
    orientation prise dans la vidéo) pour la lettre `FS:X`, s'il y en a une."""
    if not is_letter(label):
        return None
    path = ALPHABET / f"{label.split(':', 1)[1]}.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    return np.array(d["points"], dtype=np.float64), d["image"], d.get("mouvement"), d.get("paume"), d.get("orientation") == "video"


def resample(a: np.ndarray, n: int) -> np.ndarray:
    """`a` (T, …) ramené à n images par interpolation linéaire dans le temps."""
    t = np.linspace(0, len(a) - 1, n)
    i = np.minimum(t.astype(int), len(a) - 2) if len(a) > 1 else np.zeros(n, int)
    w = (t - i).reshape(-1, *([1] * (a.ndim - 1)))
    return a[i] * (1 - w) + a[np.minimum(i + 1, len(a) - 1)] * w


def _similarity(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, float]:
    """Similitude (rotation propre, échelle, translation) qui envoie `src` sur `dst` (Umeyama) ;
    renvoie (rotation, échelle, translation)."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    u, sv, vt = np.linalg.svd(b.T @ a)
    d = np.diag([1, 1, np.sign(np.linalg.det(u @ vt))])
    r = u @ d @ vt
    scale = float((sv * np.diag(d)).sum() / max((a ** 2).sum(), 1e-12))
    return r, scale, md - scale * r @ ms


def face_palm(hand: np.ndarray, facing: str) -> np.ndarray:
    """La main (T, 21, 3) d'une main droite, tournée autour du poignet de chaque image par la
    plus petite rotation qui amène la normale de la paume médiane sur `FACING[facing]`."""
    p = np.median(hand, axis=0)
    n = np.cross(p[5] - p[0], p[17] - p[0])        # main droite : côté paume
    n /= max(float(np.linalg.norm(n)), 1e-9)
    t = np.array(FACING[facing], dtype=np.float64)
    axis, cos = np.cross(n, t), float(n @ t)
    if np.linalg.norm(axis) < 1e-9:                # déjà dans la direction, ou à l'opposé
        if cos > 0:
            return hand
        axis = np.cross(n, [1.0, 0, 0]) if abs(n[0]) < 0.9 else np.cross(n, [0, 1.0, 0])
    k = axis / np.linalg.norm(axis)
    angle = np.arctan2(float(np.linalg.norm(np.cross(n, t))), cos)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    r = np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * kx @ kx
    wrist = hand[:, :1]
    return (hand - wrist) @ r.T + wrist


def fit_hand(clip: np.ndarray, shape: np.ndarray, motion: np.ndarray | None = None,
             palm: str | None = None, absolute: bool = False) -> np.ndarray:
    """La forme `shape` (21, 3), recalée sur la paume de chaque image du clip (T, 21, 3).
    Coordonnées LSFB normalisées : x, z en largeur d'image, y en hauteur (vidéo 4:3). La
    chiralité (main droite ou gauche sur l'image) est choisie une fois, sur la paume médiane.
    `motion` (T, 21, 3), dans le repère de `shape` : la main de chaque image ; la rotation qui
    pose `shape` sur la paume médiane du clip lui est appliquée telle quelle, sans suivre
    l'orientation du clip, et seules la taille et la position viennent de chaque image du clip.
    `palm` : direction imposée de la paume (`face_palm`). `absolute` : aucune rotation, la main
    garde l'orientation de la vidéo de référence (filmée de face, comme LSFB)."""
    scale_xz = np.array([ASPECT, 1.0, ASPECT])
    clip_m = clip * scale_xz
    median = np.median(clip_m, axis=0)
    options = [shape, shape * np.array([-1.0, 1.0, 1.0])]

    def residual(cand: np.ndarray) -> float:
        r, s, t = _similarity(cand[PALM], median[PALM])
        return float(np.linalg.norm((s * cand[PALM] @ r.T + t) - median[PALM]))
    flip = np.array([-1.0, 1.0, 1.0]) if min(range(2), key=lambda k: residual(options[k])) else np.ones(3)
    shape = shape * flip
    out = np.empty_like(clip_m)
    if absolute:
        frames = motion * flip if motion is not None else np.repeat(shape[None], len(clip_m), axis=0)
        for i, (frame, hand) in enumerate(zip(clip_m, frames)):
            _, s, _ = _similarity(shape[PALM], frame[PALM])
            out[i] = s * (hand - hand[PALM].mean(0)) + frame[PALM].mean(0)
    elif motion is not None:
        r0, _, _ = _similarity(shape[PALM], median[PALM])
        for i, (frame, hand) in enumerate(zip(clip_m, motion * flip)):
            _, s, _ = _similarity(shape[PALM], frame[PALM])
            out[i] = s * (hand - hand[PALM].mean(0)) @ r0.T + frame[PALM].mean(0)
    else:
        for i, frame in enumerate(clip_m):
            r, s, t = _similarity(shape[PALM], frame[PALM])
            out[i] = s * shape @ r.T + t
    if palm in FACING:
        out = face_palm(out, palm)
    return out / scale_xz


def out_of_chest(j: dict[str, np.ndarray]) -> int:
    """Avance chaque main enfoncée dans le torse jusqu'à la surface de la poitrine (en
    place) ; renvoie le nombre d'images corrigées. Seules comptent les mains entre le bassin
    et les épaules, à moins de 0,25 largeur d'épaules derrière elles : une main dans le dos
    vient d'une autre erreur, on ne la ramène pas à travers le corps."""
    sw = float(np.median(np.linalg.norm(j["shoulder.L"] - j["shoulder.R"], axis=1)))
    chest = (j["shoulder.L"] + j["shoulder.R"]) / 2
    moved = np.zeros(len(chest), dtype=bool)
    for s in "LR":
        names = [f"hand.{s}.{k}" for k in range(21)]
        rel = (np.stack([j[n] for n in names], axis=1) - chest[:, None]) / sw     # (T, 21, 3)
        across = np.clip(1 - (rel[..., 0] / CHEST_HALF_WIDTH) ** 2, 0, None)
        top = np.clip(-rel[..., 2] / 0.15, 0, 1)            # poitrine arrondie vers les clavicules
        surface = -CHEST_FRONT * np.sqrt(across) * top                           # −Y = devant
        inside = ((across > 0) & (top > 0) & (rel[..., 2] * sw + chest[:, None, 2] > j["hip"][:, None, 2])
                  & (rel[..., 1] < CHEST_FRONT))
        push = np.where(inside, np.clip(rel[..., 1] - surface, 0, None), 0).max(axis=1) * sw
        # Lissée dans le temps sans repasser sous la surface : maximum sur ±2 images puis
        # moyenne sur 5 (chaque image reste au moins autant avancée qu'il le faut).
        padded = np.pad(push, 2, mode="edge")
        push = np.max([padded[k:k + len(push)] for k in range(5)], axis=0)
        push = np.convolve(np.pad(push, 2, mode="edge"), np.ones(5) / 5, mode="valid")
        for n in names + [f"wrist.{s}"]:
            j[n][:, 1] -= push
        moved |= push > 0
    return int(moved.sum())


def clean(instance: str, letter: bool = False, hand_shape: np.ndarray | None = None,
          motion: dict | None = None, palm: str | None = None,
          absolute: bool = False) -> tuple[dict[str, np.ndarray], dict]:
    load = lambda part: np.load(LSFB / "poses" / part / f"{instance}.npy").astype(np.float64)
    pose = interpolate_nan(load("pose"))
    hands = {"L": interpolate_nan(load("left_hand")), "R": interpolate_nan(load("right_hand"))}
    width = float(np.median(np.linalg.norm(pose[:, SHOULDER["L"], :2] - pose[:, SHOULDER["R"], :2], axis=1)))

    # 1. Main dominante : longueur du trajet de chaque poignet ; lettre : poignet le plus haut
    #    (y image vers le bas).
    if letter:
        mirrored = float(np.mean(pose[:, WRIST["L"], 1])) < float(np.mean(pose[:, WRIST["R"], 1]))
    else:
        path = {s: float(np.linalg.norm(np.diff(hands[s][:, 0, :2], axis=0), axis=1).sum()) for s in "LR"}
        mirrored = path["L"] > 1.3 * path["R"]
    if mirrored:
        pose, hands = mirror(pose, hands)
    if motion is not None:                # durée de la vidéo de référence
        n = max(MIN_FRAMES, round(len(motion["points"]) / motion["fps"] * FPS))
        pose, hands = resample(pose, n), {s: resample(h, n) for s, h in hands.items()}
        moves = resample(np.array(motion["points"], dtype=np.float64), n)
    if hand_shape is not None:            # doigts de l'image de référence, placement du clip
        hands["R"] = fit_hand(hands["R"], hand_shape, moves if motion is not None else None, palm, absolute)

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
    # 6. Mains hors de la poitrine.
    chest = out_of_chest(j)
    meta = {"instance": instance, "retourne": bool(mirrored), "sauts_corriges": spikes, "mains_avancees": chest,
            "images_source": int(len(load("pose"))), "debut": int(start), "fin": int(end),
            "images": int(end - start), "fps": FPS, "mouvement": motion is not None,
            "orientation_video": absolute}
    return j, meta


def main() -> None:
    with open(SELECTION, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    OUT.mkdir(parents=True, exist_ok=True)
    stats = {"retourne": 0, "sauts": 0, "images": [], "erreurs": [], "images_ref": []}
    for r in rows:
        try:
            shape = letter_shape(r["etiquette_lsfb"])
            j, meta = clean(r["instance"], is_letter(r["etiquette_lsfb"]), shape[0] if shape else None,
                            shape[2] if shape else None, shape[3] if shape else None,
                            bool(shape and shape[4]))
            if shape:
                meta["main_image"] = shape[1]
                stats["images_ref"].append(r["etiquette_lsfb"])
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
    # Signes qui ne sont plus dans la sélection (glose passée à une autre variante) : sinon
    # l'index et la galerie de l'évaluation les garderaient.
    kept = {f"{safe_name(r['etiquette_lsfb'])}.npz" for r in rows}
    stale = sorted(p for p in OUT.glob("*.npz") if p.name not in kept)
    for path in stale:
        path.unlink()
    n = len(stats["images"])
    print(f"{n} signes nettoyés → {OUT.relative_to(HERE.parent.parent.parent)}"
          + (f" ; {len(stale)} ancien(s) retiré(s) : {', '.join(p.stem for p in stale)}" if stale else ""))
    if n:
        print(f"  retournés (gauchers) : {stats['retourne']} ; images corrigées (sauts) : {stats['sauts']} ; "
              f"durée médiane : {np.median(stats['images']) / FPS:.2f} s")
    if stats["images_ref"]:
        print(f"  lettres avec la main d'une image de référence : {', '.join(stats['images_ref'])}")
    for e in stats["erreurs"]:
        print(f"  ⚠️  {e}")


if __name__ == "__main__":
    main()
