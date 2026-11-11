"""Descripteurs d'un signe et reconnaisseur par plus proches voisins (numpy seul).

Un même descripteur sert pour l'avatar (articulations relevées dans Blender) et pour
les signeurs LSFB (poses nettoyées par `clean_signs.clean`). Il ne dépend ni de la
taille ni des proportions du corps, comme le transfert sur l'avatar :

- **bras** : direction du bras et de l'avant-bras, position du poignet par rapport à
  l'épaule rapportée à la longueur du bras, **dans le plan de l'image** (x, z) : la
  profondeur des bras est reconstruite (`pose3d.lift`), donc peu fiable ;
- **orientation des mains** : axe des doigts et normale de la paume (repère du corps) ;
- **configuration des mains** : direction de chaque segment de doigt dans le repère de
  la main (indépendante de l'orientation du bras).

Chaque groupe pèse autant dans la distance, et la main dominante (droite : les clips
de gauchers sont retournés) compte double. Choix mesurés sur les vrais signeurs seuls
(un signeur laissé de côté, 330 signes) : top-5 24 % -> 29 %, top-10 34 % -> 40 %.
Le reconnaisseur ramène chaque séquence à
`LENGTH` images puis compare par DTW (alignement temporel souple) à une galerie
d'exemples réels ; la classe d'un exemple est son étiquette LSFB.
"""
from __future__ import annotations

import numpy as np

LENGTH = 24
DOMINANT = 2.0                # poids de la main / du bras droits
FINGER_CHAINS = ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12), (0, 13, 14, 15, 16),
                 (0, 17, 18, 19, 20))
SEGMENTS = [(c[i], c[i + 1]) for c in FINGER_CHAINS for i in range(4)]   # 20 segments
GROUPS = ("bras", "orientation", "configuration")


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def hand_frame(j: dict[str, np.ndarray], s: str) -> np.ndarray:
    """Repère de la main (T, 3, 3) : lignes = axe des doigts, travers, normale de la paume."""
    y = _unit(j[f"hand.{s}.9"] - j[f"hand.{s}.0"])
    across = j[f"hand.{s}.5"] - j[f"hand.{s}.17"]
    x = _unit(across - (across * y).sum(-1, keepdims=True) * y)
    return np.stack([y, x, np.cross(y, x)], axis=1)


def describe(j: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Descripteurs (T, d) par groupe."""
    arms, orient, shape = [], [], []
    for s in "LR":
        sh, el, wr = j[f"shoulder.{s}"], j[f"elbow.{s}"], j[f"wrist.{s}"]
        length = np.linalg.norm(el - sh, axis=-1, keepdims=True) + np.linalg.norm(wr - el, axis=-1, keepdims=True)
        arms += [_unit(el - sh), _unit(wr - el), (wr - sh) / np.maximum(length, 1e-9)]
        frame = hand_frame(j, s)
        orient += [frame[:, 0], frame[:, 2]]
        for a, b in SEGMENTS:
            d = _unit(j[f"hand.{s}.{b}"] - j[f"hand.{s}.{a}"])
            shape.append(np.einsum("tij,tj->ti", frame, d))            # dans le repère de la main
    return {"bras": np.concatenate(arms, 1), "orientation": np.concatenate(orient, 1),
            "configuration": np.concatenate(shape, 1)}


def vector(j: dict[str, np.ndarray]) -> np.ndarray:
    """Descripteur pondéré (T, D) : chaque groupe a la même norme moyenne, bras dans le
    plan de l'image, côté droit (seconde moitié de chaque groupe) compté double."""
    d = describe(j)
    out = []
    for g in GROUPS:
        x = d[g]
        if g == "bras":
            x = x.reshape(len(x), -1, 3)[..., [0, 2]].reshape(len(x), -1)
        w = np.ones(x.shape[1])
        w[x.shape[1] // 2:] = DOMINANT
        out.append(x * w / np.sqrt(x.shape[1]))
    return np.concatenate(out, axis=1)


def resample(x: np.ndarray, n: int = LENGTH) -> np.ndarray:
    t = np.linspace(0, len(x) - 1, n)
    k = np.clip(t.astype(int), 0, len(x) - 2) if len(x) > 1 else np.zeros(n, int)
    w = (t - k)[:, None]
    nxt = np.minimum(k + 1, len(x) - 1)
    return x[k] * (1 - w) + x[nxt] * w


def dtw_batch(q: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Distance DTW de q (L, D) à chaque séquence de g (P, L, D), normalisée par la longueur."""
    cost = np.sqrt(np.maximum(((q[None, :, None] - g[:, None]) ** 2).sum(-1), 0))   # (P, L, L)
    P, L, _ = cost.shape
    acc = np.full((P, L + 1, L + 1), np.inf)
    acc[:, 0, 0] = 0.0
    for i in range(1, L + 1):
        for k in range(1, L + 1):
            acc[:, i, k] = cost[:, i - 1, k - 1] + np.minimum(np.minimum(acc[:, i - 1, k], acc[:, i, k - 1]),
                                                              acc[:, i - 1, k - 1])
    return acc[:, L, L] / (2 * L)


class Gallery:
    """Exemples réels (étiquette, signeur, instance, descripteur rééchantillonné)."""

    def __init__(self, labels: list[str], signers: list[str], instances: list[str], seqs: np.ndarray):
        self.labels, self.signers, self.instances = np.array(labels), np.array(signers), np.array(instances)
        self.seqs = seqs.astype(np.float32)

    def rank(self, query: np.ndarray, exclude_signer: str | None = None) -> list[tuple[str, float]]:
        """Classes triées de la plus proche à la plus lointaine (distance au plus proche
        exemple de la classe). Les exemples du signeur exclu sont ignorés : on ne doit pas
        reconnaître un signe parce qu'on reconnaît son signeur."""
        keep = self.signers != exclude_signer if exclude_signer else np.ones(len(self.labels), bool)
        d = dtw_batch(resample(query).astype(np.float32), self.seqs[keep])
        best: dict[str, float] = {}
        for lab, dist in zip(self.labels[keep], d):
            if dist < best.get(lab, np.inf):
                best[lab] = float(dist)
        return sorted(best.items(), key=lambda x: x[1])


def rank_of(ranking: list[tuple[str, float]], label: str) -> tuple[int, int] | None:
    """(rang (1 = premier) de l'étiquette attendue, nombre de classes), None si elle n'est
    pas dans la galerie."""
    return next(((i + 1, len(ranking)) for i, (lab, _) in enumerate(ranking) if lab == label), None)


def percentile(r: tuple[int, int] | None) -> float | None:
    """Rang centile : part des autres signes classés derrière le bon (1 = premier,
    0,5 = hasard). Continu, il reste informatif quand le top-1 est bas."""
    return None if r is None else 1 - (r[0] - 1) / max(1, r[1] - 1)


def angle_error(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, float]:
    """Écart angulaire moyen (degrés) entre deux réalisations de même durée, par groupe :
    bras (bras + avant-bras) et configuration des doigts."""
    da, db = describe(a), describe(b)
    out = {}
    for g, cols in (("bras", [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14]), ("configuration", None)):
        x, y = da[g], db[g]
        if cols is not None:
            x, y = x[:, cols], y[:, cols]
        x, y = x.reshape(len(x), -1, 3), y.reshape(len(y), -1, 3)
        cos = np.clip((_unit(x) * _unit(y)).sum(-1), -1, 1)
        out[g] = float(np.degrees(np.arccos(cos)).mean())
    return out
