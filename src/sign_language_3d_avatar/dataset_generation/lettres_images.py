"""Forme de main de chaque lettre de l'alphabet manuel, extraite d'une image ou d'une vidéo
de référence.

Pourquoi : certaines lettres LSFB ont très peu d'exemples vidéo (N : 4) ou des clips peu
nets ; une photo de référence donne une forme de main propre. Seuls les **doigts** viennent
de l'image : `avatar_animation/clean_signs.py` garde le bras, la position et l'orientation
du clip LSFB choisi pour la lettre, et y recale la main de l'image, image par image, sur la
paume du clip (similitude au sens des moindres carrés, main droite ou gauche de l'image).

Entrée, dans `data/alphabet/` (ignoré par git : droits des auteurs), nommée par la lettre :
- une **vidéo** (`A.mp4`, `B.mov`…) d'une personne qui signe la lettre, de préférence : la
  forme retenue est la médiane des images où la main bouge le moins dans la moitié centrale
  de la vidéo (la lettre tenue, `HOLD_FRAMES` images), recalées les unes sur les autres par
  la paume ;
- sinon une **photo** (`N.jpg`, `P.png`…) de vraie main ; MediaPipe lit mal les dessins.
Une vidéo de profil (`K-profil.mp4`) passe avant la vidéo de face : de profil, la flexion
de chaque doigt se voit. Si une lettre a une vidéo et une photo, la vidéo l'emporte, sauf si la page de vérification
(`verif_lettres.csv`, colonnes `verdict` et `image_reference`) a jugé correcte (ou douteuse)
l'autre source : la photo, ou le clip LSFB seul (`image_reference` vide : le `.json` de la
lettre est retiré). Une lettre jugée incorrecte reprend la vidéo en priorité. MediaPipe
suit mal les doigts repliés cachés par la main (dos de la main ou doigts vers la caméra) :
pour ces lettres, filmer la main tournée de sorte que tous les doigts se voient.

Vidéo entière : pour une lettre qui n'est pas encore jugée correcte (ou douteuse) avec cette
vidéo (ou déjà jouée ainsi), la main de **chaque image** est aussi gardée (champ `mouvement`) : l'avatar rejoue
la vidéo, doigts et rotation du poignet compris (K, J, Z…). L'orientation de la vidéo est
prise relativement à la lettre tenue, elle-même posée sur l'orientation du clip LSFB : une
vidéo de profil donne donc la même rotation qu'une vidéo de face. Une lettre jugée correcte
garde la forme sous laquelle elle a été jugée (le rendu validé ne change pas).
Orientation : pour ces mêmes lettres (pas encore jugées correctes), l'orientation de la main
vient aussi de la vidéo (`orientation: video`, vidéo filmée de face comme les clips LSFB) et
non plus du clip ; il ne reste du clip que le bras, la position et la taille de la main.

Doigts imposés : quand la mesure se trompe (doigt à moitié plié, crochet vu comme droit…),
la page de vérification (`verif_lettres.py`) permet de fixer l'état de chaque doigt (tendu,
plié, crochet ; colonnes `pouce`… `auriculaire` de `verif_lettres.csv`). Le doigt est alors
reconstruit dans cet état depuis sa base mesurée, avec ses longueurs mesurées (`pose_fingers`).
Même page, colonne `rotation_max` : rotation du poignet la plus grande (degrés, par rapport à
la lettre tenue) ; celle de la vidéo est réduite à proportion (`limit_rotation`). Colonne
`paume` (bas, haut, avant, soi) : recopiée dans le `.json`, appliquée par `clean_signs.py`.
Ces réglages ne valent que pour la source jugée (`image_reference`) : une nouvelle vidéo
repart de la mesure.

Galerie de l'évaluation : `galerie_lettres.json` (même dossier) garde, pour chaque lettre, la
main qui signe à chaque image de sa vidéo de face (`<LETTRE>.mp4`, sinon de profil), quelle
que soit la source retenue pour l'avatar ; `evaluation/evaluate_avatar.py` y compare les
lettres de l'avatar au lieu des exemples LSFB (option `--lettres videos`, par défaut).

Sortie, dans le même dossier : `<LETTRE>.json` (21 points de la main, en unités de hauteur
d'image) et `<LETTRE>_controle.png` (l'image, ou l'image du milieu de la lettre tenue, avec
les points détectés, à vérifier à l'œil : un doigt mal placé ici le sera aussi sur
l'avatar). Supprimer le `.json` d'une lettre revient au clip LSFB seul.

MediaPipe n'est pas une dépendance du projet : le lancer avec `uv run --with mediapipe`.
Le modèle (`hand_landmarker.task`, Google, Apache 2.0) est téléchargé une fois.

Usage :
    uv run --with mediapipe python src/sign_language_3d_avatar/dataset_generation/lettres_images.py
puis clean_signs.py, sequence.py index et le dictionnaire Blender
(voir avatar_animation/README.md).
"""
from __future__ import annotations

import csv
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "avatar_animation"))
from clean_signs import PALM, _similarity  # noqa: E402

ALPHABET = Path(__file__).resolve().parent / "data" / "alphabet"
MODEL = ALPHABET / "hand_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/"
             "float16/latest/hand_landmarker.task")
EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
VIDEOS = (".mp4", ".mov", ".m4v")
VERIFS = Path(__file__).resolve().parent / "verif_lettres.csv"
GALLERY = ALPHABET / "galerie_lettres.json"
FINGERS = {"pouce": (1, 2, 3, 4), "index": (5, 6, 7, 8), "majeur": (9, 10, 11, 12),
           "annulaire": (13, 14, 15, 16), "auriculaire": (17, 18, 19, 20)}
# Flexion cumulée de chaque phalange vers la paume (degrés : base, milieu, bout).
BENDS = {"tendu": (0, 0, 0), "plié": (80, 100, 60), "crochet": (0, 95, 75)}
HOLD_FRAMES = 9                  # lettre tenue : ~0,3 s à 30 images/s
# Os de la main (MediaPipe) pour l'image de contrôle.
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
         (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


def letter_of(path: Path) -> str | None:
    """`N.jpg` -> N, `K-profil.mp4` -> K ; None pour les autres fichiers."""
    stem = path.stem.upper()
    if path.suffix.lower() not in EXTENSIONS + VIDEOS:
        return None
    if len(stem) == 1 and stem.isalpha():
        return stem
    if len(stem) == 8 and stem.endswith("-PROFIL") and stem[0].isalpha() and path.suffix.lower() in VIDEOS:
        return stem[0]
    return None


def priority(path: Path) -> int:
    """Ordre de préférence des sources d'une lettre : vidéo de profil, vidéo de face, photo."""
    if path.suffix.lower() not in VIDEOS:
        return 2
    return 0 if path.stem.upper().endswith("-PROFIL") else 1


def detector(video: bool = False):
    """Un détecteur MediaPipe ; en mode vidéo, un neuf par vidéo (horodatages croissants)."""
    from mediapipe.tasks.python import BaseOptions, vision
    if not MODEL.exists():
        print("Téléchargement du modèle MediaPipe Hand Landmarker…", flush=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL)
    mode = vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE
    return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)), num_hands=2, running_mode=mode,
        min_hand_detection_confidence=0.3))


def largest_hand(res, w: int, h: int) -> tuple[np.ndarray, int, float] | None:
    """(21 points en unités de hauteur d'image, mains détectées, score) de la plus grande main."""
    if not res.hand_landmarks:
        return None
    hands = [np.array([[p.x * w / h, p.y, p.z * w / h] for p in lm]) for lm in res.hand_landmarks]
    k = int(np.argmax([np.ptp(a[:, 0]) * np.ptp(a[:, 1]) for a in hands]))
    return hands[k], len(hands), float(res.handedness[k][0].score)


def draw(bgr: np.ndarray, pts: np.ndarray, out: Path) -> None:
    import cv2
    h = bgr.shape[0]
    preview = bgr.copy()
    px = lambda p: (int(p[0] * h), int(p[1] * h))
    for a, b in BONES:
        cv2.line(preview, px(pts[a]), px(pts[b]), (40, 200, 40), max(2, h // 250))
    for p in pts:
        cv2.circle(preview, px(p), max(3, h // 150), (30, 30, 230), -1)
    cv2.imwrite(str(out), preview)


def extract(det, path: Path) -> dict | None:
    """La main la plus grande de l'image (21 points, unités de hauteur d'image), ou None."""
    import cv2
    import mediapipe as mp
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    found = largest_hand(det.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                             data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))), w, h)
    if found is None:
        return None
    pts, n, score = found
    draw(bgr, pts, path.with_name(f"{letter_of(path)}_controle.png"))
    return {"lettre": letter_of(path), "image": path.name, "points": pts.round(5).tolist(),
            "mains_detectees": n, "score": round(score, 3)}


def _normalised(pts: np.ndarray) -> np.ndarray:
    """Main recentrée sur la paume et ramenée à une paume de taille 1 (pour comparer les images)."""
    c = pts - pts[PALM].mean(0)
    return c / max(float(np.linalg.norm(c[PALM])), 1e-9)


def detect_video(path: Path) -> tuple[float, list[list[tuple[np.ndarray, str]]]]:
    """(images/s, pour chaque image : [(21 points en unités de hauteur d'image, main
    « Left »/« Right » selon MediaPipe)])."""
    import cv2
    import mediapipe as mp
    det = detector(video=True)
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        h, w = bgr.shape[:2]
        res = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                            data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), int(len(out) * 1000 / fps))
        out.append([(np.array([[p.x * w / h, p.y, p.z * w / h] for p in lm]), res.handedness[k][0].category_name)
                    for k, lm in enumerate(res.hand_landmarks)])
    cap.release()
    det.close()
    return fps, out


def signing_side(videos: dict[Path, list]) -> str | None:
    """La main qui signe (« Left »/« Right » au sens de MediaPipe) : la plus fréquente parmi
    les images où une seule main est visible, toutes vidéos confondues."""
    sides = [hands[0][1] for _, frames in videos.values() for hands in frames if len(hands) == 1]
    return max(set(sides), key=sides.count) if sides else None


def extract_video(path: Path, fps: float, detections: list, side: str | None,
                  motion: bool = False) -> dict | None:
    """Forme de la main qui signe pendant la lettre tenue : fenêtre de HOLD_FRAMES images
    consécutives (main détectée), centrée dans la moitié centrale de la vidéo, où la forme
    change le moins ; images recalées par la paume sur celle du milieu, puis médiane point
    par point. La moitié centrale, car la main est souvent au repos (ouverte, immobile) au
    début et à la fin. Quand deux mains sont visibles (l'autre posée au menton…), celle
    du côté qui signe (`signing_side`).

    `motion` : garde aussi la main de chaque image, de la première à la dernière où elle
    est détectée (trous comblés par l'image voisine), pour les lettres qui bougent (rotation
    du poignet, tracé de Z…) : champ `mouvement`."""
    import cv2
    hands = []
    for found in detections:
        same = [pts for pts, s in found if s == side]
        hands.append(same[0] if same else found[0][0] if len(found) == 1 else None)
    n = min(HOLD_FRAMES, sum(x is not None for x in hands))
    if n < 3:
        return None
    best, best_cost = None, np.inf
    T = len(hands)
    starts = [a for a in range(T - n + 1) if T / 4 <= a + n / 2 <= 3 * T / 4] or range(T - n + 1)
    for a in starts:
        win = hands[a:a + n]
        if any(x is None for x in win):
            continue
        shapes = [_normalised(x) for x in win]
        cost = sum(float(np.linalg.norm(shapes[k + 1] - shapes[k])) for k in range(n - 1))
        if cost < best_cost:
            best, best_cost = a, cost
    if best is None:
        return None
    mid = best + n // 2
    ref = hands[mid]
    aligned = []
    for x in hands[best:best + n]:
        r, s, t = _similarity(x[PALM], ref[PALM])
        aligned.append(s * x @ r.T + t)
    pts = np.median(np.stack(aligned), axis=0)
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ok, bgr = cap.read()
    cap.release()
    if ok:
        draw(bgr, pts, path.with_name(f"{letter_of(path)}_controle.png"))
    out = {"lettre": letter_of(path), "image": path.name, "points": pts.round(5).tolist(),
           "mains_detectees": max(len(f) for f in detections), "score": None,
           "lettre_tenue": [round(best / fps, 2), round((best + n) / fps, 2)],
           "images_avec_main": f"{sum(x is not None for x in hands)}/{len(hands)}"}
    if motion:
        seen = [i for i, x in enumerate(hands) if x is not None]
        seq = [hands[min(seen, key=lambda j: abs(j - i))] for i in range(seen[0], seen[-1] + 1)]
        out["mouvement"] = {"fps": round(fps, 3), "image_tenue": mid - seen[0],
                            "points": np.stack(seq).round(5).tolist()}
    return out


def openness(p: np.ndarray, chain: tuple[int, ...]) -> float:
    """Distance base-bout rapportée à la longueur du doigt : 1 tendu, ~0,4 replié."""
    return float(np.linalg.norm(p[chain[3]] - p[chain[0]])
                 / sum(np.linalg.norm(p[chain[i + 1]] - p[chain[i]]) for i in range(3)))


def palm_side(p: np.ndarray) -> np.ndarray:
    """Normale unitaire de la paume, côté intérieur (là où les doigts se replient)."""
    n = np.cross(p[5] - p[0], p[17] - p[0])
    n /= max(float(np.linalg.norm(n)), 1e-9)
    folded = [c for f, c in FINGERS.items() if f != "pouce" and openness(p, c) < 0.85]
    probe = (np.mean([p[c[3]] - p[c[0]] for c in folded], axis=0) if folded
             else p[4] - p[0])                      # sans doigt replié : côté du pouce
    return n if float(probe @ n) >= 0 else -n


def pose_fingers(p: np.ndarray, states: dict[str, str]) -> np.ndarray:
    """Points de la main avec chaque doigt de `states` reconstruit dans l'état voulu
    (`BENDS`), depuis sa base mesurée et avec ses longueurs mesurées ; le pouce plié
    se pose en travers de la paume, devant le majeur et l'annulaire."""
    p = p.copy()
    n = palm_side(p)
    for finger, state in states.items():
        if state not in BENDS:
            continue
        chain = FINGERS[finger]
        lengths = [float(np.linalg.norm(p[chain[i + 1]] - p[chain[i]])) for i in range(3)]
        if finger == "pouce":
            if state == "tendu":
                d = p[chain[1]] - p[chain[0]]
                d /= max(float(np.linalg.norm(d)), 1e-9)
                for i in range(1, 3):
                    p[chain[i + 1]] = p[chain[i]] + d * lengths[i]
            else:
                palm = float(np.linalg.norm(p[9] - p[0]))
                target = 0.5 * (p[10] + p[14]) + n * 0.15 * palm
                for i in range(1, 3):
                    d = target - p[chain[i]]
                    p[chain[i + 1]] = p[chain[i]] + d / max(float(np.linalg.norm(d)), 1e-9) * lengths[i]
            continue
        d0 = p[chain[0]] - p[0]
        d0 -= (d0 @ n) * n                                  # direction du doigt dans le plan de la paume
        d0 /= max(float(np.linalg.norm(d0)), 1e-9)
        angle = 0.0
        for i, bend in enumerate(BENDS[state]):
            angle += np.radians(bend)
            p[chain[i + 1]] = p[chain[i]] + (d0 * np.cos(angle) + n * np.sin(angle)) * lengths[i]
    return p


def _quat(r: np.ndarray) -> np.ndarray:
    """Quaternion unitaire (w, x, y, z) d'une matrice de rotation."""
    w = np.sqrt(max(0.0, 1 + np.trace(r))) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + r[0, 0] - r[1, 1] - r[2, 2])) / 2, r[2, 1] - r[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - r[0, 0] + r[1, 1] - r[2, 2])) / 2, r[0, 2] - r[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - r[0, 0] - r[1, 1] + r[2, 2])) / 2, r[1, 0] - r[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def _scaled(r: np.ndarray, f: float) -> np.ndarray:
    """La rotation `r` ramenée à f fois son angle, autour du même axe."""
    w, *v = _quat(r)
    v = np.array(v)
    sin = float(np.linalg.norm(v))
    if sin < 1e-9:
        return np.eye(3)
    angle = 2 * np.arctan2(sin, w) * f
    k = v / sin
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * kx @ kx


def rotation_angle(ref: np.ndarray, hand: np.ndarray) -> float:
    """Angle (degrés) de la rotation de paume qui mène `ref` à `hand`."""
    r = _similarity(ref[PALM], hand[PALM])[0]
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))))


def limit_rotation(ref: np.ndarray, frames: list[np.ndarray], most: float) -> list[np.ndarray]:
    """`frames`, avec la rotation de chaque paume par rapport à `ref` (la lettre tenue)
    réduite à proportion, pour que la plus grande fasse `most` degrés (si elle les dépasse)."""
    biggest = max(rotation_angle(ref, f) for f in frames)
    if biggest <= most:
        return frames
    out = []
    for f in frames:
        r = _similarity(ref[PALM], f[PALM])[0]
        c = f[PALM].mean(0)
        out.append((f - c) @ r @ _scaled(r, most / biggest).T + c)
    return out


def settings() -> dict[str, tuple[str, dict[str, str]]]:
    """Lettre -> (source jugée, {rotation_max, paume} non vides) d'après `verif_lettres.csv`."""
    if not VERIFS.exists():
        return {}
    with open(VERIFS, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return {r["lettre"].upper(): (r["image_reference"],
                                  {k: r[k].strip() for k in ("rotation_max", "paume") if (r.get(k) or "").strip()})
            for r in rows}


def video_orientation(path: Path, keep: dict[str, str]) -> bool:
    """Prendre l'orientation de la main dans la vidéo (et non dans le clip LSFB) ? Oui si la
    lettre n'est pas encore jugée correcte (ou douteuse) avec cette vidéo ; une lettre jugée
    garde l'orientation sous laquelle elle a été jugée."""
    if keep.get(letter_of(path)) != path.name:
        return True
    old = ALPHABET / f"{letter_of(path)}.json"
    d = json.loads(old.read_text(encoding="utf-8")) if old.exists() else {}
    return d.get("image") == path.name and d.get("orientation") == "video"


def whole_video(path: Path, keep: dict[str, str]) -> bool:
    """Jouer la vidéo entière ? Oui si la lettre n'est pas encore jugée correcte (ou douteuse)
    avec cette vidéo ; une lettre jugée l'est telle qu'elle était jouée (figée ou entière)."""
    if keep.get(letter_of(path)) != path.name:
        return True
    old = ALPHABET / f"{letter_of(path)}.json"
    d = json.loads(old.read_text(encoding="utf-8")) if old.exists() else {}
    return d.get("image") == path.name and "mouvement" in d


def finger_states() -> dict[str, tuple[str, dict[str, str]]]:
    """Lettre -> (source jugée, {doigt: état imposé}) d'après `verif_lettres.csv` (« mesuré » ou vide : rien)."""
    if not VERIFS.exists():
        return {}
    with open(VERIFS, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return {r["lettre"].upper(): (r["image_reference"], {k: r[k] for k in FINGERS if r.get(k) in BENDS}) for r in rows
            if any(r.get(k) in BENDS for k in FINGERS)}


def approved() -> dict[str, str]:
    """Lettre -> source jugée correcte ou douteuse sur la page de vérification ("" : clip LSFB)."""
    if not VERIFS.exists():
        return {}
    with open(VERIFS, newline="", encoding="utf-8-sig") as f:
        return {r["lettre"].upper(): r["image_reference"] for r in csv.DictReader(f)
                if r["verdict"] in ("correcte", "douteuse")}


def reference_videos(found: dict[str, list[Path]]) -> dict[str, Path]:
    """Lettre -> sa vidéo de référence pour la galerie de l'évaluation : de face si elle
    existe, sinon de profil."""
    out = {}
    for letter, paths in found.items():
        vids = sorted((p for p in paths if p.suffix.lower() in VIDEOS), key=lambda p: -priority(p))
        if vids:
            out[letter] = vids[0]
    return out


def write_gallery(refs: dict[str, Path], videos: dict[Path, tuple], side: str | None) -> None:
    """`galerie_lettres.json` : par lettre, la main qui signe à chaque image de sa vidéo de
    référence (de la première à la dernière image où elle est vue, trous comblés par la
    voisine), en unités de hauteur d'image, côté MediaPipe `cote`."""
    letters = {}
    for letter, path in sorted(refs.items()):
        fps, detections = videos[path]
        hands = []
        for found in detections:
            same = [pts for pts, s in found if s == side]
            hands.append(same[0] if same else found[0][0] if len(found) == 1 else None)
        seen = [i for i, x in enumerate(hands) if x is not None]
        if not seen:
            continue
        seq = [hands[min(seen, key=lambda j: abs(j - i))] for i in range(seen[0], seen[-1] + 1)]
        letters[letter] = {"video": path.name, "fps": round(fps, 3), "points": np.stack(seq).round(4).tolist()}
    GALLERY.write_text(json.dumps({"cote": side, "lettres": letters}), encoding="utf-8")
    print(f"Galerie de l'évaluation : {len(letters)} lettres d'après vos vidéos → {GALLERY.name}")


def main() -> None:
    keep = approved()
    imposed = finger_states()
    tuned = settings()
    found: dict[str, list[Path]] = {}
    for p in sorted(ALPHABET.glob("*")):
        if letter_of(p):
            found.setdefault(letter_of(p), []).append(p)
    sources, clip_only, kept = {}, [], []
    for letter, paths in found.items():
        paths = sorted(paths, key=priority)
        ref = keep.get(letter)
        if ref == "":                                    # clip LSFB jugé correct : on le garde
            clip_only.append(letter)
            for old in (ALPHABET / f"{letter}.json", ALPHABET / f"{letter}_controle.png"):
                old.unlink(missing_ok=True)
            continue
        chosen = next((p for p in paths if p.name == ref), paths[0])
        if chosen.suffix.lower() not in VIDEOS and chosen.name == ref:
            kept.append(f"{letter} ({chosen.name})")
        sources[letter] = chosen
    if not sources and not clip_only:
        sys.exit(f"Rien dans {ALPHABET} : déposer une vidéo ou une photo par lettre (N.mp4, P.png…).")
    if clip_only or kept:
        print("Sources jugées correctes gardées : " + ", ".join(
            [f"{l} (clip LSFB)" for l in sorted(clip_only)] + kept))
    refs = reference_videos(found)
    videos = {p: detect_video(p) for p in {*sources.values(), *refs.values()} if p.suffix.lower() in VIDEOS}
    side = signing_side(videos)
    det = None
    ok, failed = [], []
    for path in sources.values():
        if path.suffix.lower() in VIDEOS:
            hand = extract_video(path, *videos[path], side, motion=whole_video(path, keep))
            if hand is not None and video_orientation(path, keep):
                hand["orientation"] = "video"
        else:
            det = det or detector()
            hand = extract(det, path)
        if hand is None:
            failed.append(path.name)
            continue
        # Réglages de la page : seulement pour la source sur laquelle ils ont été choisis.
        source, fingers = imposed.get(hand["lettre"], ("", {}))
        fingers = fingers if source == path.name else {}
        if fingers:
            hand["points"] = pose_fingers(np.array(hand["points"]), fingers).round(5).tolist()
            hand["doigts_imposes"] = fingers
            if "mouvement" in hand:
                m = hand["mouvement"]
                m["points"] = [pose_fingers(np.array(f), fingers).round(5).tolist() for f in m["points"]]
        source, extra = tuned.get(hand["lettre"], ("", {}))
        extra = extra if source == path.name else {}
        if "mouvement" in hand:
            m = hand["mouvement"]
            frames = [np.array(f) for f in m["points"]]
            m["rotation_mesuree"] = round(max(rotation_angle(np.array(hand["points"]), f) for f in frames))
            if extra.get("rotation_max"):
                most = float(extra["rotation_max"].replace(",", "."))
                m["points"] = [f.round(5).tolist() for f in limit_rotation(np.array(hand["points"]), frames, most)]
                m["rotation_max"] = most
        if extra.get("paume"):
            hand["paume"] = extra["paume"]
        (ALPHABET / f"{hand['lettre']}.json").write_text(json.dumps(hand, ensure_ascii=False), encoding="utf-8")
        ok.append(hand["lettre"] + (f" ({path.name})" if priority(path) != 1 else "")
                  + (" (vidéo entière)" if "mouvement" in hand else "")
                  + (" (orientation de la vidéo)" if hand.get("orientation") == "video" else "")
                  + ((" (2 mains : celle qui signe)" if path.suffix.lower() in VIDEOS else " (2 mains : la plus grande)")
                     if hand["mains_detectees"] > 1 else ""))
    print(f"Formes de main extraites : {', '.join(ok) or 'aucune'}")
    write_gallery(refs, videos, side)
    if failed:
        print(f"⚠️  aucune main détectée : {', '.join(failed)} (photo plus nette, main entière visible)")
    print(f"Vérifier les images *_controle.png dans {ALPHABET}, puis relancer clean_signs.py.")


if __name__ == "__main__":
    main()
