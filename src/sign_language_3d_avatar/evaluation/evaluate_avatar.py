"""Évaluation de l'avatar seul : signe-t-il bien des phrases déjà en gloses LSFB ?

La traduction n'intervient pas : les phrases du banc d'essai sont des gloses de
référence. Par défaut, `banc_avatar.jsonl` : les 119 phrases de `src/eval/dataset.jsonl`
et `src/eval/dev.jsonl`, traduites en gloses LSFB par Claude avec les seuls signes du
dictionnaire (8 phrases reformulées quand un mot n'a aucun signe : `texte_original`,
`note`). Les fichiers de `src/eval/` restent la référence du traducteur.
L'avatar (mannequin ou VRoid) joue chaque signe du dictionnaire et chaque phrase dans
Blender ; on relève ses articulations (`export_avatar_blender.py`) et on mesure :

| mesure | question |
|---|---|
| couverture | chaque glose des phrases est-elle jouée par un vrai signe LSFB ? |
| reconnaissance des signes isolés | un reconnaisseur retrouve-t-il le bon signe en regardant l'avatar ? |
| reconnaissance en contexte | et une fois le signe enchaîné dans une phrase (transitions) ? |
| qualité du mouvement | vitesses ou à-coups anormaux, main dans le torse |
| phrases | la phrase entière est-elle jouable (aucun mot sans signe), et chacun de ses signes reconnu ? |

Le reconnaisseur compare l'avatar aux exemples **réels** de LSFB-ISOL (~9 signeurs par
signe, `features.py`). Le signeur du clip copié par l'avatar est exclu de la galerie :
sinon on mesurerait seulement que l'avatar copie bien son clip. La **référence** est le
même test passé sur ce clip source, joué par le vrai signeur : l'écart entre les deux
est exactement ce que perdent le transfert sur l'avatar et l'enchaînement en phrase.
Un clip d'un autre signeur sert de second repère : s'il est bien reconnu alors que le
clip source ne l'est pas, c'est le choix du clip (`select_examples.py`) qui est en cause.

Le reconnaisseur (plus proches voisins, ~9 exemples par signe, ~400 signes) est faible
en top-1 même sur de vrais signeurs (~10 %) : le score s'appuie donc sur le **rang
centile** du bon signe (part des autres signes classés derrière lui : 100 % = premier,
50 % = hasard), continu et mesuré de la même façon pour l'avatar et pour les humains.

**Chiffre de tête** : (centile en contexte − 50 %) ÷ (centile des clips sources − 50 %),
soit la part de la reconnaissabilité du vrai signeur que l'avatar conserve au-dessus du
hasard (100 % = aussi reconnaissable que le signeur qu'il copie). Les pointages (générés) et les mots sans signe
ne sont pas reconnaissables par la galerie : ils comptent dans la couverture seulement. Les
lettres des mots épelés (`FS:A`…`FS:Z`) sont reconnues à part (ligne « Lettres ») et ne
comptent pas dans le chiffre de tête ; elles comptent dans « phrases reconnues ».
La fidélité au clip source (écart angulaire bras / doigts) explique les signes ratés.

    uv run python src/sign_language_3d_avatar/evaluation/evaluate_avatar.py [--avatar mon_avatar.vrm] \\
        [--phrases fichier.jsonl ...] [--save] [--label "…"]

La galerie ne garde que la moitié « eval » des signeurs (`features.signer_half`) : l'autre
moitié sert à choisir les clips du dictionnaire (`select_examples.py`). `--galerie toutes`
reprend l'ancienne galerie (lignes de RESULTS.md antérieures au 2026-11-11 18 h).

Lettres (`--lettres videos`, par défaut) : les lettres de l'avatar, isolées et dans les mots
épelés, sont comparées aux **vidéos de lettres filmées** du projet (une par lettre,
`dataset_generation/data/alphabet/galerie_lettres.json`, écrit par `lettres_images.py`) et
non aux exemples LSFB, avec la main seule (orientation et doigts, `features.hand_vector` :
les vidéos ne montrent pas le bras) ; rang parmi les 26 lettres. Les lettres LSFB sont alors
retirées de la galerie des signes. Comme l'avatar tire ses lettres de ces mêmes vidéos, la
mesure dit s'il les reproduit et les distingue, pas si un autre signeur les reconnaîtrait.
`--lettres lsfb` reprend les exemples LSFB (runs antérieurs au 2026-11-15 19 h).

Signes vérifiés à l'œil (`evaluation/signes_verifies.csv`) : signes jugés corrects par une
personne alors que le reconnaisseur ne les reconnaît pas (aucun clip réel ne passe non plus).
Ils ne figurent plus parmi les signes bloquants, et « phrases reconnues » donne un second
chiffre qui les compte comme reconnus ; le premier chiffre reste la mesure automatique seule.
Un verdict ne vaut que pour le clip relu (colonne `instance`) : si le clip change, le signe
est de nouveau mesuré comme les autres.

`--save` archive le run (JSON dans `evaluation/results/`, ligne dans `RESULTS.md`).
Données dérivées de LSFB-ISOL (CC BY-NC-SA 4.0) : les caches restent dans
`avatar_animation/data/dictionnaire/evaluation/` (ignoré par git) ; les records ne
contiennent que des étiquettes et des scores.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
PROJECT = BASE.parent.parent
sys.path.insert(0, str(BASE / "avatar_animation"))
sys.path.insert(0, str(HERE))
from clean_signs import LSFB, clean, is_letter  # noqa: E402
from features import Gallery, angle_error, hand_vector, percentile, rank_of, resample, signer_half, vector  # noqa: E402
from pose3d import FPS, to_blender, unstack  # noqa: E402
from sequence import DICT, INDEX, load_index, load_sign, resolve  # noqa: E402

CACHE = DICT / "evaluation"
LETTER_GALLERY = BASE / "dataset_generation" / "data" / "alphabet" / "galerie_lettres.json"
RESULTS = HERE / "results"
TABLE = HERE / "RESULTS.md"
VERIFIED = HERE / "signes_verifies.csv"        # signes jugés corrects à l'œil
SELECTION = BASE / "dataset_generation" / "selection.csv"   # critère « relecture » : clip choisi à l'œil
EXPORT = HERE / "export_avatar_blender.py"
DEFAULT_BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"
DEFAULT_BENCH = [HERE / "banc_avatar.jsonl"]
HAND_POINTS = (0, 4, 8, 12, 16, 20)      # poignet et bouts des doigts


def sha(*parts: bytes) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.hexdigest()[:12]


def git(*a: str) -> str:
    return subprocess.run(["git", *a], capture_output=True, text=True, cwd=PROJECT).stdout.strip()


# ------------------------------------------------------------------- données
def load_bench(paths: list[Path]) -> tuple[list[list[str]], str]:
    phrases, raw = [], b""
    for p in paths:
        data = p.read_bytes()
        raw += data
        phrases += [json.loads(l)["glosses"] for l in data.decode("utf-8").splitlines() if l.strip()]
    return phrases, sha(raw)


def build_letter_gallery() -> Gallery:
    """Lettres filmées du projet (`galerie_lettres.json`) : main qui signe ramenée à une main
    droite (miroir si MediaPipe la voit gauche), repère Blender, descripteur de la main seule."""
    if not LETTER_GALLERY.exists():
        sys.exit(f"{LETTER_GALLERY} absent : lancer d'abord dataset_generation/lettres_images.py "
                 "(ou --lettres lsfb)")
    d = json.loads(LETTER_GALLERY.read_text(encoding="utf-8"))
    flip = np.array([-1.0, 1.0, 1.0]) if d["cote"] == "Left" else np.ones(3)
    labs, insts, seqs = [], [], []
    for letter, v in sorted(d["lettres"].items()):
        h = to_blender(np.array(v["points"], dtype=np.float64) * flip, 1.0)
        labs.append(f"FS:{letter}"), insts.append(v["video"])
        seqs.append(resample(hand_vector({f"hand.R.{k}": h[:, k] for k in range(21)})))
    return Gallery(labs, ["vidéos du projet"] * len(labs), insts, np.stack(seqs))


def build_gallery(index: dict, half: str | None = "eval", letters: bool = True) -> tuple[Gallery, dict]:
    """Exemples réels LSFB-ISOL des signes du dictionnaire, nettoyés comme ceux de
    l'avatar (droitier, sauts, lissage, découpe), et référence humaine du mouvement
    (seuils de vitesse / accélération = 99,5e centile des vrais signeurs) ; mis en cache.
    `half` : seulement les signeurs de cette moitié (`features.signer_half`) ; l'autre a
    servi à choisir les clips du dictionnaire. `letters` False : sans les lettres (FS:…)."""
    labels = {l for l in index["signes"] if letters or not is_letter(l)}
    with open(LSFB / "instances.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["sign"] in labels
                and (half is None or signer_half(r["signer"]) == half)
                and (LSFB / "poses" / "pose" / f"{r['id']}.npy").exists()]
    key = sha("|".join(sorted(r["id"] for r in rows)).encode(), (BASE / "avatar_animation" / "clean_signs.py").read_bytes(),
              (HERE / "features.py").read_bytes())
    path = CACHE / f"galerie_{key}.npz"
    if path.exists():
        d = np.load(path)
        return (Gallery(list(d["labels"]), list(d["signers"]), list(d["instances"]), d["seqs"]),
                json.loads(str(d["reference"])))
    print(f"Galerie : {len(rows)} exemples réels à préparer (une fois, mis en cache)…", flush=True)
    labs, signers, insts, seqs, speeds = [], [], [], [], []
    for k, r in enumerate(rows):
        try:
            j, _ = clean(r["id"], is_letter(r["sign"]))
        except Exception:
            continue
        labs.append(r["sign"]), signers.append(r["signer"]), insts.append(r["id"])
        seqs.append(resample(vector(j)))
        speeds.append(motion_stats(j))
        if k % 500 == 0:
            print(f"  {k}/{len(rows)}", flush=True)
    ref = {k: float(np.percentile(np.concatenate([s[k] for s in speeds]), 99.5)) for k in ("vitesse", "acceleration")}
    ref["torse"] = float(np.mean(np.concatenate([s["torse"] for s in speeds])))
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, labels=np.array(labs), signers=np.array(signers), instances=np.array(insts),
                        seqs=np.stack(seqs).astype(np.float32), reference=json.dumps(ref))
    return Gallery(labs, signers, insts, np.stack(seqs)), ref


def export_avatar(index: dict, phrases: list[list[str]], args) -> tuple[dict, dict]:
    """Articulations de l'avatar pour chaque signe et chaque phrase (Blender, mis en cache)."""
    job = {"signes": sorted(index["signes"]), "phrases": phrases}
    avatar = Path(args.avatar).expanduser().resolve() if args.avatar else None
    key = sha(json.dumps(job, ensure_ascii=False).encode(), INDEX.read_bytes(),
              (avatar.read_bytes() if avatar else b"mannequin"), EXPORT.read_bytes(),
              (BASE / "avatar_animation" / "avatar_rig.py").read_bytes(),
              (BASE / "avatar_animation" / "sequence.py").read_bytes(),
              # Les signes nettoyés eux-mêmes : un nouveau nettoyage garde souvent le même index.
              *((DICT / index["signes"][s]["fichier"]).read_bytes() for s in job["signes"]))
    out = CACHE / f"avatar_{key}.npz"
    if not out.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        jobfile = CACHE / f"travail_{key}.json"
        jobfile.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
        cmd = [args.blender, "-b", "--factory-startup", "--python", str(EXPORT), "--",
               "--travail", str(jobfile), "--sortie", str(out)] + (["--avatar", str(avatar)] if avatar else [])
        print(f"Avatar : {len(job['signes'])} signes et {len(phrases)} phrases joués dans Blender…", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not out.exists():
            sys.exit("Échec de Blender :\n" + "\n".join((proc.stdout + proc.stderr).splitlines()[-20:]))
    d = np.load(out)
    names = list(d["noms"])
    items = {k: unstack(names, d[k].astype(np.float64)) for k in d.files if "::" in k}
    return items, json.loads(str(d["meta"])) | {"cache": out.name}


# ------------------------------------------------------------------- mesures
def motion_stats(j: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Par image : vitesse et accélération max des mains (largeurs d'épaules par s et par
    s²), et main enfoncée dans le torse (0/1)."""
    sw = float(np.median(np.linalg.norm(j["shoulder.L"] - j["shoulder.R"], axis=1)))
    pts = np.stack([j[f"hand.{s}.{k}"] for s in "LR" for k in HAND_POINTS], axis=1) / sw   # (T, 12, 3)
    vel = np.diff(pts, axis=0) * FPS
    speed = np.linalg.norm(vel, axis=2).max(axis=1)
    acc = np.linalg.norm(np.diff(vel, axis=0) * FPS, axis=2).max(axis=1)
    # Torse : cylindre elliptique sous les épaules (demi-largeur 0,4, demi-profondeur 0,2,
    # hauteur 1,3 en largeurs d'épaules), plus petit que le corps : on ne compte que
    # les mains franchement dedans, pas celles qui le touchent.
    chest = ((j["shoulder.L"] + j["shoulder.R"]) / 2)[:, None] / sw
    rel = pts - chest
    inside = ((rel[..., 0] / 0.4) ** 2 + (rel[..., 1] / 0.2) ** 2 < 1) & (rel[..., 2] < -0.15) & (rel[..., 2] > -1.3)
    return {"vitesse": speed, "acceleration": acc, "torse": inside.any(axis=1).astype(float)}


def top(ranks: list[tuple[int, int] | None], k: int) -> float:
    return sum(r is not None and r[0] <= k for r in ranks) / max(1, len(ranks))


def mean_centile(ranks: list[tuple[int, int] | None]) -> float:
    vals = [percentile(r) for r in ranks if r is not None]
    return float(np.mean(vals)) if vals else float("nan")


def evaluate(args) -> dict:
    index = load_index()
    bench_paths = [Path(p) for p in args.phrases] if args.phrases else DEFAULT_BENCH
    phrases, bench_hash = load_bench(bench_paths)
    half = None if args.galerie == "toutes" else "eval"
    gallery, ref = build_gallery(index, half, letters=args.lettres == "lsfb")
    letter_gallery = build_letter_gallery() if args.lettres == "videos" else None
    items, meta = export_avatar(index, phrases, args)
    t0 = time.time()

    # 1. Couverture des gloses du banc.
    kinds = {"signe": 0, "composé": 0, "épelé": 0, "pointage": 0, "absent": 0}
    for g in (g for p in phrases for g in p):
        kinds[resolve(g, index)["type"]] += 1
    n_gloss = sum(kinds.values())

    # 2. Signes isolés : avatar, clip source joué par le vrai signeur, clip d'un autre
    #    signeur, et fidélité de l'avatar au clip source.
    signs = []
    real_by_label: dict[str, list[int]] = {}
    for i, lab in enumerate(gallery.labels):
        real_by_label.setdefault(str(lab), []).append(i)
    print(f"Reconnaissance de {len(index['signes'])} signes isolés…", flush=True)
    for lab, info in sorted(index["signes"].items()):
        av = items[f"signe::{lab}"]
        if letter_gallery is not None and is_letter(lab):      # lettre : vidéos filmées, main seule
            rank = rank_of(letter_gallery.rank(hand_vector(av)), lab)
            s_rank = rank_of(letter_gallery.rank(hand_vector(clean(info["instance"], True)[0])), lab)
            fid = angle_error(av, load_sign(lab, index))
            signs.append({"etiquette": lab, "rang_avatar": rank and rank[0], "rang_clip_source": s_rank and s_rank[0],
                          "rang_autre_signeur": None, "centile_avatar": percentile(rank),
                          "centile_clip_source": percentile(s_rank), "centile_autre_signeur": None,
                          "_r": rank, "_s": s_rank, "_h": None, "exemples_galerie": 1,
                          "ecart_bras_deg": round(fid["bras"], 1), "ecart_doigts_deg": round(fid["configuration"], 1)})
            continue
        rank = rank_of(gallery.rank(vector(av), exclude_signer=info["signeur"]), lab)
        own = [i for i in real_by_label.get(lab, []) if gallery.instances[i] == info["instance"]]
        if own:
            s_rank = rank_of(_rank_real(gallery, own[0]), lab)
        else:                                 # clip source de l'autre moitié des signeurs
            src_seq = resample(vector(clean(info["instance"], is_letter(lab))[0]))
            s_rank = rank_of(gallery.rank(src_seq, exclude_signer=info["signeur"]), lab)
        others = sorted(i for i in real_by_label.get(lab, []) if gallery.signers[i] != info["signeur"])
        h_rank = rank_of(_rank_real(gallery, others[0]), lab) if others else None
        fid = angle_error(av, load_sign(lab, index))
        signs.append({"etiquette": lab, "rang_avatar": rank and rank[0], "rang_clip_source": s_rank and s_rank[0],
                      "rang_autre_signeur": h_rank and h_rank[0], "centile_avatar": percentile(rank),
                      "centile_clip_source": percentile(s_rank), "centile_autre_signeur": percentile(h_rank),
                      "_r": rank, "_s": s_rank, "_h": h_rank,
                      "exemples_galerie": len(others), "ecart_bras_deg": round(fid["bras"], 1),
                      "ecart_doigts_deg": round(fid["configuration"], 1)})

    # 3. Signes en contexte et qualité du mouvement des phrases.
    print(f"Reconnaissance en contexte ({len(phrases)} phrases)…", flush=True)
    ctx, ctx_labels, letters, quality, per_phrase = [], [], [], {"vitesse": 0, "acceleration": 0, "torse": 0, "images": 0}, []
    for k, ph in enumerate(meta["phrases"]):
        j = items[f"phrase::{k}"]
        found = []
        for start, end, gloss, kind, _, lab in ph["chronologie"]:
            if kind not in ("signe", "lettre"):
                continue
            seg = {n: v[start:end] for n, v in j.items()}
            if kind == "lettre" and letter_gallery is not None:
                r = rank_of(letter_gallery.rank(hand_vector(seg)), lab)
            else:
                r = rank_of(gallery.rank(vector(seg), exclude_signer=index["signes"][lab]["signeur"]), lab)
            if kind == "lettre":              # mesurées à part : le score reste celui des vrais signes
                letters.append(r)
            else:
                ctx.append(r)
                ctx_labels.append(lab)
            found.append({"glose": gloss, "etiquette": lab, "rang": r and r[0], "centile": percentile(r)}
                         | ({"lettre": True} if kind == "lettre" else {}))
        m = motion_stats(j)
        bad = {"vitesse": m["vitesse"] > ref["vitesse"], "acceleration": m["acceleration"] > ref["acceleration"]}
        quality["vitesse"] += int(bad["vitesse"].sum())
        quality["acceleration"] += int(bad["acceleration"].sum())
        quality["torse"] += int(m["torse"].sum())
        quality["images"] += len(m["torse"])
        per_phrase.append({"gloses": ph["gloses"], "signes": found,
                           "absents": [g for g in ph["gloses"] if resolve(g, index)["type"] == "absent"],
                           "images_defaut": int(bad["vitesse"].sum() + bad["acceleration"].sum() + m["torse"].sum())})

    with open(SELECTION, newline="", encoding="utf-8") as f:
        by_eye = {r["etiquette_lsfb"] for r in csv.DictReader(f) if r["critere"] == "relecture"}
    for s in signs:
        s["clip_relu"] = s["etiquette"] in by_eye
    iso_av = [s.pop("_r") for s in signs]
    iso_s = [x for x in (s.pop("_s") for s in signs) if x is not None]
    iso_h = [x for x in (s.pop("_h") for s in signs) if x is not None]
    src = {s["etiquette"]: s["centile_clip_source"] for s in signs}
    c_iso, c_src, c_ctx = mean_centile(iso_av), mean_centile(iso_s), mean_centile(ctx)
    # Référence du contexte : les clips sources, pondérés comme les signes des phrases.
    c_src_ctx = float(np.mean([src[l] for l in ctx_labels if src.get(l) is not None] or [np.nan]))
    gain = lambda c, ref: (c - 0.5) / (ref - 0.5) if ref > 0.5 else None
    n_img = max(1, quality["images"])
    metrics = {
        "couverture": {"signe": (kinds["signe"] + kinds["composé"]) / max(1, n_gloss),
                       "epele": kinds["épelé"] / max(1, n_gloss), "pointage": kinds["pointage"] / max(1, n_gloss),
                       "absent": kinds["absent"] / max(1, n_gloss), "gloses": n_gloss},
        "isoles": {"centile": c_iso, "top1": top(iso_av, 1), "top5": top(iso_av, 5),
                   "top10": top(iso_av, 10), "n": len(iso_av)},
        "contexte": {"centile": c_ctx, "top1": top(ctx, 1), "top5": top(ctx, 5), "top10": top(ctx, 10),
                     "n": len(ctx), "centile_clips_sources": c_src_ctx},
        "lettres": {"centile": mean_centile(letters), "top1": top(letters, 1), "top5": top(letters, 5),
                    "top10": top(letters, 10), "n": len(letters),
                    "centile_clips_sources": float(np.mean([src[s["etiquette"]] for p in per_phrase for s in p["signes"]
                                                            if s.get("lettre") and src.get(s["etiquette"]) is not None]
                                                           or [np.nan]))},
        "clip_source": {"centile": c_src, "top1": top(iso_s, 1), "top5": top(iso_s, 5),
                        "top10": top(iso_s, 10), "n": len(iso_s)},
        "autre_signeur": {"centile": mean_centile(iso_h), "top1": top(iso_h, 1), "top5": top(iso_h, 5),
                          "top10": top(iso_h, 10), "n": len(iso_h)},
        "score": gain(c_ctx, c_src_ctx),
        "score_isoles": gain(c_iso, c_src),
        "mouvement": {"vitesse_anormale": quality["vitesse"] / n_img, "a_coups": quality["acceleration"] / n_img,
                      "main_dans_torse": quality["torse"] / n_img,
                      "images_sans_defaut": 1 - (quality["vitesse"] + quality["acceleration"] + quality["torse"]) / n_img,
                      "reference_humaine_main_dans_torse": ref["torse"]},
        "fidelite": {"bras_deg": float(np.median([s["ecart_bras_deg"] for s in signs])),
                     "doigts_deg": float(np.median([s["ecart_doigts_deg"] for s in signs]))},
        "phrases": phrase_scores(per_phrase, src, signs, verified_signs(index)),
    }
    print(f"(mesures en {time.time() - t0:.0f} s)")
    return {"timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), "label": args.label,
            "avatar": meta["avatar"], "dictionnaire": sha(INDEX.read_bytes()), "banc": bench_hash,
            "banc_fichiers": [str(p.relative_to(PROJECT)) for p in bench_paths], "phrases": len(phrases),
            "galerie": {"exemples": len(gallery.labels), "signes": len(set(gallery.labels)),
                        "signeurs": len(set(gallery.signers)), "moitie": half or "toutes",
                        "lettres": args.lettres,
                        "lettres_videos": len(letter_gallery.labels) if letter_gallery is not None else 0},
            "code": sha(*(p.read_bytes() for p in (HERE / "features.py", Path(__file__), EXPORT))),
            "git_commit": git("rev-parse", "--short", "HEAD") or None, "git_dirty": bool(git("status", "--porcelain")),
            "metrics": metrics, "signes": signs, "par_phrase": per_phrase}


PHRASE_THRESHOLDS = (0.8, 0.9)            # rang centile minimal de chaque signe (top 20 % / top 10 %)


def verified_signs(index: dict) -> set[str]:
    """Signes jugés corrects à l'œil, pour le clip que l'avatar copie aujourd'hui."""
    if not VERIFIED.exists():
        return set()
    with open(VERIFIED, newline="", encoding="utf-8-sig") as f:
        return {r["etiquette_lsfb"] for r in csv.DictReader(f) if r["verdict"] == "correct"
                and index["signes"].get(r["etiquette_lsfb"], {}).get("instance") == r["instance"]}


def beyond_measure(signs: list[dict], th: float) -> set[str]:
    """Signes hors de portée du reconnaisseur au seuil `th` : ni le clip humain copié ni le clip
    d'un autre signeur ne passent, et l'avatar fait au moins aussi bien que le clip copié (à 10
    points près). Changer de clip ou corriger l'avatar n'y peut rien : la galerie a trop peu
    d'exemples du signe, ou des exemples signés autrement."""
    return {s["etiquette"] for s in signs
            if s["centile_clip_source"] is not None and s["centile_clip_source"] < th
            and (s["centile_autre_signeur"] is None or s["centile_autre_signeur"] < th)
            and (s["centile_avatar"] or 0) >= s["centile_clip_source"] - 0.1}


def phrase_scores(per_phrase: list[dict], src: dict[str, float | None], signs: list[dict],
                  verified: set[str] = frozenset()) -> dict:
    """La phrase entière est-elle bien signée ? « Jouable » : chaque glose a un vrai signe
    ou un pointage. « Reconnue au seuil s » : jouable et chaque vrai signe, en contexte,
    a un rang centile ≥ s. Référence : la même phrase où chaque signe est remplacé par son
    clip source joué par le vrai signeur. Les signes que la galerie ne classe pas
    (centile inconnu) ne comptent ni pour l'avatar ni pour la référence. « Limite de la
    mesure » : phrases jouables qui échouent seulement à cause de signes hors de portée du
    reconnaisseur (`beyond_measure`). « Avec vérifiés » : les signes jugés corrects à l'œil
    (`verified_signs`) comptent comme reconnus."""
    n = max(1, len(per_phrase))
    playable = [not p["absents"] for p in per_phrase]
    out = {"n": len(per_phrase), "jouables": sum(playable) / n, "reconnues": {}, "reconnues_clips_sources": {},
           "limite_mesure": {}, "reconnues_avec_verifies": {}, "signes_bloquants": {}}
    for th in PHRASE_THRESHOLDS:
        limit = beyond_measure(signs, th)
        av = hu = lim = ver = 0
        for p, ok in zip(per_phrase, playable):
            measured = [s for s in p["signes"] if s["centile"] is not None]
            failing = {s["etiquette"] for s in measured if s["centile"] < th}
            av += ok and not failing
            hu += ok and all((src.get(s["etiquette"]) or 0) >= th for s in measured)
            lim += ok and bool(failing) and failing <= limit
            ver += ok and failing <= verified
        out["reconnues"][f"{th:.0%}"] = av / n
        out["reconnues_clips_sources"][f"{th:.0%}"] = hu / n
        out["limite_mesure"][f"{th:.0%}"] = lim / n
        out["reconnues_avec_verifies"][f"{th:.0%}"] = ver / n
        out["signes_bloquants"][f"{th:.0%}"] = [b | {"limite": b["etiquette"] in limit, "verifie": b["etiquette"] in verified}
                                                for b in blocking_signs(per_phrase, th)]
    return out


def blocking_signs(per_phrase: list[dict], th: float) -> list[dict]:
    """Signes et lettres épelées (FS:N : lettre N d'un mot épelé, pas un signe à part) sous le seuil dans des phrases jouables : combien de
    phrases ils font échouer, dont combien à eux seuls. Du plus bloquant au moins bloquant."""
    count: dict[str, list[int]] = {}
    for p in per_phrase:
        if p["absents"]:
            continue
        bad = {s["etiquette"] for s in p["signes"] if s["centile"] is not None and s["centile"] < th}
        for lab in bad:
            c = count.setdefault(lab, [0, 0])
            c[0] += 1
            c[1] += len(bad) == 1
    return [{"etiquette": lab, "phrases": n, "seul": alone}
            for lab, (n, alone) in sorted(count.items(), key=lambda x: (-x[1][0], -x[1][1], x[0]))]


def _rank_real(gallery: Gallery, q: int):
    """Classement d'un clip réel de la galerie, sans son signeur (plafond humain)."""
    return gallery.rank(gallery.seqs[q].astype(np.float64), exclude_signer=str(gallery.signers[q]))


# ------------------------------------------------------------------- sorties
def pct(x: float | None) -> str:
    return "—" if x is None else f"{x:.0%}"


def report(rec: dict) -> None:
    m = rec["metrics"]
    c = m["couverture"]
    print(f"\n### Avatar {rec['avatar']} — dictionnaire {rec['dictionnaire']} — banc {rec['banc']} "
          f"({rec['phrases']} phrases, {c['gloses']} gloses)")
    print(f"  Score        : {pct(m['score'])} de la reconnaissabilité du vrai signeur copié "
          f"(rang centile en contexte {pct(m['contexte']['centile'])} contre {pct(m['contexte']['centile_clips_sources'])} "
          f"pour les clips sources ; hasard 50 %) ; signes isolés : {pct(m['score_isoles'])}")
    ph = m["phrases"]
    print(f"  Phrases      : {pct(ph['jouables'])} jouables en entier (aucun mot sans signe) ; reconnues "
          + ", ".join(f"{pct(ph['reconnues'][s])} au seuil {s} (vrais signeurs {pct(ph['reconnues_clips_sources'][s])})"
                      for s in ph["reconnues"]) + f" — sur {ph['n']} phrases")
    if ph.get("reconnues_avec_verifies"):
        print(f"  Avec vérifiés: {pct(ph['reconnues_avec_verifies']['80%'])} au seuil 80 %, "
              f"{pct(ph['reconnues_avec_verifies']['90%'])} au seuil 90 %, en comptant comme reconnus les signes "
              f"jugés corrects à l'œil ({VERIFIED.name})")
    if "limite_mesure" in ph:
        print(f"  Limite       : {pct(ph['limite_mesure']['80%'])} des phrases échouent au seuil 80 % seulement à cause "
              f"de signes hors de portée du reconnaisseur (aucun clip réel du signe ne passe non plus)")
    print(f"  Couverture   : {pct(c['signe'])} signes réels, {pct(c.get('epele'))} épelés, {pct(c['pointage'])} pointages, "
          f"{pct(c['absent'])} sans signe")
    if m.get("lettres", {}).get("n"):
        x = m["lettres"]
        ref_l = ("vos vidéos de lettres, main seule" if rec.get("galerie", {}).get("lettres") == "videos"
                 else "exemples LSFB")
        print(f"  Lettres      : comparées à {ref_l} ; centile {pct(x['centile'])} (clips sources {pct(x['centile_clips_sources'])}), "
              f"top-10 {pct(x['top10'])} ({x['n']} lettres épelées en phrase)")
    for name, k, what in (("Avatar isolé ", "isoles", "signes"), ("Avatar phrase", "contexte", "signes en phrase"),
                          ("Clip source  ", "clip_source", "clips réels copiés"),
                          ("Autre signeur", "autre_signeur", "clips réels d'un autre signeur")):
        x = m[k]
        print(f"  {name}: centile {pct(x['centile'])}, top-1 {pct(x['top1'])}, top-5 {pct(x['top5'])}, "
              f"top-10 {pct(x['top10'])} ({x['n']} {what})")
    mv = m["mouvement"]
    print(f"  Mouvement    : {pct(mv['images_sans_defaut'])} d'images sans défaut (vitesse anormale "
          f"{mv['vitesse_anormale']:.1%}, à-coups {mv['a_coups']:.1%}, main dans le torse {mv['main_dans_torse']:.1%} ; "
          f"humains : {mv['reference_humaine_main_dans_torse']:.1%})")
    print(f"  Fidélité     : écart médian au clip source bras {m['fidelite']['bras_deg']:.0f}°, "
          f"doigts {m['fidelite']['doigts_deg']:.0f}°")
    ok = [s for s in rec["signes"] if s["centile_clip_source"] is not None and s["centile_avatar"] is not None]
    lost = sorted((s for s in ok if s["centile_clip_source"] - s["centile_avatar"] > 0.1),
                  key=lambda s: s["centile_avatar"] - s["centile_clip_source"])[:12]
    print("  Perdus par l'avatar (clip source mieux reconnu) : "
          + (", ".join(f"{s['etiquette']} (rang {s['rang_avatar']} contre {s['rang_clip_source']}, doigts "
                       f"{s['ecart_doigts_deg']:.0f}°)" for s in lost) or "aucun"))
    bad_clip = sorted((s for s in ok if s["rang_clip_source"] > 50 and (s["rang_autre_signeur"] or 999) <= 10
                       and not s.get("clip_relu")),                  # clip choisi ou gardé à l'œil : on s'y tient
                      key=lambda s: -s["rang_clip_source"])[:12]
    blocking = m["phrases"].get("signes_bloquants", {}).get("80%", [])
    fmt = lambda bs: ", ".join(f"{b['etiquette']} {b['phrases']} (dont {b['seul']} à lui seul)" for b in bs) or "aucun"
    for name, keep in (("Signes", lambda l: not is_letter(l)), ("Lettres", is_letter)):
        worst = [b for b in blocking if keep(b["etiquette"]) and not b.get("limite") and not b.get("verifie")][:12]
        print(f"  {name} qui font échouer des phrases jouables (seuil 80 %) : {fmt(worst)}")
    beyond = [b for b in blocking if b.get("limite") and not b.get("verifie")]
    if beyond:
        print(f"  Hors de portée du reconnaisseur (mesure peu fiable, à vérifier à l'œil) : {fmt(beyond)}")
    checked = [b for b in blocking if b.get("verifie")]
    if checked:
        print(f"  Vérifiés à l'œil (corrects, non comptés comme bloquants) : {', '.join(b['etiquette'] for b in checked)}")
    print("  Clips sources à changer (un autre signeur est bien reconnu ; hors clips relus à l'œil) : "
          + (", ".join(f"{s['etiquette']} (rang {s['rang_clip_source']} contre {s['rang_autre_signeur']})"
                       for s in bad_clip) or "aucun"))


def phrase_cells(m: dict) -> str:
    """Cellules « Phrases jouables » et « Phrases reconnues » (seuil 80 %) de RESULTS.md."""
    ph = m["phrases"]
    return (f"{pct(ph['jouables'])} | {pct(ph['reconnues']['80%'])} / "
            f"{pct(ph['reconnues_clips_sources']['80%'])}")


def save(rec: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{rec['timestamp'].replace(':', '-')}__{Path(rec['avatar']).stem}.json"
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    m = rec["metrics"]
    if not TABLE.exists():
        TABLE.write_text("# Résultats de l'évaluation de l'avatar\n\n"
                         "Généré par `evaluate_avatar.py --save`. Score = (centile en contexte − 50 %) ÷ "
                         "(centile des clips sources − 50 %). Ne comparer que des lignes de même banc, même galerie "
                         "et même code d'évaluation.\n\n"
                         "| Date | Heure | Description | Avatar | Dictionnaire | Banc | Score | Phrases jouables | "
                         "Phrases reconnues (avatar / vrais signeurs) | Centile contexte | "
                         "Centile isolés | Centile clips sources | Couverture signes | Mouvement sans défaut | "
                         "Fidélité bras / doigts | Record |\n"
                         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n", encoding="utf-8")
    d, t = rec["timestamp"].split("T")
    half = rec.get("galerie", {}).get("moitie")
    label = ((rec["label"] or "") + (" — galerie : moitié eval" if half == "eval" else "")
             + (" ; lettres : vidéos filmées" if rec.get("galerie", {}).get("lettres") == "videos" else ""))
    row = (f"| {d} | {t} | {label} | {rec['avatar']} | {rec['dictionnaire']} | {rec['banc']} | "
           f"**{pct(m['score'])}** | {phrase_cells(m)} | {pct(m['contexte']['centile'])} | {pct(m['isoles']['centile'])} | "
           f"{pct(m['clip_source']['centile'])} | {pct(m['couverture']['signe'])} | "
           f"{pct(m['mouvement']['images_sans_defaut'])} | {m['fidelite']['bras_deg']:.0f}° / "
           f"{m['fidelite']['doigts_deg']:.0f}° | `results/{path.name}` |\n")
    with open(TABLE, "a", encoding="utf-8") as f:
        f.write(row)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--avatar", help="fichier .vrm (VRoid Studio) ; défaut : mannequin")
    ap.add_argument("--phrases", nargs="+", help="banc d'essai : jsonl avec un champ « glosses » (gloses LSFB de référence)")
    ap.add_argument("--blender", default=os.environ.get("BLENDER", DEFAULT_BLENDER))
    ap.add_argument("--save", action="store_true", help="archive le run (results/ + RESULTS.md)")
    ap.add_argument("--label", default="", help="description courte du run")
    ap.add_argument("--galerie", choices=("eval", "toutes"), default="eval",
                    help="signeurs de la galerie : moitié « eval » (défaut ; l'autre moitié choisit les "
                         "clips du dictionnaire) ou tous (runs antérieurs, à ne pas comparer)")
    ap.add_argument("--lettres", choices=("videos", "lsfb"), default="videos",
                    help="référence des lettres : les vidéos filmées du projet (défaut, main seule) "
                         "ou les exemples LSFB (runs antérieurs)")
    args = ap.parse_args()
    if not Path(args.blender).exists():
        sys.exit(f"Blender introuvable : {args.blender} (option --blender ou variable BLENDER)")
    rec = evaluate(args)
    report(rec)
    if args.save:
        print(f"\n💾  Record : {save(rec).relative_to(PROJECT)}  —  tableau : {TABLE.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
