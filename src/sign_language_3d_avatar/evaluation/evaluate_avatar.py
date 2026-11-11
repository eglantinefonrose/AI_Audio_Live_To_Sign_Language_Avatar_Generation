"""Évaluation de l'avatar seul : signe-t-il bien des phrases déjà en gloses LSFB ?

La traduction n'intervient pas : les phrases du banc d'essai sont des gloses de
référence (par défaut celles de `src/eval/dataset.jsonl` et `src/eval/dev.jsonl`).
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

Le reconnaisseur (plus proches voisins, ~9 exemples par signe, 330 signes) est faible
en top-1 même sur de vrais signeurs (~10 %) : le score s'appuie donc sur le **rang
centile** du bon signe (part des autres signes classés derrière lui : 100 % = premier,
50 % = hasard), continu et mesuré de la même façon pour l'avatar et pour les humains.

**Chiffre de tête** : (centile en contexte − 50 %) ÷ (centile des clips sources − 50 %),
soit la part de la reconnaissabilité du vrai signeur que l'avatar conserve au-dessus du
hasard (100 % = aussi reconnaissable que le signeur qu'il copie). Les pointages (générés) et les mots sans signe
ne sont pas reconnaissables par la galerie : ils comptent dans la couverture seulement.
La fidélité au clip source (écart angulaire bras / doigts) explique les signes ratés.

    uv run python src/sign_language_3d_avatar/evaluation/evaluate_avatar.py [--avatar mon_avatar.vrm] \\
        [--phrases fichier.jsonl ...] [--save] [--label "…"]

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
from clean_signs import LSFB, clean  # noqa: E402
from features import Gallery, angle_error, percentile, rank_of, resample, vector  # noqa: E402
from pose3d import FPS, unstack  # noqa: E402
from sequence import DICT, INDEX, load_index, load_sign, resolve  # noqa: E402

CACHE = DICT / "evaluation"
RESULTS = HERE / "results"
TABLE = HERE / "RESULTS.md"
EXPORT = HERE / "export_avatar_blender.py"
DEFAULT_BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"
DEFAULT_BENCH = [PROJECT / "src" / "eval" / "dataset.jsonl", PROJECT / "src" / "eval" / "dev.jsonl"]
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


def build_gallery(index: dict) -> tuple[Gallery, dict]:
    """Exemples réels LSFB-ISOL des signes du dictionnaire, nettoyés comme ceux de
    l'avatar (droitier, sauts, lissage, découpe), et référence humaine du mouvement
    (seuils de vitesse / accélération = 99,5e centile des vrais signeurs) ; mis en cache."""
    labels = set(index["signes"])
    with open(LSFB / "instances.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["sign"] in labels
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
            j, _ = clean(r["id"])
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
              (BASE / "avatar_animation" / "sequence.py").read_bytes())
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
    gallery, ref = build_gallery(index)
    items, meta = export_avatar(index, phrases, args)
    t0 = time.time()

    # 1. Couverture des gloses du banc.
    kinds = {"signe": 0, "pointage": 0, "absent": 0}
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
        rank = rank_of(gallery.rank(vector(av), exclude_signer=info["signeur"]), lab)
        own = [i for i in real_by_label.get(lab, []) if gallery.instances[i] == info["instance"]]
        s_rank = rank_of(_rank_real(gallery, own[0]), lab) if own else None
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
    ctx, ctx_labels, quality, per_phrase = [], [], {"vitesse": 0, "acceleration": 0, "torse": 0, "images": 0}, []
    for k, ph in enumerate(meta["phrases"]):
        j = items[f"phrase::{k}"]
        found = []
        for start, end, gloss, kind, _ in ph["chronologie"]:
            if kind != "signe":
                continue
            lab = resolve(gloss, index)["signe"]
            seg = {n: v[start:end] for n, v in j.items()}
            r = rank_of(gallery.rank(vector(seg), exclude_signer=index["signes"][lab]["signeur"]), lab)
            ctx.append(r)
            ctx_labels.append(lab)
            found.append({"glose": gloss, "etiquette": lab, "rang": r and r[0], "centile": percentile(r)})
        m = motion_stats(j)
        bad = {"vitesse": m["vitesse"] > ref["vitesse"], "acceleration": m["acceleration"] > ref["acceleration"]}
        quality["vitesse"] += int(bad["vitesse"].sum())
        quality["acceleration"] += int(bad["acceleration"].sum())
        quality["torse"] += int(m["torse"].sum())
        quality["images"] += len(m["torse"])
        per_phrase.append({"gloses": ph["gloses"], "signes": found,
                           "absents": [g for g in ph["gloses"] if resolve(g, index)["type"] == "absent"],
                           "images_defaut": int(bad["vitesse"].sum() + bad["acceleration"].sum() + m["torse"].sum())})

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
        "couverture": {"signe": kinds["signe"] / max(1, n_gloss), "pointage": kinds["pointage"] / max(1, n_gloss),
                       "absent": kinds["absent"] / max(1, n_gloss), "gloses": n_gloss},
        "isoles": {"centile": c_iso, "top1": top(iso_av, 1), "top5": top(iso_av, 5),
                   "top10": top(iso_av, 10), "n": len(iso_av)},
        "contexte": {"centile": c_ctx, "top1": top(ctx, 1), "top5": top(ctx, 5), "top10": top(ctx, 10),
                     "n": len(ctx), "centile_clips_sources": c_src_ctx},
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
        "phrases": phrase_scores(per_phrase, src),
    }
    print(f"(mesures en {time.time() - t0:.0f} s)")
    return {"timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), "label": args.label,
            "avatar": meta["avatar"], "dictionnaire": sha(INDEX.read_bytes()), "banc": bench_hash,
            "banc_fichiers": [str(p.relative_to(PROJECT)) for p in bench_paths], "phrases": len(phrases),
            "galerie": {"exemples": len(gallery.labels), "signes": len(set(gallery.labels)),
                        "signeurs": len(set(gallery.signers))},
            "code": sha(*(p.read_bytes() for p in (HERE / "features.py", Path(__file__), EXPORT))),
            "git_commit": git("rev-parse", "--short", "HEAD") or None, "git_dirty": bool(git("status", "--porcelain")),
            "metrics": metrics, "signes": signs, "par_phrase": per_phrase}


PHRASE_THRESHOLDS = (0.8, 0.9)            # rang centile minimal de chaque signe (top 20 % / top 10 %)


def phrase_scores(per_phrase: list[dict], src: dict[str, float | None]) -> dict:
    """La phrase entière est-elle bien signée ? « Jouable » : chaque glose a un vrai signe
    ou un pointage. « Reconnue au seuil s » : jouable et chaque vrai signe, en contexte,
    a un rang centile ≥ s. Référence : la même phrase où chaque signe est remplacé par son
    clip source joué par le vrai signeur. Les signes que la galerie ne classe pas
    (centile inconnu) ne comptent ni pour l'avatar ni pour la référence."""
    n = max(1, len(per_phrase))
    playable = [not p["absents"] for p in per_phrase]
    out = {"n": len(per_phrase), "jouables": sum(playable) / n, "reconnues": {}, "reconnues_clips_sources": {}}
    for th in PHRASE_THRESHOLDS:
        av = hu = 0
        for p, ok in zip(per_phrase, playable):
            measured = [s for s in p["signes"] if s["centile"] is not None]
            av += ok and all(s["centile"] >= th for s in measured)
            hu += ok and all((src.get(s["etiquette"]) or 0) >= th for s in measured)
        out["reconnues"][f"{th:.0%}"] = av / n
        out["reconnues_clips_sources"][f"{th:.0%}"] = hu / n
    return out


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
    print(f"  Couverture   : {pct(c['signe'])} signes réels, {pct(c['pointage'])} pointages, {pct(c['absent'])} sans signe")
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
    bad_clip = sorted((s for s in ok if s["rang_clip_source"] > 50 and (s["rang_autre_signeur"] or 999) <= 10),
                      key=lambda s: -s["rang_clip_source"])[:12]
    print("  Clips sources à changer (un autre signeur est bien reconnu) : "
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
    row = (f"| {d} | {t} | {rec['label'] or ''} | {rec['avatar']} | {rec['dictionnaire']} | {rec['banc']} | "
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
    args = ap.parse_args()
    if not Path(args.blender).exists():
        sys.exit(f"Blender introuvable : {args.blender} (option --blender ou variable BLENDER)")
    rec = evaluate(args)
    report(rec)
    if args.save:
        print(f"\n💾  Record : {save(rec).relative_to(PROJECT)}  —  tableau : {TABLE.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
