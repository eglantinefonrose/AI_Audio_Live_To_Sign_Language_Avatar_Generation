"""Construit le vocabulaire LSFB du traducteur : `vocabulaire_lsfb.csv`.

C'est le **contrat** entre la traduction et l'avatar : la liste des gloses (au format du
projet, cf. CONVENTIONS.md : majuscules, accents, nombres en lettres) qui ont un signe
LSFB, avec l'étiquette LSFB correspondante. Le traducteur ne doit produire que ces
gloses (plus les pointages MOI/TOI/…, générés par l'avatar) ; un mot sans signe est
gardé tel quel et sera épelé (LSFB-ISOL contient l'alphabet manuel complet, FS:A…FS:Z).

Source : l'index de LSFB-ISOL (`instances.csv`, téléchargé par
`sign_language_3d_avatar/dataset_generation/lsfb_subset.py map`). Sont retenus :
- les signes lexicaux avec au moins `--min` exemples (10 par défaut : assez pour que
  l'avatar choisisse un bon exemple) ; une étiquette `BASE.PRÉCISION` est une variante
  de BASE : la plus fréquente devient le signe par défaut de la glose ;
- les signes-noms `NS:…` (pays, villes, personnes) avec au moins `--min` exemples ;
- les signes déjà utilisés par le dictionnaire actuel de l'avatar (`selection.csv`),
  même s'ils ont moins d'exemples.

Glose = étiquette LSFB réaccentuée : glose déjà employée par le projet
(`lsfb_mapping.csv`) si elle existe, sinon lemme français de simplemma (en cas
d'ambiguïté, la forme qui n'est pas un participe en -É : ÉLÈVE plutôt qu'ÉLEVÉ) ;
nombres écrits en lettres (2 -> DEUX).

    uv run python -m src.lsf_translator.vocabulaire_lsfb [--min 10]
"""
from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
AVATAR_DATA = SRC / "sign_language_3d_avatar" / "dataset_generation"
INSTANCES = AVATAR_DATA / "data" / "lsfb" / "instances.csv"
MAPPING = AVATAR_DATA / "lsfb_mapping.csv"
SELECTION = AVATAR_DATA / "selection.csv"
OUT = HERE / "vocabulaire_lsfb.csv"
FIELDS = ["glose", "etiquette_lsfb", "type", "exemples", "variantes"]
LABEL_RE = re.compile(r"^[A-Z0-9][A-Z0-9\-]*(\.[A-Z0-9\-]+)?$")
NAME_RE = re.compile(r"^NS:[A-Z][A-Z\-]+$")
# Étiquettes LSFB dont la glose du projet n'est pas le mot de l'étiquette.
RENAME = {"OU-LIEU": "OÙ"}
# Étiquettes écartées : « ou » conjonction (mot-outil), à ne pas confondre avec OÙ.
EXCLUDED = {"OU", "OU.OU"}

UNITS = ["ZÉRO", "UN", "DEUX", "TROIS", "QUATRE", "CINQ", "SIX", "SEPT", "HUIT", "NEUF", "DIX",
         "ONZE", "DOUZE", "TREIZE", "QUATORZE", "QUINZE", "SEIZE"]
TENS = {20: "VINGT", 30: "TRENTE", 40: "QUARANTE", 50: "CINQUANTE", 60: "SOIXANTE"}


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def number_words(n: int) -> str:
    """Nombre en lettres, au format des gloses (DEUX, VINGT-ET-UN, DEUX-MILLE…)."""
    if n <= 16:
        return UNITS[n]
    if n < 20:
        return "DIX-" + UNITS[n - 10]
    if n < 70:
        t, u = (n // 10) * 10, n % 10
        return TENS[t] + ("" if u == 0 else "-ET-UN" if u == 1 else "-" + UNITS[u])
    if n < 80:
        return "SOIXANTE-" + ("ET-ONZE" if n == 71 else number_words(n - 60))
    if n < 100:
        return "QUATRE-VINGT" + ("S" if n == 80 else "-" + number_words(n - 80))
    if n < 1000:
        h, r = divmod(n, 100)
        head = "CENT" if h == 1 else UNITS[h] + "-CENT" + ("S" if r == 0 else "")
        return head + ("" if r == 0 else "-" + number_words(r))
    th, r = divmod(n, 1000)
    head = "MILLE" if th == 1 else number_words(th) + "-MILLE"
    return head + ("" if r == 0 else "-" + number_words(r))


@lru_cache(maxsize=1)
def _accented() -> dict[str, set[str]]:
    """Forme sans accents -> lemmes français accentués (dictionnaire de simplemma)."""
    from simplemma.strategies.dictionaries import DefaultDictionaryFactory
    d = DefaultDictionaryFactory().get_dictionary("fr")
    out: dict[str, set[str]] = defaultdict(set)
    for lem in set(d.values()):
        lem = lem.decode() if isinstance(lem, bytes) else lem
        out[strip_accents(lem).upper()].add(lem.upper())
    return out


def accent(word: str, known: dict[str, str]) -> str:
    """Réaccentue un mot LSFB (ECOLE -> ÉCOLE), partie par partie pour les composés."""
    if word in known:
        return known[word]
    parts = []
    for p in word.split("-"):
        cands = _accented().get(p, set())
        if p in known:
            parts.append(known[p])
        elif not cands or p in cands:
            parts.append(p)
        else:
            # Ambiguïté : on évite les participes en -É (ÉLEVÉ, TÉLÉPHONÉ, MÉMÉ).
            parts.append(min(cands, key=lambda c: (c.endswith("É"), c)))
    return "-".join(parts)


def build(min_examples: int = 10) -> list[dict]:
    with open(INSTANCES, newline="", encoding="utf-8") as f:
        counts = Counter(r["sign"] for r in csv.DictReader(f))
    known: dict[str, str] = {}              # étiquette LSFB (sans accents) -> glose du projet
    if MAPPING.exists():
        with open(MAPPING, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                g = r["glose_projet"]
                if r["statut"] == "exact" and "-À-" not in g:
                    prev = known.get(r["etiquette_lsfb"])
                    # Deux gloses du projet pour une étiquette (ÉLÈVE/ÉLEVÉ) : pas le participe.
                    if prev is None or (prev.endswith("É"), prev) > (g.endswith("É"), g):
                        known[r["etiquette_lsfb"]] = g
    forced = set()
    if SELECTION.exists():
        with open(SELECTION, newline="", encoding="utf-8") as f:
            forced = {r["etiquette_lsfb"] for r in csv.DictReader(f)}

    # Signes lexicaux : variantes BASE.X regroupées sous BASE.
    groups: dict[str, list[str]] = defaultdict(list)
    for label, n in counts.items():
        if label in EXCLUDED or label in RENAME:
            continue
        if LABEL_RE.match(label) and (n >= min_examples or label in forced):
            groups[label.split(".")[0]].append(label)
    rows = []
    for base, labels in groups.items():
        labels.sort(key=lambda l: (-counts[l], l))
        # Signe par défaut : celui du dictionnaire actuel, sinon l'étiquette sans
        # suffixe si elle a assez d'exemples (MARCHER plutôt que MARCHER.PIED), sinon
        # la variante la plus fréquente.
        plain = base if base in labels and counts[base] >= min_examples else None
        default = next((l for l in labels if l in forced), plain or labels[0])
        gloss = number_words(int(base)) if base.isdigit() else accent(base, known)
        rows.append({"glose": gloss, "etiquette_lsfb": default, "type": "nombre" if base.isdigit() else "signe",
                     "exemples": counts[default], "variantes": " ".join(l for l in labels if l != default)})
    for label, gloss in RENAME.items():
        if counts.get(label, 0) >= min_examples:
            rows.append({"glose": gloss, "etiquette_lsfb": label, "type": "signe",
                         "exemples": counts[label], "variantes": ""})
    # Signes-noms (lieux, personnes).
    for label, n in counts.items():
        if NAME_RE.match(label) and n >= min_examples:
            rows.append({"glose": accent(label[3:], known), "etiquette_lsfb": label, "type": "nom-propre",
                         "exemples": n, "variantes": ""})
    # Une glose = un signe : en cas de collision (deux étiquettes réaccentuées pareil),
    # on garde la plus fréquente.
    best: dict[str, dict] = {}
    for r in rows:
        if r["glose"] not in best or r["exemples"] > best[r["glose"]]["exemples"]:
            best[r["glose"]] = r
    return sorted(best.values(), key=lambda r: r["glose"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min", type=int, default=10, help="exemples LSFB minimum par signe (défaut 10)")
    args = ap.parse_args()
    if not INSTANCES.exists():
        raise SystemExit(f"Index LSFB introuvable : {INSTANCES}\nLance d'abord : uv run python "
                         "src/sign_language_3d_avatar/dataset_generation/lsfb_subset.py map")
    rows = build(args.min)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    types = Counter(r["type"] for r in rows)
    print(f"{len(rows)} gloses avec un signe LSFB ({', '.join(f'{k} {v}' for k, v in types.most_common())}) "
          f"→ {OUT.relative_to(SRC.parent)}")


if __name__ == "__main__":
    main()
