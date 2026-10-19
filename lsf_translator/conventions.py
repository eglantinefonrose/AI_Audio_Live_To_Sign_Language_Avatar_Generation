"""Vérificateur des conventions de glose (CONVENTIONS.md).

Contrôle les règles **mécaniquement vérifiables** d'une paire (phrase, gloses) :
forme, mots-outils, pronoms, possessifs, négation, questions, lexique imposé,
temps explicites, verbes directionnels. L'ordre sujet/objet et le choix des
signes restent à relire : ce n'est pas un juge LSF, c'est un garde-fou de
cohérence entre prompt, gold d'éval et datasets.

Usage :
    uv run python -m lsf_translator.conventions                 # gold + few-shot
    uv run python -m lsf_translator.conventions <fichier.jsonl>  # {text, glosses}
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

GLOSS_RE = re.compile(r"[A-ZÀ-ÖØ-ÞŒÆ0-9'\-]+")

# Mots-outils jamais glosés (articles, prépositions, conjonctions, copule…).
FUNCTION_WORDS = {
    "LE", "LA", "LES", "L'", "UN", "UNE", "DES", "DU", "DE", "D'", "À", "AU", "AUX",
    "DANS", "POUR", "PAR", "SUR", "SOUS", "AVEC", "SANS", "CHEZ", "EN", "ET", "OU",
    "MAIS", "QUE", "QU'", "CE", "CET", "CETTE", "CES", "SE", "NE", "N'", "Y",
    "ÊTRE", "EST", "SONT", "AVOIR", "A", "ONT", "VA", "DONC", "ALORS", "DU-COUP",
}
PRONOUN_FORMS = {"JE", "J'", "TU", "IL", "ELLE", "ILS", "ELLES", "ON", "ME", "M'",
                 "TE", "T'", "LEUR", "I", "YOU", "HE", "SHE", "WE", "THEY"}
POSSESSIVES = {"MON", "MA", "MES", "TON", "TA", "TES", "SON", "SA", "SES", "NOTRE",
               "NOS", "VOTRE", "VOS", "LEURS", "MY", "YOUR", "HIS", "HER", "OUR"}
# Noms composés dont la fin ressemble à un pronom (pas des directionnels).
COMPOUNDS = {"rendez-vous": "RENDEZ-VOUS", "chez-moi": "CHEZ-MOI", "chez-soi": "CHEZ-SOI"}
POINTS = {"MOI", "TOI", "LUI", "NOUS", "VOUS", "EUX"}
WH = {"QUOI", "OÙ", "QUI", "QUAND", "POURQUOI", "COMMENT", "COMBIEN"}
NEGATIONS = {"NON", "JAMAIS", "RIEN", "PERSONNE"}
TIME_MARKERS = {"AVANT", "BIENTÔT"}
# Temps explicites : mot source -> glose attendue.
EXPLICIT_TIME = {
    "hier": "HIER", "demain": "DEMAIN", "aujourd'hui": "AUJOURD'HUI",
    "maintenant": "MAINTENANT", "matin": "MATIN", "soir": "SOIR",
    "week-end": "WEEK-END", "midi": "MIDI", "nuit": "NUIT",
    "yesterday": "HIER", "tomorrow": "DEMAIN", "today": "AUJOURD'HUI",
}
CONNECTOR_END = ("pour finir", "pour conclure", "en conclusion", "pour terminer")
TECH_TERMS = {
    "dataset": "DATASET", "fine tuning": "FINE-TUNING", "fine-tuning": "FINE-TUNING",
    "random forest": "RANDOM-FOREST", "machine learning": "MACHINE-LEARNING",
    "deep learning": "DEEP-LEARNING", "camembert": "CAMEMBERT",
}
# Possessif source -> pointage attendu après le nom.
POSS_SRC = {"mon": "MOI", "ma": "MOI", "mes": "MOI", "ton": "TOI", "ta": "TOI",
            "tes": "TOI", "son": "LUI", "sa": "LUI", "ses": "LUI", "notre": "NOUS",
            "nos": "NOUS", "votre": "VOUS", "vos": "VOUS", "leurs": "EUX",
            "my": "MOI", "your": "TOI", "his": "LUI", "her": "LUI", "our": "NOUS",
            "their": "EUX"}
# Pronom sujet source -> pointage attendu (« il » impersonnel exclu).
SUBJ_SRC = {"je": "MOI", "j'": "MOI", "tu": "TOI", "nous": "NOUS", "vous": "VOUS",
            "il": "LUI", "elle": "LUI", "ils": "EUX", "elles": "EUX", "on": "NOUS",
            "i": "MOI", "we": "NOUS", "he": "LUI", "she": "LUI", "they": "EUX"}
IMPERSONAL = re.compile(
    r"\bil (pleut|neige|fait|fera|faisait|y a|y avait|y aura|faut|faudra|fallait|"
    r"est (temps|tard|tôt|\w+ de)|a plu|a neigé|a fait|va (pleuvoir|neiger|faire))\b", re.I)
# Passé composé (avoir + participe, ou être + participe de mouvement).
PAST_RE = re.compile(
    r"\b(j'ai|ai|as|a|avons|avez|ont)\s+((pas|plus|jamais|déjà|bien|beaucoup|encore|rien|tout)\s+)?"
    r"(?!(du|un|une|des|de|le|la|les|au|aux|deux|trois|six|dix|cent|mille|"
    r"beaucoup|peu|assez|trop|faim|soif|froid|chaud|peur|besoin|envie|raison|tort)\b)\w+(é|ée|és|ées|i|is|it|u|ue|us|ert|ort|eint|ait)\b|"
    r"\b(suis|es|est|sommes|êtes|sont)\s+(allée?s?|venue?s?|partie?s?|arrivée?s?|sortie?s?|"
    r"entrée?s?|restée?s?|tombée?s?|rentrée?s?|revenue?s?|montée?s?|descendue?s?|passée?s?)\b",
    re.I)
FUTURE_RE = re.compile(
    # Futur simple : -er-/-ir- + terminaison, et radicaux irréguliers (pourr-, verr-,
    # aur-, ser-, fer-…). Pas de « -rons » générique (« comparons », « préparons »).
    r"\b\w{2,}(erai|eras|era|erons|erez|eront|irai|iras|ira|irons|irez|iront|"
    r"rrai|rras|rra|rrons|rrez|rront|drai|dras|dra|drons|drez|dront)\b|"
    r"\b(aurai|auras|aura|aurons|aurez|auront)\b|"
    r"\b(vais|vas|va|allons|allez|vont)\s+((vous|te|me|le|la|les|lui|leur|nous|t'|m')\s*)?"
    r"\w+(er|ir|re|oir)\b", re.I)
NOT_FUTURE = {"entre", "notre", "votre", "autre", "contre", "être", "lire", "écrire",
              "dire", "faire", "boire", "prendre", "rire", "vivre", "suivre", "mettre"}
NEG_RE = re.compile(r"\b(ne|n')\b.*\b(pas|plus|jamais|rien|personne)\b|"
                    r"\b(don't|doesn't|didn't|not|never|can't|won't)\b", re.I)


def _strip(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def check(text: str, glosses: list[str]) -> list[str]:
    """Liste des écarts aux conventions (vide = conforme)."""
    errs: list[str] = []
    low = text.lower()
    g = glosses
    if not g:
        return ["aucune glose"]

    # 1. Forme
    for x in g:
        if not GLOSS_RE.fullmatch(x):
            errs.append(f"forme invalide « {x} » (MAJUSCULES, traits d'union)")
    # 3. Mots supprimés
    errs += [f"mot-outil glosé « {x} »" for x in g if x in FUNCTION_WORDS]
    # 4. Pronoms / possessifs
    errs += [f"pronom non pointé « {x} » (→ MOI/TOI/LUI/NOUS/VOUS/EUX)"
             for x in g if x in PRONOUN_FORMS]
    errs += [f"possessif « {x} » (→ pointage après le nom)" for x in g if x in POSSESSIVES]

    # 9. Questions
    has_wh = [x for x in g if x in WH]
    # Le marqueur est le QUESTION final d'une phrase interrogative ; ailleurs,
    # QUESTION est le nom « question » (« je n'ai pas compris la question »).
    is_q = text.strip().endswith("?")
    marker = is_q and g[-1] == "QUESTION"
    if marker and has_wh:
        errs.append("QUESTION et mot interrogatif ensemble")
    if has_wh and g[-1] not in WH:
        errs.append(f"interrogatif « {has_wh[0]} » pas en dernier")
    if is_q and g[-1] not in WH | {"QUESTION"}:
        errs.append("question sans QUESTION ni interrogatif final")
    if not is_q and has_wh and g[-1] in WH:
        errs.append("interrogatif final sur une phrase non interrogative")

    # 8. Négation : après le verbe, suivie seulement d'un interrogatif
    negs = [i for i, x in enumerate(g) if x in NEGATIONS]
    if NEG_RE.search(text) and not negs:
        errs.append("négation source non glosée (NON/JAMAIS/RIEN/PERSONNE)")
    for i in negs:
        if i == 0:
            errs.append(f"{g[i]} en tête (doit suivre le verbe)")
        if any(x not in WH | {"QUESTION"} for x in g[i + 1:]) or \
                ("QUESTION" in g[i + 1:] and not marker):
            errs.append(f"{g[i]} suivi d'autre chose qu'un interrogatif")

    # 11. Lexique imposé
    if re.search(r"\ben (direct|live)\b|\blive\b", low) and "DIRECT" not in g:
        errs.append("« en direct / live » sans DIRECT")
    if "LIVE" in g:
        errs.append("LIVE au lieu de DIRECT")
    if low.startswith(CONNECTOR_END):
        if "ENFIN" not in g:
            errs.append("connecteur de conclusion sans ENFIN")
        elif g[0] != "ENFIN":
            errs.append("ENFIN pas en tout premier")
    for src, gl in TECH_TERMS.items():
        if re.search(rf"\b{re.escape(src)}\b", low) and gl not in g:
            errs.append(f"terme « {src} » non repris tel quel ({gl})")

    # 7. Temps : explicite gardé, pas de AVANT/BIENTÔT en plus, temps en tête
    explicit = {gl for w, gl in EXPLICIT_TIME.items() if re.search(rf"\b{re.escape(w)}\b", low)}
    for gl in explicit:
        if gl not in g:
            errs.append(f"temps explicite non gardé ({gl})")
    if explicit and any(m in g for m in TIME_MARKERS):
        errs.append("AVANT/BIENTÔT en plus d'un temps explicite")
    if "AVANT" in g and "BIENTÔT" in g:
        errs.append("AVANT et BIENTÔT ensemble")
    times = {*explicit, *TIME_MARKERS} & set(g)
    if times:
        start = 1 if g[0] == "ENFIN" else 0
        for i, x in enumerate(g):
            if x in times and any(y not in times and y not in {"ENFIN"} for y in g[start:i]):
                errs.append(f"temps « {x} » pas en tête")
                break

    # 7 bis. Passé / futur sans temps explicite -> AVANT / BIENTÔT en tête
    no_time = not explicit and not re.search(r"tout à l'heure|\bdans \w+ (jours?|semaines?|mois|ans?)\b|"
                                             r"\b(semaine|année|mois) (prochaine?|dernière?)\b|"
                                             r"\b(lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche)\b|"
                                             r"\bbientôt\b|\bdéjà\b|\b\w+ heures?\b", low)
    fut = [m.group(0) for m in FUTURE_RE.finditer(low)
           if m.group(0).split()[-1] not in NOT_FUTURE and not m.group(0).startswith(("va ", "vas "))
           or m.group(0).startswith(("va ", "vas ")) and m.group(0).split()[-1] not in NOT_FUTURE]
    if no_time and PAST_RE.search(low) and "AVANT" not in g:
        errs.append("passé sans temps explicite : AVANT attendu en tête")
    if no_time and fut and "BIENTÔT" not in g:
        errs.append(f"futur sans temps explicite (« {fut[0]} ») : BIENTÔT attendu en tête")
    if re.search(r"\b(vais|vas|va|allons|allez|vont)\s+((vous|te|me|le|la|les|lui|leur|nous|t'|m')\s*)?"
                 r"\w+(er|ir|re|oir)\b", low) and "ALLER" in g:
        errs.append("« aller » + infinitif glosé ALLER (futur proche → BIENTÔT)")

    # 4 bis. Pronoms sujets et possessifs de la source -> pointages présents
    # Chaque pronom sujet et chaque possessif demande son pointage (comptés :
    # « je … mes clés » -> deux MOI).
    low_tok = low
    for c in COMPOUNDS:
        low_tok = low_tok.replace(c, " ")
    toks = re.findall(r"[a-zà-ÿ]+'|[a-zà-ÿ]+", low_tok)
    need: dict[str, list[str]] = {}
    for t in toks:
        if t in POSS_SRC:
            need.setdefault(POSS_SRC[t], []).append(t)
        elif t in SUBJ_SRC and not (t == "il" and IMPERSONAL.search(low)):
            need.setdefault(SUBJ_SRC[t], []).append(t)
    for p, srcs in need.items():
        # « nous/vous » objets ou toniques (« avec moi ») ne sont pas des sujets :
        # on n'exige qu'un pointage par sujet + un par possessif.
        if p in {"NOUS", "VOUS"}:
            srcs = srcs[:1] + [t for t in srcs[1:] if t in POSS_SRC]
        # Un pronom complément (« je vous remercie ») est porté par le directionnel.
        found = g.count(p) + sum(x.endswith(f"-À-{p}") for x in g)
        if found < len(srcs):
            errs.append(f"{len(srcs)} pointage(s) {p} attendu(s) ({', '.join(srcs)}), "
                        f"{found} trouvé(s)")

    # 6. Verbes directionnels : VERBE-À-<PRONOM>
    for x in g:
        m = re.fullmatch(r"(.+)-(MOI|TOI|LUI|NOUS|VOUS|EUX)", x)
        if m and x not in COMPOUNDS.values() and not m.group(1).endswith("-À") and "-" not in m.group(1):
            errs.append(f"directionnel « {x} » sans -À- (→ {m.group(1)}-À-{m.group(2)})")

    # 1. Mot source repris : accents conservés
    src_up = {t.upper() for t in re.findall(r"[\wÀ-ÿ'\-]+", text)}
    for x in g:
        for t in src_up:
            if x != t and _strip(x) == _strip(t):
                errs.append(f"accents modifiés « {x} » (source « {t} »)")
    return errs


def check_rows(rows: list[tuple[str, list[str]]]) -> list[tuple[str, list[str], list[str]]]:
    return [(t, g, e) for t, g in rows if (e := check(t, g))]


def _report(name: str, rows: list[tuple[str, list[str]]]) -> int:
    bad = check_rows(rows)
    print(f"── {name} : {len(rows) - len(bad)}/{len(rows)} conformes")
    for t, g, e in bad:
        print(f"  ✗ {t}\n      {' '.join(g)}\n      → " + " ; ".join(e))
    return len(bad)


def _load_jsonl(path: Path) -> list[tuple[str, list[str]]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if "messages" in r:   # format chat (train/valid)
                text = re.search(r'"(.*)"\s*$', r["messages"][1]["content"]).group(1)
                rows.append((text, json.loads(r["messages"][2]["content"])["glosses"]))
            else:
                rows.append((r["text"], r["glosses"]))
    return rows


if __name__ == "__main__":
    project = Path(__file__).resolve().parent.parent
    if len(sys.argv) > 1:
        n = sum(_report(p, _load_jsonl(Path(p))) for p in sys.argv[1:])
    else:
        from lsf_translator.prompts import FEW_SHOT_EXAMPLES
        n = _report("eval/dataset.jsonl", _load_jsonl(project / "eval" / "dataset.jsonl"))
        n += _report("few-shot (prompts.py)", list(FEW_SHOT_EXAMPLES))
    sys.exit(1 if n else 0)
