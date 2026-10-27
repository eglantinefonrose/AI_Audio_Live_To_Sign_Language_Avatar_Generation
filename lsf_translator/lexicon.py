"""Lexique des gloses connues + outils lexicaux du post-traitement.

Le lexique (`lexicon.txt`) = toutes les gloses du split **train** d'un dataset
rangé + celles des exemples few-shot. Jamais le split valid (jeu de dev) ni le
jeu d'éval : sinon le post-traitement verrait les réponses.

Régénération :
    uv run python -m lsf_translator.lexicon finetune/data/2184f8f3751e
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

import simplemma

LEXICON_PATH = Path(__file__).with_name("lexicon.txt")

# Classe fermée : gloses légitimes sans mot source correspondant.
POINTS = ("MOI", "TOI", "LUI", "NOUS", "VOUS", "EUX")
WH = ("QUOI", "OÙ", "QUI", "QUAND", "POURQUOI", "COMMENT", "COMBIEN")
NEGATIONS = ("NON", "JAMAIS", "RIEN", "PERSONNE")
CLOSED_CLASS = frozenset({*POINTS, *WH, *NEGATIONS, "QUESTION", "AVANT", "BIENTÔT",
                          "ENFIN", "DIRECT", "AIMER", "BEAUCOUP", "PARLER"})
# Mots source ramenés à un pointage (pronoms sujets, toniques, possessifs, clitiques).
POINTING_OF = {
    "je": "MOI", "j": "MOI", "moi": "MOI", "me": "MOI", "m": "MOI", "mon": "MOI",
    "ma": "MOI", "mes": "MOI", "tu": "TOI", "toi": "TOI", "te": "TOI", "t": "TOI",
    "ton": "TOI", "ta": "TOI", "tes": "TOI", "il": "LUI", "elle": "LUI", "lui": "LUI",
    "son": "LUI", "sa": "LUI", "ses": "LUI", "nous": "NOUS", "on": "NOUS",
    "notre": "NOUS", "nos": "NOUS", "vous": "VOUS", "votre": "VOUS", "vos": "VOUS",
    "ils": "EUX", "elles": "EUX", "eux": "EUX", "leur": "EUX", "leurs": "EUX",
}

INFINITIVE_RE = re.compile(r"(ER|IR|RE|OIR)$")
TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿŒœÆæ0-9]+(?:'[a-zà-ÿ]+)?")


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def tokens(text: str) -> list[str]:
    """Mots de la phrase, en minuscules ; « j'ai » -> ["j", "ai"], mais
    « aujourd'hui » reste entier."""
    out: list[str] = []
    for t in TOKEN_RE.findall(text.lower().replace("’", "'")):
        if t == "aujourd'hui":
            out.append(t)
        elif "'" in t:
            head, tail = t.split("'", 1)
            out += [head, tail]
        else:
            out.append(t)
    return out


@lru_cache(maxsize=4096)
def lemma(word: str) -> str:
    """Lemme français (infinitif d'un verbe, singulier d'un nom)."""
    return simplemma.lemmatize(word, lang="fr")


@lru_cache(maxsize=1)
def lexicon() -> frozenset[str]:
    if not LEXICON_PATH.exists():
        return frozenset()
    return frozenset(l for l in LEXICON_PATH.read_text(encoding="utf-8").splitlines()
                     if l and not l.startswith("#"))


@lru_cache(maxsize=1)
def lexicon_parts() -> frozenset[str]:
    """Mots des gloses du lexique (composés et directionnels découpés)."""
    out: set[str] = set()
    for g in lexicon():
        out.add(g)
        out.update(p for p in g.split("-À-")[0].split("-") if p)
    return frozenset(out)


@lru_cache(maxsize=1)
def directional_verbs() -> frozenset[str]:
    """Verbes que les données composent avec leur destinataire (VERBE-À-X)."""
    return frozenset(g.split("-À-")[0] for g in lexicon() if "-À-" in g)


VERB_ENDINGS = ("ent", "ons", "ez", "es", "e", "s", "t", "é", "ée", "és", "ées",
                "i", "ie", "is", "it", "u", "ue", "us")
INFINITIVES = ("er", "ir", "re", "ire", "tir", "oir")


def verb_guesses(word: str) -> set[str]:
    """Infinitifs attestés dans le lexique pour une forme conjuguée ou un
    participe que le lemmatiseur rate (« montres » -> MONTRER, « fini » ->
    FINIR, « écrit » -> ÉCRIRE, « pars » -> PARTIR). Seuls les infinitifs du
    lexique sont proposés : une glose inconnue n'est jamais inventée."""
    out: set[str] = set()
    w = word.lower()
    for end in VERB_ENDINGS:
        if w.endswith(end) and len(w) - len(end) >= 2:
            for inf in INFINITIVES:
                cand = (w[: -len(end)] + inf).upper()
                if cand in lexicon_parts() and INFINITIVE_RE.search(cand):
                    out.add(cand)
    return out


def build(data_dir: Path) -> list[str]:
    """Gloses du split train de `data_dir` + few-shot (triées)."""
    from lsf_translator.prompts import FEW_SHOT_EXAMPLES
    found: set[str] = set()
    for line in (data_dir / "train.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            msgs = json.loads(line)["messages"]
            found.update(json.loads(msgs[-1]["content"])["glosses"])
    for _, glosses in FEW_SHOT_EXAMPLES:
        found.update(glosses)
    return sorted(found)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = Path(sys.argv[1])
    glosses = build(src)
    LEXICON_PATH.write_text(
        f"# Gloses du split train de {src.name} + few-shot ({len(glosses)}). "
        "Généré par lsf_translator/lexicon.py.\n" + "\n".join(glosses) + "\n",
        encoding="utf-8")
    print(f"{len(glosses)} gloses -> {LEXICON_PATH}")
