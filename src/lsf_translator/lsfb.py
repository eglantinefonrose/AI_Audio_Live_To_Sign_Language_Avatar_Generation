"""Vocabulaire LSFB à l'exécution : gloses jouables par l'avatar, conversion, couverture.

- `vocabulaire_lsfb.csv` (généré par `vocabulaire_lsfb.py`) : gloses qui ont un signe
  LSFB ; avec les pointages MOI/TOI/LUI/NOUS/VOUS/EUX (générés par l'avatar), ce sont
  les gloses **jouables** ;
- `synonymes_lsfb.csv` (édité à la main) : glose sans signe -> glose(s) jouable(s) de
  même sens (PÈRE -> PAPA, TRÈS -> BEAUCOUP…) ;
- tout autre mot (nom propre, terme technique) est gardé tel quel : l'avatar l'épellera.

    from src.lsf_translator import lsfb
    lsfb.playable("ÉCOLE")                     # True
    lsfb.convert(["MOI", "PÈRE", "AIMER"])     # ["MOI", "PAPA", "AIMER"]
    lsfb.convert(["REMERCIER-À-VOUS"])         # ["MERCI-À-VOUS"] (destinataire gardé)
    lsfb.hints("J'adore mon père.")            # (["AIMER", "PAPA", …], [])
    uv run python -m src.lsf_translator.lsfb couverture src/eval/dataset.jsonl
    uv run python -m src.lsf_translator.lsfb convertir entree.jsonl sortie.jsonl
"""
from __future__ import annotations

import csv
import json
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
VOCAB_PATH = HERE / "vocabulaire_lsfb.csv"
SYNONYMS_PATH = HERE / "synonymes_lsfb.csv"
POINTS = frozenset({"MOI", "TOI", "LUI", "NOUS", "VOUS", "EUX"})


def _norm(g: str) -> str:
    s = "".join(c for c in unicodedata.normalize("NFD", g.upper()) if unicodedata.category(c) != "Mn")
    return s.replace("'", "-")


@lru_cache(maxsize=1)
def vocab() -> dict[str, dict]:
    """Glose -> ligne du vocabulaire (étiquette LSFB, type, exemples, variantes)."""
    if not VOCAB_PATH.exists():
        return {}
    with open(VOCAB_PATH, newline="", encoding="utf-8") as f:
        return {r["glose"]: r for r in csv.DictReader(f)}


@lru_cache(maxsize=1)
def _by_norm() -> dict[str, str]:
    return {_norm(g): g for g in vocab()}


@lru_cache(maxsize=1)
def synonyms() -> dict[str, list[str]]:
    """Glose sans signe -> gloses jouables de même sens (seulement si toutes jouables)."""
    if not SYNONYMS_PATH.exists():
        return {}
    out = {}
    with open(SYNONYMS_PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            repl = r["remplacement"].split()
            if r["glose"] not in vocab() and repl and all(p in vocab() for p in repl):
                out[r["glose"]] = repl
    return out


def playable(g: str) -> bool:
    """Glose jouée par un vrai signe (ou un pointage) : pas épelée."""
    return g in POINTS or g.split("-À-")[0] in vocab()


def to_lsfb(g: str) -> list[str]:
    """Glose(s) du vocabulaire LSFB pour une glose : inchangée si jouable, sinon
    graphie du vocabulaire (sans accents -> avec), synonyme, ou telle quelle (épelée)."""
    if playable(g):
        return [g]
    base, sep, target = g.partition("-À-")
    # ECOLE -> ÉCOLE : seulement pour une glose écrite sans accents. Une glose déjà
    # accentuée n'est jamais rapprochée d'un homographe (ÉLEVÉ n'est pas ÉLÈVE).
    if base.isascii() and _norm(base) in _by_norm():
        fixed = _by_norm()[_norm(base)]
        return [f"{fixed}-À-{target}" if sep else fixed]
    if base in synonyms():                              # PÈRE -> PAPA
        repl = list(synonyms()[base])
        if sep and len(repl) == 1:                      # REMERCIER-À-VOUS -> MERCI-À-VOUS
            return [f"{repl[0]}-À-{target}"]
        return repl
    return [g]


def convert(glosses: list[str]) -> list[str]:
    out: list[str] = []
    for g in glosses:
        for x in to_lsfb(g):
            if not (out and out[-1] == x):                  # pas de doublon créé par la conversion
                out.append(x)
    return out


def coverage(glosses: list[str]) -> tuple[int, int]:
    """(gloses jouables, gloses) d'une sortie."""
    return sum(playable(g) for g in glosses), len(glosses)


def hints(text: str) -> tuple[list[str], list[str], list[str]]:
    """(signes LSFB disponibles, mots à garder tels quels, mots sans signe) d'une phrase.

    Sert à guider un modèle entraîné avec ces indications. Le lexique imposé passe
    avant (« en direct/live » -> DIRECT, connecteur de conclusion -> ENFIN) ; termes
    techniques et noms propres sont « à garder » (épelés), jamais à reformuler ; les
    mots de la négation (« pas », « du tout »…) ne sont pas proposés (NON s'en charge).
    Les listes sont triées par ordre alphabétique pour ne pas suggérer l'ordre français."""
    import re
    from src.lsf_translator.conventions import CONNECTOR_END, FUNCTION_WORDS, NEG_RE, TECH_TERMS
    from src.lsf_translator.lexicon import POINTING_OF, lemma, tokens, verb_guesses
    low = text.lower().replace("’", "'")
    found, keep, missing = set(), set(), set()
    # Lexique imposé et termes techniques : retirés du texte avant l'analyse mot à mot.
    if re.search(r"\ben (direct|live)\b|\blive\b", low):
        found.add("DIRECT")
        low = re.sub(r"\ben (direct|live)\b|\blive\b", " ", low)
    if low.startswith(CONNECTOR_END):
        found.add("ENFIN")
        for c in CONNECTOR_END:
            low = low.replace(c, " ")
    low = re.sub(r"^(du coup|donc|alors)\b|\b(pas )?du tout\b", " ", low)
    for src, gl in TECH_TERMS.items():
        if re.search(rf"\b{re.escape(src)}\b", low):
            (found if gl in vocab() else keep).add(gl)
            low = re.sub(rf"\b{re.escape(src)}\b", " ", low)
    negation = bool(NEG_RE.search(text.lower()))
    # Noms propres : mots à majuscule hors début de phrase (« Camembert est… » compris
    # s'il n'est pas un mot courant).
    proper = {w.lower() for k, w in enumerate(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ'-]+", text))
              if w[0].isupper() and (k > 0 or lemma(w.lower()).upper() not in vocab())}
    for t in tokens(low):
        if (t in POINTING_OF or len(t) == 1 or t.upper() in FUNCTION_WORDS or t.isdigit()
                or lemma(t) in ("être", "avoir") or (negation and t in ("pas", "plus"))):
            continue
        hit = None
        for c in (t.upper(), lemma(t).upper(), *sorted(verb_guesses(t))):
            if c in vocab():
                hit = [c]
            elif c.isascii() and _norm(c) in _by_norm():
                hit = [_by_norm()[_norm(c)]]
            elif c in synonyms():
                hit = synonyms()[c]
            if hit:
                break
        if hit:
            found.update(hit)
        elif t in proper:
            keep.add(t.upper())
        else:
            missing.add(t)
    return sorted(found), sorted(keep), sorted(missing)


def hint_text(text: str) -> str:
    """Lignes ajoutées au message utilisateur quand les indications LSFB sont activées."""
    found, keep, missing = hints(text)
    lines = [f"Signes LSFB disponibles (ordre alphabétique) : {', '.join(found) if found else '(aucun)'}."]
    if keep:
        lines.append(f"À garder tels quels (épelés) : {', '.join(keep)}.")
    if missing:
        lines.append(f"Mots sans signe LSFB : {', '.join(missing)} (utilise un signe disponible "
                     "de même sens si possible, sinon garde le mot).")
    return "\n".join(lines)


# ---------------------------------------------------------------------- CLI
def _rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _glosses(row: dict) -> list[str]:
    if "messages" in row:                                   # format chat (train/valid)
        return json.loads(row["messages"][-1]["content"])["glosses"]
    return row["glosses"]


def main() -> None:
    if len(sys.argv) >= 3 and sys.argv[1] == "couverture":
        for p in sys.argv[2:]:
            ok = tot = 0
            unplayable: dict[str, int] = {}
            for r in _rows(Path(p)):
                for g in _glosses(r):
                    tot += 1
                    if playable(g):
                        ok += 1
                    else:
                        unplayable[g] = unplayable.get(g, 0) + 1
            print(f"{p} : {ok}/{tot} gloses jouables ({ok / max(1, tot):.0%}) ; épelées : "
                  + ", ".join(f"{g}×{n}" if n > 1 else g for g, n in sorted(unplayable.items(), key=lambda x: -x[1])))
    elif len(sys.argv) == 4 and sys.argv[1] == "convertir":
        rows = _rows(Path(sys.argv[2]))
        changed = 0
        with open(sys.argv[3], "w", encoding="utf-8") as f:
            for r in rows:
                new = convert(r["glosses"])
                changed += new != r["glosses"]
                f.write(json.dumps(r | {"glosses": new}, ensure_ascii=False) + "\n")
        print(f"{changed}/{len(rows)} lignes converties → {sys.argv[3]}")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
