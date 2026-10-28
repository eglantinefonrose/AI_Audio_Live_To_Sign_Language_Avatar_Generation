"""Post-traitement déterministe des gloses produites par le LLM.

Applique, à partir de la phrase source, les règles de CONVENTIONS.md qui se
vérifient mécaniquement et que le modèle ne suit pas toujours : forme, pronoms,
mots-outils, lexique imposé, orthographe des mots source, gloses inventées,
verbes directionnels, possessifs, ils/elles -> EUX, temps (AVANT/BIENTÔT, temps
explicite en tête), négation et marqueur interrogatif en fin.

Garde-fou : appliqué aux gloses de RÉFÉRENCE (train + dev), il ne doit rien
changer (`python -m lsf_translator.postprocess`). Une règle qui modifie une
référence est trop large.

    from lsf_translator.postprocess import postprocess
    glosses, applied = postprocess("Il pleut encore ?", ["LUI", "PLEU", "ENCORE"])
"""
from __future__ import annotations

import difflib
import json
import re
import sys
from pathlib import Path

from lsf_translator.conventions import (
    CONNECTOR_END,
    FUNCTION_WORDS,
    IMPERSONAL,
    NEAR_FUTURE_RE,
    NEG_RE,
    OTHER_TIME_RE,
    TECH_TERMS,
    explicit_times,
    implicit_time,
)
from lsf_translator.lexicon import (
    CLOSED_CLASS,
    NEGATIONS,
    POINTING_OF,
    POINTS,
    WH,
    directional_verbs,
    lemma,
    lexicon,
    lexicon_parts,
    strip_accents,
    tokens,
    verb_guesses,
)

PRONOUN_GLOSS = {"JE": "MOI", "J'": "MOI", "TU": "TOI", "IL": "LUI", "ELLE": "LUI",
                 "ILS": "EUX", "ELLES": "EUX", "ON": "NOUS"}
MODALS = {"VOULOIR", "POUVOIR", "DEVOIR"}
ADVERBS = {"BEAUCOUP", "BIEN", "MAL", "ENCORE", "DÉJÀ", "SOUVENT", "TOUJOURS", "VITE",
           "DIRECT", "TROP", "PEU"}
WH_SRC = {"où": "OÙ", "quand": "QUAND", "pourquoi": "POURQUOI", "comment": "COMMENT",
          "combien": "COMBIEN", "quoi": "QUOI"}
SUBJECTS = r"(je|j'|tu|il|elle|on|nous|vous|ils|elles)"
CLITICS = r"(me|m'|te|t'|lui|leur|nous|vous)"
AUX = {"ai", "as", "a", "avons", "avez", "ont", "suis", "es", "est", "sommes", "êtes", "sont"}
# « je vous remercie », « tu lui as dit », « je vais vous présenter », « il veut te voir »
DIRECTIONAL_RE = re.compile(
    rf"\b(?:{SUBJECTS}\s+(?:ne\s+|n')?|(?:vais|vas|va|allons|allez|vont|veux|veut|voulons|"
    rf"voulez|veulent|peux|peut|pouvons|pouvez|peuvent|dois|doit|devons|devez|doivent)\s+)"
    rf"{CLITICS}\s*([a-zà-ÿ]+)(?:\s+([a-zà-ÿ]+))?")


def _ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, strip_accents(a), strip_accents(b)).ratio()


class _Ctx:
    """Ce que la phrase source permet de vérifier."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.low = text.lower().replace("’", "'")
        self.toks = tokens(text)
        self.lemmas = [lemma(t) for t in self.toks]
        self.src_upper = {t.upper() for t in self.toks}
        self.lemma_upper = {l.upper() for l in self.lemmas}
        self.verbs = {v for t in self.toks for v in verb_guesses(t)}
        self.support = self.src_upper | self.lemma_upper | self.verbs

    def supported(self, gloss: str) -> bool:
        """Gloss traçable : classe fermée, lexique imposé, ou mot/lemme source."""
        parts = gloss.split("-À-")[0].split("-")
        return all(p in CLOSED_CLASS or p in self.support or p.isdigit()
                   or any(_ratio(p, s) >= 0.7 for s in self.support) for p in parts if p)


def _head(g: list[str]) -> int:
    """Position de tête (après ENFIN)."""
    return 1 if g and g[0] == "ENFIN" else 0


def _tail(g: list[str]) -> int:
    """Début des interrogatifs finaux (QUESTION, QUOI, OÙ…)."""
    i = len(g)
    while i > 0 and (g[i - 1] in WH or g[i - 1] == "QUESTION"):
        i -= 1
    return i


# -- règles (chacune : (ctx, gloses) -> gloses) ------------------------------
def r_forme(ctx: _Ctx, g: list[str]) -> list[str]:
    return [re.sub(r"\s+", "-", x.strip().upper()) for x in g if x.strip()]


def r_pronoms(ctx: _Ctx, g: list[str]) -> list[str]:
    g = [PRONOUN_GLOSS.get(x, x) for x in g]
    # « il » impersonnel (il pleut, il faut…) : pas de pointage, sauf si la
    # phrase a un autre il/elle/lui/son… qui en demande un.
    if IMPERSONAL.search(ctx.low) and "LUI" in g:
        others = sum(POINTING_OF.get(t) == "LUI" for t in ctx.toks) - 1
        if g.count("LUI") > others:
            g.remove("LUI")
    return g


def r_mots_outils(ctx: _Ctx, g: list[str]) -> list[str]:
    return [x for x in g if x not in FUNCTION_WORDS]


def r_lexique(ctx: _Ctx, g: list[str]) -> list[str]:
    out: list[str] = []
    for i, x in enumerate(g):
        if x in {"LIVE", "EN-DIRECT", "EN-LIVE"}:
            out.append("DIRECT")
        elif x == "ADORER":
            out.append("AIMER")
            if i + 1 >= len(g) or g[i + 1] != "BEAUCOUP":
                out.append("BEAUCOUP")
        else:
            out.append(x)
    # Termes techniques en plusieurs mots : « RANDOM » « FOREST » -> RANDOM-FOREST.
    for src, gl in TECH_TERMS.items():
        parts = gl.split("-")
        if len(parts) > 1 and re.search(rf"\b{re.escape(src)}\b", ctx.low):
            for i in range(len(out) - len(parts) + 1):
                if out[i:i + len(parts)] == parts:
                    out[i:i + len(parts)] = [gl]
                    break
    return out


def _fix_word(ctx: _Ctx, x: str) -> str:
    """Graphie d'un mot : rapprochée du mot source, verbe à l'infinitif, nom au
    singulier. Ne touche jamais une glose connue du lexique."""
    lex = lexicon_parts()
    if x in CLOSED_CLASS or x in lex or x.isdigit():
        return x
    if x not in ctx.support:   # graphie inventée : mot ou lemme source le plus proche
        best = max(ctx.support, key=lambda s: _ratio(x, s), default=None)
        if best and _ratio(x, best) >= 0.75:
            x = best
    if x in ctx.src_upper and x not in lex:   # mot source tel quel mais fléchi
        t = x.lower()
        lm = lemma(t)
        if lm != t and (re.search(r"(er|ir|re|oir)$", lm) or (t[-1] in "sx" and lm == t[:-1])):
            x = lm.upper()
    return x


def r_orthographe(ctx: _Ctx, g: list[str]) -> list[str]:
    out = []
    for x in g:
        if "-À-" in x:
            verb, _, point = x.partition("-À-")
            out.append(f"{_fix_word(ctx, verb)}-À-{point}")
        elif "-" in x and x not in lexicon_parts():
            out.append("-".join(_fix_word(ctx, p) for p in x.split("-")))
        else:
            out.append(_fix_word(ctx, x))
    return out


def r_inventees(ctx: _Ctx, g: list[str]) -> list[str]:
    """Supprime une glose sans lien avec la phrase, seulement si c'est sûrement
    une invention : graphie hors lexique (« AÉ »), ou mot répété (« VOITURE »
    ×3). Un signe connu sans mot source littéral (VITE pour « rapidement »,
    ÇA-VA, PLEUVOIR pour « plu ») est gardé."""
    lex = lexicon_parts()
    return [x for x in g if ctx.supported(x)
            or (all(p in lex for p in x.split("-À-")[0].split("-") if p) and g.count(x) == 1)]


def r_doublons(ctx: _Ctx, g: list[str]) -> list[str]:
    return [x for i, x in enumerate(g) if i == 0 or x != g[i - 1]]


def r_directionnel(ctx: _Ctx, g: list[str]) -> list[str]:
    g = list(g)
    for m in DIRECTIONAL_RE.finditer(ctx.low):
        subj, clitic, w1, w2 = m.group(1), m.group(2), m.group(3), m.group(4)
        point = POINTING_OF[clitic.rstrip("'")] if clitic != "lui" else "LUI"
        if subj and POINTING_OF.get(subj.rstrip("'")) == point:
            continue   # réfléchi (« nous nous voyons »)
        verb = w2 if w1 in AUX and w2 else w1
        targets = {lemma(verb).upper(), *verb_guesses(verb)} & directional_verbs()
        if not targets or any(x.endswith(f"-À-{point}") for x in g):
            continue   # seuls les verbes que les données composent (DONNER-À-, …)
        cands = [i for i, x in enumerate(g) if x not in CLOSED_CLASS
                 and any(x == t or _ratio(x, t) >= 0.75 for t in targets)]
        if not cands:
            continue
        j = cands[0]
        g[j] = f"{g[j]}-À-{point}"
        # Le pointage isolé du destinataire est absorbé par le verbe, sauf s'il
        # est aussi sujet / possessif ailleurs dans la phrase.
        others = sum(POINTING_OF.get(t) == point for t in ctx.toks) - 1
        if g.count(point) > others:
            g.remove(point)
            j = next(i for i, x in enumerate(g) if x.endswith(f"-À-{point}"))
        # Le verbe va en fin de proposition (avant modal, adverbe, négation, interrogatif).
        stop = MODALS | ADVERBS | set(NEGATIONS) | set(WH) | {"QUESTION"}
        k = j + 1
        while k < len(g) and g[k] not in stop:
            k += 1
        g.insert(k - 1, g.pop(j))
    return g


def r_possessifs(ctx: _Ctx, g: list[str]) -> list[str]:
    g = list(g)
    poss = {"mon", "ma", "mes", "ton", "ta", "tes", "son", "sa", "ses", "notre", "nos",
            "votre", "vos", "leurs"}
    for i, t in enumerate(ctx.toks[:-1]):
        if t not in poss:
            continue
        point = POINTING_OF[t]
        if g.count(point) >= sum(POINTING_OF.get(u) == point for u in ctx.toks):
            continue   # déjà autant de pointages que de pronoms/possessifs source
        noun = ctx.toks[i + 1]
        forms = {noun.upper(), lemma(noun).upper()}
        j = next((k for k, x in enumerate(g) if x in forms
                  or any(_ratio(x, f) >= 0.85 for f in forms)), None)
        if j is not None and (j + 1 >= len(g) or g[j + 1] != point):
            g.insert(j + 1, point)
    return g


def r_eux(ctx: _Ctx, g: list[str]) -> list[str]:
    toks = set(ctx.toks)
    if toks & {"ils", "elles"} and not toks & {"il", "elle", "lui", "son", "sa", "ses"} \
            and "EUX" not in g and "LUI" in g:
        return ["EUX" if x == "LUI" else x for x in g]
    return g


def r_temps(ctx: _Ctx, g: list[str]) -> list[str]:
    g = list(g)
    explicit = explicit_times(ctx.text)
    marker = implicit_time(ctx.text)
    if explicit:
        g = [x for x in g if x not in {"AVANT", "BIENTÔT"}]
        times = [x for x in g if x in explicit]
        rest = [x for x in g if x not in explicit]
        h = _head(rest)
        g = rest[:h] + times + rest[h:]
    elif marker:
        other = "BIENTÔT" if marker == "AVANT" else "AVANT"
        g = [x for x in g if x not in {marker, other}]
        g.insert(_head(g), marker)
    elif re.search(r"\bdéjà\b", ctx.low):
        # « déjà » : adverbe après le verbe, qui dispense de AVANT.
        g = [x for x in g if x != "AVANT"]
        if "DÉJÀ" not in g:
            k = len(g)
            while k > 0 and g[k - 1] in {*NEGATIONS, *WH, "QUESTION"}:
                k -= 1
            g.insert(k, "DÉJÀ")
    elif not OTHER_TIME_RE.search(ctx.low):
        # Aucun temps dans la phrase source : AVANT / BIENTÔT inventés.
        g = [x for x in g if x not in {"AVANT", "BIENTÔT"}]
    m = NEAR_FUTURE_RE.search(ctx.low)
    if m and lemma(m.group(0).split()[-1]) != "aller" and "ALLER" in g:
        g.remove("ALLER")   # « aller » + infinitif = futur, pas le verbe ALLER
    return g


def r_negation(ctx: _Ctx, g: list[str]) -> list[str]:
    if not NEG_RE.search(ctx.text):
        return g
    g = list(g)
    specific = next((gl for w, gl in (("jamais", "JAMAIS"), ("rien", "RIEN"), ("personne", "PERSONNE"))
                     if re.search(rf"\b{w}\b", ctx.low)), None)
    if specific and "NON" in g and not re.search(r"\bpas\b", ctx.low):
        g.remove("NON")
    negs = [x for x in g if x in NEGATIONS]
    if not negs:
        negs = [specific or "NON"]
    g = [x for x in g if x not in negs]
    t = _tail(g)
    return g[:t] + negs + g[t:]


def r_question(ctx: _Ctx, g: list[str]) -> list[str]:
    if not ctx.text.strip().endswith("?"):
        return g
    g = list(g)
    wh = [x for x in g if x in WH]
    if not wh:
        src = next((WH_SRC[t] for t in ctx.toks if t in WH_SRC), None)
        if src is None and (ctx.low.startswith(("qui ", "qui est")) or " qui " in f" {ctx.low} "
                            and ctx.toks and ctx.toks[0] == "qui"):
            src = "QUI"
        if src is None and re.match(r"\s*(que|qu'est-ce)\b", ctx.low):
            src = "QUOI"
        if src:
            wh = [src]
    g = [x for x in g if x not in WH and x != "QUESTION"]
    return g + (wh if wh else ["QUESTION"])


def r_connecteur(ctx: _Ctx, g: list[str]) -> list[str]:
    if ctx.low.lstrip().startswith(CONNECTOR_END):
        return ["ENFIN"] + [x for x in g if x != "ENFIN"]
    return g


RULES = [
    ("forme", r_forme), ("pronoms", r_pronoms), ("mots-outils", r_mots_outils),
    ("lexique", r_lexique), ("orthographe", r_orthographe), ("inventées", r_inventees),
    ("doublons", r_doublons), ("directionnel", r_directionnel), ("possessifs", r_possessifs),
    ("eux", r_eux), ("temps", r_temps), ("négation", r_negation), ("question", r_question),
    ("connecteur", r_connecteur),
]


def postprocess(text: str, glosses: list[str]) -> tuple[list[str], list[str]]:
    """Gloses corrigées + noms des règles qui ont modifié quelque chose."""
    ctx = _Ctx(text)
    applied: list[str] = []
    g = list(glosses)
    if not g:
        return g, applied
    for name, rule in RULES:
        new = rule(ctx, g)
        if new != g:
            applied.append(name)
            g = new
    return g, applied


def _check_references(paths: list[Path]) -> int:
    """Garde-fou : les références ne doivent pas être modifiées."""
    changed = total = 0
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if "messages" in r:
                text = re.search(r'"(.*)"\s*$', r["messages"][-2]["content"]).group(1)
                gold = json.loads(r["messages"][-1]["content"])["glosses"]
            else:
                text, gold = r["text"], r["glosses"]
            total += 1
            out, applied = postprocess(text, gold)
            if out != gold:
                changed += 1
                print(f"  ✗ {text}\n      réf : {' '.join(gold)}\n      pp  : {' '.join(out)}  "
                      f"({', '.join(applied)})")
    print(f"── {total - changed}/{total} références inchangées")
    return changed


if __name__ == "__main__":
    project = Path(__file__).resolve().parent.parent
    default = [project / "finetune/data/2184f8f3751e/train.jsonl", project / "eval/dev.jsonl"]
    sys.exit(1 if _check_references([Path(p) for p in sys.argv[1:]] or default) else 0)
