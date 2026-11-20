"""Post-traitement déterministe des gloses produites par le LLM.

Applique, à partir de la phrase source, les règles de CONVENTIONS.md qui se
vérifient mécaniquement et que le modèle ne suit pas toujours : forme, pronoms,
mots-outils, lexique imposé, orthographe des mots source, gloses inventées,
verbes directionnels, possessifs, ils/elles -> EUX, temps (AVANT/BIENTÔT, temps
explicite en tête), négation et marqueur interrogatif en fin.

Garde-fou : appliqué aux gloses de RÉFÉRENCE (train + dev), il ne doit rien
changer (`python -m lsf_translator.postprocess`). Une règle qui modifie une
référence est trop large.

    from src.lsf_translator.postprocess import postprocess
    glosses, applied = postprocess("Il pleut encore ?", ["LUI", "PLEU", "ENCORE"])
"""
from __future__ import annotations

import difflib
import json
import re
import sys
from pathlib import Path

from src.lsf_translator.conventions import (
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
from src.lsf_translator.lexicon import (
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
                 "ILS": "EUX", "ELLES": "EUX", "ON": "NOUS",
                 # possessifs écrits tels quels (MON -> pointage, placé ensuite après le nom)
                 "MON": "MOI", "MA": "MOI", "MES": "MOI", "TON": "TOI", "TA": "TOI",
                 "TES": "TOI", "SON": "LUI", "SA": "LUI", "SES": "LUI", "NOTRE": "NOUS",
                 "NOS": "NOUS", "VOTRE": "VOUS", "VOS": "VOUS", "LEURS": "EUX"}
SECOND_PERSON = {"TOI": {"tu", "toi", "te", "t", "ton", "ta", "tes"},
                 "VOUS": {"vous", "votre", "vos"}}
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

    def __init__(self, text: str, lsfb: bool = False) -> None:
        self.text = text
        self.lsfb = lsfb
        self.low = text.lower().replace("’", "'")
        self.toks = tokens(text)
        self.lemmas = [lemma(t) for t in self.toks]
        self.src_upper = {t.upper() for t in self.toks}
        self.lemma_upper = {l.upper() for l in self.lemmas}
        self.verbs = {v for t in self.toks for v in verb_guesses(t)}
        self.support = self.src_upper | self.lemma_upper | self.verbs
        if lsfb:   # un synonyme LSFB d'un mot de la phrase est lié à elle (PÈRE -> PAPA)
            from src.lsf_translator.lsfb import synonyms
            syn = synonyms()
            self.support |= {p for w in self.support if w in syn for p in syn[w]}

    def supported(self, gloss: str) -> bool:
        """Gloss traçable : classe fermée, lexique imposé, ou mot/lemme source."""
        parts = gloss.split("-À-")[0].split("-")
        return all(p in CLOSED_CLASS or p in self.support or p.isdigit()
                   or any(_ratio(p, s) >= 0.7 for s in self.support) for p in parts if p)


def _known(ctx: _Ctx) -> frozenset[str]:
    """Gloses connues : lexique d'entraînement, plus le vocabulaire LSFB s'il est activé
    (un signe LSFB choisi par le modèle, PAPA pour « père », n'est pas une invention)."""
    if not ctx.lsfb:
        return lexicon_parts()
    from src.lsf_translator.lsfb import vocab
    return lexicon_parts() | frozenset(p for g in vocab() for p in [g, *g.split("-")])


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
    # TOI / VOUS : la personne de la phrase source (« as-tu garé » -> TOI, pas VOUS).
    toks = set(ctx.toks)
    for wrong, right in (("VOUS", "TOI"), ("TOI", "VOUS")):
        if not toks & SECOND_PERSON[wrong] and toks & SECOND_PERSON[right]:
            g = [right if x == wrong else x for x in g]
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
    lex = _known(ctx)
    if x in CLOSED_CLASS or x in lex or x.isdigit():
        return x
    attr = _attribute_form(ctx, x)
    if attr:
        return attr
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


COPULA = {"est", "sont", "suis", "es", "sommes", "êtes", "était", "étaient", "semble", "semblent"}
INTENSIFIERS = {"très", "trop", "si", "plus", "moins", "assez", "pas", "bien", "vraiment", "un", "peu"}


def _attribute_form(ctx: _Ctx, x: str) -> str | None:
    """Glose d'un attribut du sujet (« les villes sont polluées », « elle est très
    claire ») : adjectif au masculin singulier (POLLUÉ, CLAIR), jamais l'infinitif."""
    for k, t in enumerate(ctx.toks):
        if _ratio(x, t.upper()) < 0.75:
            continue
        j = k - 1
        while j >= 0 and ctx.toks[j] in INTENSIFIERS:
            j -= 1
        if j < 0 or ctx.toks[j] not in COPULA:
            continue
        w = t[:-1] if t.endswith("s") and len(t) > 3 else t
        if w.endswith("ée"):
            return w[:-1].upper()                       # polluée -> POLLUÉ
        lm = ctx.lemmas[k]
        if lm == w and w.endswith("e") and _is_french_word(w[:-1]):
            return w[:-1].upper()                       # claire -> CLAIR (lemme non fourni)
        if w.endswith("é") or re.search(r"(er|ir|re|oir)$", lm):
            return w.upper()                            # participe : forme gardée
        return lm.upper() if lm else w.upper()
    return None


def _is_french_word(w: str) -> bool:
    """Le mot existe dans le dictionnaire français de simplemma (« clair » oui, « roug » non)."""
    from src.lsf_translator.vocabulaire_lsfb import _accented, strip_accents
    return w.upper() in _accented().get(strip_accents(w).upper(), set())


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


NUMBER_WORDS = {"deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix", "onze",
                "douze", "treize", "quatorze", "quinze", "seize", "vingt", "trente", "quarante",
                "cinquante", "soixante", "cent", "cents", "mille"}
NEGATION_WORDS = {"ne", "n", "pas", "plus", "jamais", "rien", "personne"}
# Mots pris en charge par le lexique imposé ou sans signe propre (« en live » -> DIRECT,
# « ça » démonstratif, « y » de « il y a ») : jamais attendus comme glose à part.
IDIOM_WORDS = {"live", "direct", "ça", "ca", "cela", "ceci", "y", "chaque"}


def _expected(ctx: _Ctx) -> list[str]:
    """Gloses que les mots de contenu de la phrase appellent, dans l'ordre de la phrase
    (mot ou lemme connu, infinitif attesté). Hors pronoms, mots-outils, auxiliaires,
    « aller » du futur proche, négation, temps et nombres (règles dédiées)."""
    from src.lsf_translator.conventions import EXPLICIT_TIME
    known, out = _known(ctx), []
    if ctx.lsfb:
        from src.lsf_translator.lsfb import synonyms
        syn = synonyms()
    for k, t in enumerate(ctx.toks):
        lm = ctx.lemmas[k]
        nxt = ctx.toks[k + 1] if k + 1 < len(ctx.toks) else ""
        prev = ctx.toks[k - 1] if k else ""
        if (t in POINTING_OF or len(t) <= 1 or t.upper() in FUNCTION_WORDS or t.isdigit()
                or t in NEGATION_WORDS or t in NUMBER_WORDS or t in EXPLICIT_TIME
                or t in IDIOM_WORDS or lm in ("être", "avoir")
                or (lm == "aller" and (re.search(r"(er|ir|re|oir)$", nxt) or nxt == "bien"))
                or (t == "bien" and ctx.lemmas[k - 1] == "aller" if k else False)
                or (prev == "il" and IMPERSONAL.search(ctx.low))):
            continue
        for c in (t.upper(), lm.upper(), *sorted(verb_guesses(t))):
            if ctx.lsfb and c in syn:          # attendu dans le vocabulaire LSFB (MÈRE -> MAMAN)
                out.append(syn[c][0])
                break
            if c in known:
                out.append(c)
                break
    return out


def _covered(c: str, g: list[str]) -> bool:
    parts = [p for x in g for p in x.split("-À-")[0].split("-") if p]
    return any(c == p or _ratio(c, p) >= 0.85 for p in parts)


def r_inventees(ctx: _Ctx, g: list[str]) -> list[str]:
    """Glose sans lien avec la phrase (ni mot, ni lemme, ni classe fermée) :
    - s'il manque un mot de la phrase qui a un signe, elle est **remplacée** par lui, le
      plus proche en graphie (modèle quantifié : VOITURE pour « veux », GENDRE pour
      « gens », DIZER pour « dit ») ;
    - sinon supprimée si c'est sûrement une invention : graphie hors lexique (« AÉ »),
      ou mot répété (« VOITURE » ×3). Un signe connu sans mot source littéral (VITE pour
      « rapidement », ÇA-VA, PLEUVOIR pour « plu ») est gardé."""
    lex = _known(ctx)
    missing = [c for c in _expected(ctx) if not _covered(c, g)]
    out: list[str] = []
    for x in g:
        if ctx.supported(x):
            out.append(x)
        elif missing and ctx.lsfb and _lsfb_sign(x) and g.count(x) == 1 and \
                max(_ratio(x, c) for c in missing) < 0.5:
            out.append(x)   # signe LSFB choisi pour le sens (NOM pour « s'appeler »), pas une coquille
        elif missing:
            best = max(missing, key=lambda c: _ratio(x, c))
            missing.remove(best)
            out.append(best)
        elif all(p in lex for p in x.split("-À-")[0].split("-") if p) and g.count(x) == 1:
            out.append(x)
    return out


def _lsfb_sign(x: str) -> bool:
    from src.lsf_translator.lsfb import playable
    return playable(x)


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
        cands = [i for i, x in enumerate(g) if x not in POINTS
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
        elif clitic == "leur" and g[:j].count("LUI") > sum(POINTING_OF.get(t) == "LUI" for t in ctx.toks):
            # « leur » glosé LUI par le modèle : ce pointage en trop est le destinataire.
            del g[max(i for i in range(j) if g[i] == "LUI")]
            j -= 1
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


from src.lsf_translator.conventions import EXPLICIT_TIME as _EXPLICIT_TIME
EXPLICIT_TIME_WORDS = [re.escape(w) for w in _EXPLICIT_TIME]
_TIME_SRC: dict[str, str] = {}
for _w, _gl in _EXPLICIT_TIME.items():   # mot français d'abord (« hier », pas « yesterday »)
    _TIME_SRC.setdefault(_gl, _w)
DAYS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")
TIME_UNITS = {"minute": "MINUTE", "heure": "HEURE", "jour": "JOUR", "semaine": "SEMAINE",
              "mois": "MOIS", "an": "AN", "année": "ANNÉE"}
_UNIT_RE = r"(minutes?|heures?|jours?|semaines?|mois|ans?|années?)"
# « l'année prochaine », « lundi dernier », « la semaine passée »
COMPOUND_TIME_RE = re.compile(rf"\b({'|'.join(DAYS)}|semaine|année|mois|week-end)\s+"
                              r"(prochaine?|dernière?|derniers?)\b")
# « dans cinq minutes », « à huit heures », « dans trois semaines »
DURATION_RE = re.compile(rf"\b(?:dans|à|vers|pendant|depuis|en)\s+(\w+)\s+{_UNIT_RE}\b")


def _unit(w: str) -> str:
    return TIME_UNITS.get(w, TIME_UNITS.get(w[:-1] if w.endswith("s") else w, w.upper()))


def _time_block(ctx: _Ctx) -> tuple[list[str], set[str]]:
    """Temps composés de la phrase, dans l'ordre : (gloses du bloc de temps, gloses
    isolées qu'il remplace). ANNÉE-PROCHAINE, LUNDI-DERNIER, [MINUTE, CINQ], SAMEDI."""
    found: list[tuple[int, list[str]]] = []
    parts: set[str] = set()
    for m in COMPOUND_TIME_RE.finditer(ctx.low):
        a, b = m.group(1).upper(), m.group(2).upper().rstrip("S")
        found.append((m.start(), [f"{a}-{b}"]))
        parts |= {a, b, "PROCHAIN", "PROCHAINE", "DERNIER", "DERNIÈRE"}
    for m in DURATION_RE.finditer(ctx.low):
        if m.group(1) in NUMBER_WORDS:
            unit, num = _unit(m.group(2)), m.group(1).upper()
            found.append((m.start(), [unit, num]))
            parts |= {unit, num}
    for m in re.finditer(rf"\b({'|'.join(DAYS)})\b", ctx.low):
        if not any(s <= m.start() < s + 12 for s, _ in found):   # pas déjà dans un composé
            found.append((m.start(), [m.group(1).upper()]))
    return [x for _, gl in sorted(found) for x in gl], parts


def _remove_once(g: list[str], items: list[str]) -> list[str]:
    g = list(g)
    for x in items:
        if x in g:
            g.remove(x)
    return g


def r_temps(ctx: _Ctx, g: list[str]) -> list[str]:
    g = list(g)
    explicit = explicit_times(ctx.text)
    to = (lambda gl: [gl])
    if ctx.lsfb:   # un temps peut se signer autrement en LSFB (MAINTENANT -> AUJOURD'HUI)
        from src.lsf_translator.lsfb import to_lsfb as to
    block, parts = _time_block(ctx)
    if block:
        g = _remove_once(g, block)
        g = [x for x in g if x not in parts or x in explicit]
    if re.search(r"\bchaque\s+(" + "|".join(EXPLICIT_TIME_WORDS) + r")\b", ctx.low):
        g = [x for x in g if x != "CHAQUE"]   # « chaque matin » -> MATIN
    # Temps explicite oublié par le modèle : réinséré (MATIN, DEMAIN…).
    for gl in sorted(explicit, key=lambda e: ctx.low.find(_TIME_SRC.get(e, e.lower()))):
        if not any(v in g for v in (gl, *to(gl))):
            g.append(to(gl)[0])
    explicit = {e for gl in explicit for e in (gl, *to(gl))}
    marker = implicit_time(ctx.text)
    if explicit or block:
        g = [x for x in g if x not in {"AVANT", "BIENTÔT"}]
        times = [x for x in g if x in explicit]
        rest = [x for x in g if x not in explicit]
        h = _head(rest)
        g = rest[:h] + block + times + rest[h:]
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


def _matches(x: str, t: str, lm: str) -> bool:
    return x in (t.upper(), lm.upper()) or _ratio(x, t.upper()) >= 0.8


def r_nombres(ctx: _Ctx, g: list[str]) -> list[str]:
    """Nombre après le nom et ses adjectifs (« trois livres anciens » -> LIVRE ANCIEN
    TROIS) ; un nombre de la phrase oublié est réinséré après son nom."""
    g = list(g)
    block, _ = _time_block(ctx)
    toks = ctx.toks
    for k, t in enumerate(toks[:-1]):
        if t not in NUMBER_WORDS or (k and toks[k - 1] in NUMBER_WORDS):
            continue
        n = k                                       # « deux cents » -> DEUX-CENTS
        while n + 1 < len(toks) and toks[n + 1] in NUMBER_WORDS:
            n += 1
        num = "-".join(toks[k:n + 1]).upper()
        k = n
        if num in block or k + 1 >= len(toks):
            continue
        # Gloses des mots qui suivent le nombre (nom puis adjectifs), tant qu'elles se suivent.
        j = next((i for i, x in enumerate(g) if _matches(x, ctx.toks[k + 1], ctx.lemmas[k + 1])), None)
        if j is None:
            continue
        end, w = j + 1, k + 2
        while end < len(g) and w < len(ctx.toks) and _matches(g[end], ctx.toks[w], ctx.lemmas[w]):
            end, w = end + 1, w + 1
        if num in g:
            i = g.index(num)
            if i < j:
                g = g[:i] + g[i + 1:end] + [num] + g[end:]
        else:
            g.insert(end, num)
    return g


PLACE_PREP = r"(?:à|au|aux|en|dans|chez)"


def r_lieu(ctx: _Ctx, g: list[str]) -> list[str]:
    """Lieu juste après le temps (avant le sujet) : nom propre introduit par à/au/en/
    dans/chez (« à Nice », « en Italie »), ou « ici »."""
    places = [m.group(1).upper() for m in re.finditer(
        rf"\b{PLACE_PREP}\s+([A-ZÀ-Ý][\w-]+)", ctx.text.replace("’", "'"))]
    if re.search(r"\bici\b", ctx.low):
        places.append("ICI")
    places = [p for p in places if p in g]
    if not places:
        return g
    rest = [x for x in g if x not in places]
    block, _ = _time_block(ctx)
    times = {*block, *explicit_times(ctx.text), "AVANT", "BIENTÔT"}
    if ctx.lsfb:
        from src.lsf_translator.lsfb import to_lsfb
        times |= {v for e in list(times) for v in to_lsfb(e)}
    k = _head(rest)
    while k < len(rest) and rest[k] in times:
        k += 1
    return rest[:k] + places + rest[k:]


def r_negation(ctx: _Ctx, g: list[str]) -> list[str]:
    if not NEG_RE.search(ctx.text):
        return g
    g = list(g)
    specific = next((gl for w, gl in (("jamais", "JAMAIS"), ("rien", "RIEN"), ("personne", "PERSONNE"))
                     if re.search(rf"\b{w}\b", ctx.low)), None)
    if specific and "NON" in g and not re.search(r"\bpas\b", ctx.low):
        g.remove("NON")
    if re.search(r"\b(ne|n')\b.*\bplus\b", ctx.low) and "PLUS" in g:
        g.remove("PLUS")   # « ne … plus » -> NON seul
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
    if re.search(r"\bdu coup\b", ctx.low) and "COUP" in g:
        g = [x for x in g if x != "COUP"]   # connecteur de discours supprimé
    if ctx.low.lstrip().startswith(CONNECTOR_END):
        return ["ENFIN"] + [x for x in g if x != "ENFIN"]
    return g


def r_lsfb(ctx: _Ctx, g: list[str]) -> list[str]:
    """Vocabulaire LSFB (si activé) : graphie du vocabulaire (ECOLE -> ÉCOLE) ou
    synonyme jouable (PÈRE -> PAPA, REMERCIER-À-VOUS -> MERCI) ; un mot sans signe
    (nom propre, terme technique) est gardé : il sera épelé."""
    if not ctx.lsfb:
        return g
    from src.lsf_translator.lsfb import convert
    return convert(g)


RULES = [
    ("forme", r_forme), ("pronoms", r_pronoms), ("mots-outils", r_mots_outils),
    ("lexique", r_lexique), ("orthographe", r_orthographe), ("inventées", r_inventees),
    ("doublons", r_doublons), ("directionnel", r_directionnel), ("possessifs", r_possessifs),
    ("eux", r_eux), ("temps", r_temps), ("nombres", r_nombres), ("lieu", r_lieu),
    ("négation", r_negation), ("question", r_question),
    ("connecteur", r_connecteur), ("lsfb", r_lsfb),
]


def postprocess(text: str, glosses: list[str], lsfb: bool = False) -> tuple[list[str], list[str]]:
    """Gloses corrigées + noms des règles qui ont modifié quelque chose. `lsfb` ramène
    les gloses au vocabulaire LSFB (lsf_translator/lsfb.py)."""
    ctx = _Ctx(text, lsfb)
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


def source_text(messages: list[dict]) -> str:
    """Phrase source d'un exemple au format chat, quel que soit le format : avec ou sans
    message système (Gemma : règles fusionnées dans le message utilisateur), avec ou
    sans indications du vocabulaire LSFB après la phrase."""
    user = [m for m in messages if m["role"] == "user"][-1]["content"]
    line = next(l for l in user.splitlines() if l.startswith("Phrase à transcrire"))
    return re.search(r'"(.*)"\s*$', line).group(1)


def _check_references(paths: list[Path], lsfb: bool = False) -> int:
    """Garde-fou : les références ne doivent pas être modifiées."""
    changed = total = 0
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if "messages" in r:
                text = source_text(r["messages"])
                gold = json.loads(r["messages"][-1]["content"])["glosses"]
            else:
                text, gold = r["text"], r["glosses"]
            total += 1
            out, applied = postprocess(text, gold, lsfb=lsfb)
            if out != gold:
                changed += 1
                print(f"  ✗ {text}\n      réf : {' '.join(gold)}\n      pp  : {' '.join(out)}  "
                      f"({', '.join(applied)})")
    print(f"── {total - changed}/{total} références inchangées")
    return changed


if __name__ == "__main__":
    # python -m src.lsf_translator.postprocess [--lsfb] [fichiers.jsonl…]
    project = Path(__file__).resolve().parent.parent
    args = [a for a in sys.argv[1:] if a != "--lsfb"]
    default = [project / "finetune/data/2184f8f3751e/train.jsonl", project / "eval/dev.jsonl"]
    sys.exit(1 if _check_references([Path(p) for p in args] or default, lsfb="--lsfb" in sys.argv) else 0)
