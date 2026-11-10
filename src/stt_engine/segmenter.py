"""Découpe des transcriptions trop longues en segments digestes pour le LLM.

La VAD de Kyutai valide une phrase sur un **silence** : si le locuteur enchaîne
sans pause franche, elle peut produire un « pavé » de plusieurs phrases (90+
mots). Or un LLM décroche sur ces entrées longues (translittération mot à mot,
fuite de connecteurs). Ce module **post-traite** le texte de la VAD — il ne la
remplace pas — pour émettre des unités courtes, en respectant au maximum les
frontières naturelles :

1. coupe d'abord sur la ponctuation de **fin de phrase** (``.!?…``) ;
2. regroupe les phrases courtes voisines jusqu'à ``max_words`` ;
3. si une phrase dépasse ``hard_cap``, coupe sur la ponctuation **interne**
   (``,;:``) ; en tout dernier recours, coupe par fenêtres de mots.

Fonctions pures, sans dépendance : faciles à tester isolément.
"""
from __future__ import annotations

import re

# Frontières de découpe (on coupe APRÈS la ponctuation, d'où le lookbehind).
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
_CLAUSE = re.compile(r"(?<=[,;:])\s+")


def _word_count(text: str) -> int:
    return len(text.split())


def _pack(units: list[str], max_words: int) -> list[str]:
    """Regroupe des unités adjacentes en segments d'au plus ``max_words`` mots.

    Une unité dont la taille dépasse déjà ``max_words`` est émise seule (elle
    n'est pas tronquée ici — c'est à l'appelant de l'avoir pré-découpée).
    """
    segments: list[str] = []
    buf: list[str] = []
    buf_words = 0
    for unit in units:
        uw = _word_count(unit)
        if buf and buf_words + uw > max_words:
            segments.append(" ".join(buf))
            buf, buf_words = [], 0
        buf.append(unit)
        buf_words += uw
    if buf:
        segments.append(" ".join(buf))
    return segments


def _split_by_words(text: str, size: int) -> list[str]:
    """Dernier recours : coupe brute par fenêtres de ``size`` mots."""
    words = text.split()
    return [" ".join(words[i : i + size]) for i in range(0, len(words), size)]


def split_into_segments(
    text: str,
    max_words: int = 25,
    hard_cap: int | None = None,
) -> list[str]:
    """Découpe ``text`` en segments courts adaptés au LLM.

    :param max_words: taille cible d'un segment (regroupement des phrases
        courtes jusqu'à ce seuil).
    :param hard_cap: au-delà de cette taille, une phrase est elle-même
        redécoupée (défaut : ``max_words`` × 1.6). Permet de garder entières les
        phrases un peu longues sans les charcuter, tout en bornant les pavés.
    :return: liste de segments non vides (jamais ``None``).
    """
    text = " ".join((text or "").split())  # normalise les espaces
    if not text:
        return []
    # Désactivation explicite : max_words <= 0 => aucun découpage.
    if max_words <= 0 or _word_count(text) <= max_words:
        return [text]

    if hard_cap is None:
        hard_cap = max(max_words, int(max_words * 1.6))

    pieces: list[str] = []
    for sentence in _SENTENCE_END.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if _word_count(sentence) <= hard_cap:
            # Phrase de taille raisonnable : gardée entière (sera éventuellement
            # regroupée avec ses voisines courtes à l'étape de packing).
            pieces.append(sentence)
            continue
        # Phrase trop longue : on coupe sur la ponctuation interne…
        clauses = [c.strip() for c in _CLAUSE.split(sentence) if c.strip()]
        for packed in _pack(clauses, max_words):
            if _word_count(packed) <= hard_cap:
                pieces.append(packed)
            else:
                # …puis, en dernier recours, découpe brute par mots.
                pieces.extend(_split_by_words(packed, max_words))

    # Regroupe les phrases courtes voisines pour éviter les micro-fragments.
    return [s for s in _pack(pieces, max_words) if s]


__all__ = ["split_into_segments"]
