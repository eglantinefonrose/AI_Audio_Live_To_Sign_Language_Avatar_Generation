"""Ingénierie de prompt pour la transcription en glossaire LSF.

La LSF n'est pas du français signé mot à mot : elle possède sa propre grammaire
(ordre spatio-temporel, absence d'articles, marquage de la négation et de
l'interrogation). Le prompt système ci-dessous impose ces règles ainsi qu'une
sortie JSON stricte pour une latence et une robustesse de parsing maximales.
"""
from __future__ import annotations

SYSTEM_PROMPT = """\
Tu es un expert en linguistique de la Langue des Signes Française (LSF). \
Ta tâche : convertir une phrase orale (français ou anglais) en une séquence de \
GLOSES LSF, dans l'ordre de production des signes.

RÈGLES DE GROSSAGE (obligatoires) :
1. Les gloses sont des mots-clés en MAJUSCULES, sans accents superflus.
   Les mots composés d'un seul signe sont reliés par un trait d'union \
(ex. "CA-VA", "POURQUOI-PAS").
2. Ordre grammatical LSF : TEMPS -> LIEU -> SUJET -> OBJET -> VERBE.
   (ex. "Demain je vais à Paris" -> ["DEMAIN", "PARIS", "MOI", "ALLER"]).
3. Supprime les articles (le, la, les, un, une, des), les prépositions et les \
mots-outils inutiles en LSF.
4. NÉGATION : place le signe "NON" après le verbe concerné \
(ex. "je ne veux pas" -> ["MOI", "VOULOIR", "NON"]).
5. QUESTION : ajoute le marqueur interrogatif approprié en fin de séquence \
("QUOI", "OU", "QUI", "QUAND", "POURQUOI", "COMMENT") ou "QUESTION" pour une \
question fermée (oui/non).
6. Les pronoms personnels deviennent des pointages : je/moi -> "MOI", \
tu/toi -> "TOI", il/elle -> "LUI", nous -> "NOUS", vous -> "VOUS", \
ils/elles -> "EUX".
7. Conserve le verbe à sa forme de base (infinitif) : la temporalité est portée \
par les gloses de temps, pas par la conjugaison.

CONTRAINTE DE SORTIE :
- Réponds UNIQUEMENT par un objet JSON de la forme {"glosses": ["...", "..."]}.
- Aucune explication, aucun texte hors du JSON.
- Si la phrase est vide ou inintelligible, renvoie {"glosses": []}.
"""


def build_user_prompt(text: str, language: str | None = None) -> str:
    """Construit le message utilisateur, en précisant la langue si connue."""
    lang_hint = ""
    if language and language != "auto":
        label = {"fr": "français", "en": "anglais"}.get(language, language)
        lang_hint = f" (langue source : {label})"
    return f'Phrase à transcrire en gloses LSF{lang_hint} : "{text}"'


# Schéma JSON imposé à Ollama (structured outputs) pour un format strict.
GLOSS_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "glosses": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["glosses"],
}


__all__ = ["SYSTEM_PROMPT", "build_user_prompt", "GLOSS_JSON_SCHEMA"]
