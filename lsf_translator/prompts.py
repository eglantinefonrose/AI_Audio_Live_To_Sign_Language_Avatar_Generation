"""Ingénierie de prompt pour la transcription en glossaire LSF.

La LSF n'est pas du français signé mot à mot : elle possède sa propre grammaire
(ordre spatio-temporel, absence d'articles/prépositions, marquage de la négation
et de l'interrogation, verbe en fin). Ce module combine :

1. un **prompt système** qui énonce les règles de glossage ;
2. un jeu d'**exemples few-shot** (paires phrase -> gloses) donnés comme de
   vrais tours de dialogue user/assistant au format JSON attendu — c'est le
   levier le plus efficace pour qu'un LLM reproduise à la fois la grammaire LSF
   et le format de sortie ;
3. une **sortie JSON stricte** (schéma imposé à Ollama) pour un parsing robuste.

Montrer vaut mieux que décrire : les règles négatives (« n'utilise pas DE »)
sont mal suivies seules, mais le sont bien lorsqu'elles sont *démontrées* par les
exemples. Pour régler la latence/coût en temps réel, on peut réduire
``FEW_SHOT_EXAMPLES`` : la diversité compte plus que le nombre.
"""
from __future__ import annotations

import json

SYSTEM_PROMPT = """\
Tu es un expert en linguistique de la Langue des Signes Française (LSF). \
Ta tâche : convertir une phrase orale (français ou anglais) en une séquence de \
GLOSES LSF, dans l'ordre de production des signes.

RÈGLES DE GLOSSAGE (obligatoires) :
1. Les gloses sont des mots-clés en MAJUSCULES. Les notions rendues par un seul \
signe sont reliées par un trait d'union (ex. "CA-VA", "DONNER-À-TOI").
2. Ordre grammatical LSF : TEMPS -> LIEU -> SUJET -> OBJET -> VERBE. \
Le verbe se place en fin ; le thème (de quoi on parle) se place en tête \
(structure topic–commentaire).
3. SUPPRIME les mots-outils absents de la LSF : articles (le, la, les, un, une, \
des), prépositions et conjonctions (de, dans, à, pour, que, avec, en...). \
Ne les transforme JAMAIS en glose.
4. NÉGATION : place le signe "NON" APRÈS le verbe concerné.
5. QUESTION : place le marqueur interrogatif EN FIN de séquence \
("QUOI", "OÙ", "QUI", "QUAND", "POURQUOI", "COMMENT", "COMBIEN"). N'ajoute \
"QUESTION" QUE pour une question fermée (oui/non) SANS mot interrogatif — ne \
mets JAMAIS un mot interrogatif ET "QUESTION" ensemble. La négation ("NON") \
d'une question reste présente (ex. "pourquoi tu ne viens pas ?" -> \
"TOI" "VENIR" "NON" "POURQUOI").
6. Les pronoms deviennent des pointages : je/moi -> "MOI", tu/toi -> "TOI", \
il/elle -> "LUI", nous -> "NOUS", vous -> "VOUS", ils/elles -> "EUX".
7. L'adjectif se place APRÈS le nom (ex. "voiture bleue" -> "VOITURE" "BLEU"). \
Le nombre se place APRÈS le nom (ex. "deux méthodes" -> "MÉTHODE" "DEUX").
8. Le verbe reste à l'INFINITIF (forme de base), jamais conjugué \
(ex. "tu pars" -> "PARTIR" et non "PARS" ; "il a acheté" -> "ACHETER"). La \
temporalité est portée par une glose de TEMPS en tête ("AVANT" = passé, \
"BIENTÔT"/"FUTUR" = futur, "MAINTENANT"), jamais par un auxiliaire. N'emploie \
PAS "aller"/"avoir" comme auxiliaire : "je vais manger" -> "BIENTÔT" "MOI" \
"MANGER" (et non "MOI" "ALLER" "MANGER").
9. Les verbes directionnels incorporent leurs actants \
(ex. "je te donne" -> "DONNER-À-TOI").
10. La possession se marque par un pointage après le nom \
(ex. "ma voiture" -> "VOITURE" "MOI").
11. CONSERVE LES MOTS SOURCE TELS QUELS. Ne ré-orthographie pas, ne traduis pas, \
n'invente aucune graphie. Une glose issue d'un mot de la phrase est ce mot \
simplement mis en MAJUSCULES, avec ses lettres et accents d'origine. \
Les noms propres, termes techniques et mots étrangers (ex. "camembert", \
"random forest", "fine tuning", "dataset") sont repris À L'IDENTIQUE, jamais \
traduits ni francisés (ex. "forest" -> "FOREST", et NON "FORÊT"). \
En cas de doute sur un mot, recopie-le tel quel plutôt que de le deviner.

LEXIQUE IMPOSÉ (utilise EXACTEMENT ces gloses, sans les traduire autrement) :
- "en direct", "en live", "live" -> DIRECT
- "pour finir", "pour conclure", "en conclusion", "pour terminer" -> ENFIN
- "du coup", "donc", "alors" (connecteurs de discours) -> SUPPRIMÉS
- termes techniques ou anglais du domaine ("dataset", "fine tuning", \
"random forest", "camembert", "machine learning") -> repris À L'IDENTIQUE en \
MAJUSCULES, jamais traduits ni francisés.

CONTRAINTE DE SORTIE :
- Réponds UNIQUEMENT par un objet JSON de la forme {"glosses": ["...", "..."]}.
- Aucune explication, aucun texte hors du JSON.
- Si la phrase est vide ou inintelligible, renvoie {"glosses": []}.

Des exemples te sont fournis ci-dessous : suis scrupuleusement le même style, \
le même ordre et le même format de sortie.
"""


# Exemples few-shot : (phrase source, gloses attendues).
# Choisis pour leur DIVERSITÉ (chacun illustre un phénomène grammatical précis),
# pas pour leur nombre. Ils sont injectés comme de vrais tours user/assistant.
FEW_SHOT_EXAMPLES: list[tuple[str, list[str]]] = [
    # Temps en tête + lieu + sujet + verbe final.
    ("Demain, je vais à Paris.", ["DEMAIN", "PARIS", "MOI", "ALLER"]),
    # Négation placée après le verbe ; thème (réunion) en tête.
    ("Je ne veux pas aller à la réunion.",
     ["RÉUNION", "MOI", "ALLER", "VOULOIR", "NON"]),
    # Question ouverte : marqueur interrogatif en fin.
    ("Où habites-tu ?", ["TOI", "HABITER", "OÙ"]),
    # Question fermée (oui/non) : "QUESTION" en fin.
    ("Est-ce que tu viens demain ?", ["DEMAIN", "TOI", "VENIR", "QUESTION"]),
    # Possession par pointage + adjectif après le nom + suppression de "est".
    ("Ma voiture est bleue.", ["VOITURE", "MOI", "BLEU"]),
    # Nombre après le nom + "avoir" non calqué (existence).
    ("J'ai deux chats.", ["MOI", "CHAT", "DEUX"]),
    # Suppression explicite de "de", "dans", "un" ; thème en tête, verbe en fin.
    ("Je parle de la détection dans un flux.",
     ["FLUX", "DÉTECTION", "MOI", "EXPLIQUER"]),
    # Verbe directionnel : le pronom objet est incorporé au verbe.
    ("Je te donne le livre.", ["LIVRE", "MOI", "DONNER-À-TOI"]),
    # Topicalisation : le thème fronté, commentaire ensuite.
    ("Le café, j'adore ça.", ["CAFÉ", "MOI", "AIMER", "BEAUCOUP"]),
    # Futur proche : marqueur temporel, PAS d'auxiliaire "aller".
    ("Je vais manger.", ["BIENTÔT", "MOI", "MANGER"]),
    # Passé : marqueur "AVANT", "j'ai" non glosé ; nombre après le nom.
    ("J'ai utilisé deux méthodes.",
     ["AVANT", "MÉTHODE", "DEUX", "MOI", "UTILISER"]),
    # Marqueur de discours "pour finir" -> "ENFIN" ; verbe directionnel.
    ("Pour finir, je vais vous expliquer les résultats.",
     ["ENFIN", "RÉSULTAT", "MOI", "EXPLIQUER-À-VOUS"]),
    # Lexique : "en direct" -> DIRECT (jamais "vivant"/"en-direct").
    ("Je suis l'émission en direct.", ["ÉMISSION", "MOI", "SUIVRE", "DIRECT"]),
    # Lexique : connecteur "pour conclure" -> ENFIN (exemple sans verbe
    # directionnel, pour ne pas déborder sur des cas comme "remercier").
    ("Pour conclure, tout va bien.", ["ENFIN", "TOUT", "CA-VA"]),
    # Question ouverte + négation : mot interrogatif en fin, "NON" conservé,
    # PAS de "QUESTION" en plus ; verbe à l'infinitif.
    ("Pourquoi il ne répond pas ?", ["LUI", "RÉPONDRE", "NON", "POURQUOI"]),
    # Source anglaise : négation après le verbe.
    ("I don't understand.", ["MOI", "COMPRENDRE", "NON"]),
]


def build_user_prompt(text: str, language: str | None = None) -> str:
    """Construit le message utilisateur, en précisant la langue si connue."""
    lang_hint = ""
    if language and language != "auto":
        label = {"fr": "français", "en": "anglais"}.get(language, language)
        lang_hint = f" (langue source : {label})"
    return f'Phrase à transcrire en gloses LSF{lang_hint} : "{text}"'


def _assistant_example(glosses: list[str]) -> str:
    """Rend un exemple de sortie exactement au format JSON attendu."""
    return json.dumps({"glosses": glosses}, ensure_ascii=False)


def build_messages(text: str, language: str | None = None) -> list[dict]:
    """Assemble la liste de messages complète pour l'appel au LLM.

    Structure : [système] + [paires user/assistant d'exemples] + [phrase réelle].
    Fournir les exemples comme de vrais tours de dialogue (plutôt que dans le
    prompt système) ancre fortement le format de sortie JSON et le style des
    gloses — c'est la bonne pratique de few-shot avec une API de chat.
    """
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for source, glosses in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": build_user_prompt(source)})
        messages.append({"role": "assistant", "content": _assistant_example(glosses)})
    messages.append({"role": "user", "content": build_user_prompt(text, language)})
    return messages


# System prompt d'ENTRAÎNEMENT (fine-tuning) : le SYSTEM_PROMPT de prod moins la
# phrase finale qui renvoie aux exemples few-shot (absents du train — sinon
# consigne mensongère). C'est aussi le SYSTEM de finetune/Modelfile.
TRAIN_SYSTEM = SYSTEM_PROMPT.split("\nDes exemples te sont fournis")[0].rstrip() + "\n"


def build_rules_only_messages(text: str, language: str | None = None) -> list[dict]:
    """Prompt « règles seules », identique à celui des exemples d'entraînement :
    [TRAIN_SYSTEM] + [phrase réelle], sans few-shot. À utiliser pour servir ou
    évaluer un modèle fine-tuné (la connaissance est dans les poids)."""
    return [
        {"role": "system", "content": TRAIN_SYSTEM},
        {"role": "user", "content": build_user_prompt(text, language)},
    ]


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


__all__ = [
    "SYSTEM_PROMPT",
    "FEW_SHOT_EXAMPLES",
    "build_user_prompt",
    "build_messages",
    "GLOSS_JSON_SCHEMA",
]
