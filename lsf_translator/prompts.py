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
1. Les gloses sont des mots-clés en MAJUSCULES, accents conservés. Les notions \
rendues par un seul signe sont reliées par un trait d'union (ex. "ÇA-VA", \
"TOUT-LE-MONDE", "SEMAINE-PROCHAINE").
2. ORDRE : [CONNECTEUR] -> TEMPS -> LIEU -> SUJET -> OBJET -> VERBE -> [MODAL] \
-> [ADVERBE] -> [NÉGATION] -> [INTERROGATIF]. Le SUJET vient AVANT l'objet \
(ex. "je ne comprends pas cette phrase" -> "MOI" "PHRASE" "COMPRENDRE" "NON"). \
Le LIEU (endroit explicite ou destination d'un déplacement) vient avant le sujet \
(ex. "au parc, nous promenons le chien" -> "PARC" "NOUS" "CHIEN" "PROMENER"). \
Un élément n'est mis en tête que si la phrase le met déjà en avant \
("le café, j'adore ça" -> "CAFÉ" "MOI" "AIMER" "BEAUCOUP").
3. SUPPRIME les mots-outils absents de la LSF : articles (le, la, les, un, une, \
des), prépositions et conjonctions (de, dans, à, pour, que, avec, en...), \
copule et auxiliaires ("être", "avoir"), "il y a". Ne les transforme JAMAIS en glose.
4. NÉGATION : "NON" APRÈS le verbe (et ses adverbes), avant un éventuel mot \
interrogatif. "ne... jamais" -> "JAMAIS", "ne... rien" -> "RIEN", \
"ne... personne" -> "PERSONNE" ; "pas du tout" -> "NON".
5. QUESTION : place le marqueur interrogatif EN FIN de séquence \
("QUOI", "OÙ", "QUI", "QUAND", "POURQUOI", "COMMENT", "COMBIEN" ; "que" -> \
"QUOI"). N'ajoute "QUESTION" QUE pour une question fermée (oui/non) SANS mot \
interrogatif — ne mets JAMAIS un mot interrogatif ET "QUESTION" ensemble. La \
négation d'une question reste présente (ex. "pourquoi tu ne viens pas ?" -> \
"TOI" "VENIR" "NON" "POURQUOI").
6. Les pronoms deviennent des pointages : je/moi -> "MOI", tu/toi -> "TOI", \
il/elle -> "LUI", nous/on -> "NOUS", vous -> "VOUS", ils/elles -> "EUX". \
Jamais "JE", "IL", "ELLE".
7. L'adjectif se place APRÈS le nom, au masculin singulier (ex. "voiture bleue" \
-> "VOITURE" "BLEU"). Le nombre et les quantités se placent APRÈS le nom \
(ex. "deux méthodes" -> "MÉTHODE" "DEUX"). Pluriel non marqué : nom au singulier.
8. Le verbe reste à l'INFINITIF, jamais conjugué (ex. "tu pars" -> "PARTIR" ; \
"il a acheté" -> "ACHETER"). Les modaux (vouloir, pouvoir, devoir) viennent \
APRÈS le verbe principal ("je veux partir" -> "MOI" "PARTIR" "VOULOIR"). \
Les adverbes de manière ou d'intensité viennent après le verbe \
("il cuisine bien" -> "LUI" "CUISINER" "BIEN").
9. TEMPS en tête, du plus général au plus précis. Un temps EXPLICITE garde son \
signe : "HIER", "DEMAIN", "AUJOURD'HUI", "MAINTENANT", "MATIN", "SOIR", \
"WEEK-END", "SEMAINE-PROCHAINE"... Sans temps explicite : passé -> "AVANT" en \
tête, futur (y compris "aller" + infinitif) -> "BIENTÔT" en tête. Jamais \
"AVANT"/"BIENTÔT" en plus d'un temps explicite. "aller" + infinitif n'est pas \
glosé "ALLER" ("je vais manger" -> "BIENTÔT" "MOI" "MANGER").
10. Les verbes directionnels incorporent leur destinataire : VERBE-À-<POINTAGE> \
(ex. "je te donne" -> "MOI" ... "DONNER-À-TOI" ; "je vous remercie" -> "MOI" \
"REMERCIER-À-VOUS"). Le sujet reste exprimé devant.
11. Possession : CHAQUE possessif devient un pointage APRÈS le nom \
(ex. "ma voiture" -> "VOITURE" "MOI" ; "ses projets" -> "PROJET" "LUI"). \
Complément du nom : nom principal puis complément ("la détection dans un flux" \
-> "DÉTECTION" "FLUX").
12. CONSERVE LES MOTS SOURCE TELS QUELS. Ne ré-orthographie pas, ne traduis pas, \
n'invente aucune graphie. Une glose issue d'un mot de la phrase est ce mot \
mis en MAJUSCULES, avec ses lettres et accents d'origine (verbes à l'infinitif). \
Les noms propres, termes techniques et mots étrangers sont repris À \
L'IDENTIQUE, jamais traduits ni francisés (ex. "forest" -> "FOREST", et NON \
"FORÊT"). En cas de doute sur un mot, recopie-le tel quel plutôt que de le deviner.

LEXIQUE IMPOSÉ (utilise EXACTEMENT ces gloses, sans les traduire autrement) :
- "en direct", "en live", "live" -> DIRECT
- "pour finir", "pour conclure", "en conclusion", "pour terminer" -> ENFIN, en \
tout premier
- "du coup", "donc", "alors" (connecteurs de discours) -> SUPPRIMÉS
- "adorer" -> AIMER BEAUCOUP ; "parler de" -> PARLER
- termes techniques ou anglais du domaine ("dataset", "fine tuning", \
"random forest", "machine learning") -> repris À L'IDENTIQUE en MAJUSCULES, \
mots reliés par des traits d'union ("FINE-TUNING", "RANDOM-FOREST").

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
    # Temps en tête + lieu (destination) avant le sujet + verbe final.
    ("Demain, je vais à Paris.", ["DEMAIN", "PARIS", "MOI", "ALLER"]),
    # Négation après le modal ; destination en tête ; modal après le verbe.
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
    # Sujet avant objet ; complément du nom ; "parler de" -> PARLER.
    ("Je parle de la détection dans un flux.",
     ["MOI", "DÉTECTION", "FLUX", "PARLER"]),
    # Verbe directionnel : le destinataire est incorporé au verbe.
    ("Je te donne le livre.", ["MOI", "LIVRE", "DONNER-À-TOI"]),
    # Topicalisation : la phrase met elle-même le thème en avant.
    ("Le café, j'adore ça.", ["CAFÉ", "MOI", "AIMER", "BEAUCOUP"]),
    # Futur proche : marqueur temporel, PAS d'auxiliaire "aller".
    ("Je vais manger.", ["BIENTÔT", "MOI", "MANGER"]),
    # Passé sans temps explicite : "AVANT" en tête ; nombre après le nom.
    ("J'ai utilisé deux méthodes.",
     ["AVANT", "MOI", "MÉTHODE", "DEUX", "UTILISER"]),
    # Connecteur "pour finir" -> ENFIN en premier ; futur -> BIENTÔT ; directionnel.
    ("Pour finir, je vais vous expliquer les résultats.",
     ["ENFIN", "BIENTÔT", "MOI", "RÉSULTAT", "EXPLIQUER-À-VOUS"]),
    # Lexique : "en direct" -> DIRECT, après le verbe (jamais "vivant"/"en-direct").
    ("Je suis l'émission en direct.", ["MOI", "ÉMISSION", "SUIVRE", "DIRECT"]),
    # Temps explicite au passé : le signe du temps est gardé (pas AVANT).
    ("Hier soir, nous avons regardé un film.",
     ["HIER", "SOIR", "NOUS", "FILM", "REGARDER"]),
    # Question ouverte + négation : mot interrogatif en fin, "NON" conservé,
    # PAS de "QUESTION" en plus ; verbe à l'infinitif.
    ("Pourquoi il ne répond pas ?", ["LUI", "RÉPONDRE", "NON", "POURQUOI"]),
    # Source anglaise : négation après le verbe.
    ("I don't understand.", ["MOI", "COMPRENDRE", "NON"]),
]


# Few-shot CIBLÉ (option `few_shot_targeted` / `eval/evaluate.py --few-shot-cible`) :
# exemples AJOUTÉS aux précédents, pris dans le split TRAIN de
# finetune/data/2184f8f3751e (jamais le dev ni l'éval), sur les phénomènes les
# moins bien rendus : directionnels (chaque destinataire), passé sans temps
# explicite (AVANT), ils -> EUX, possessifs, question fermée, « live », pluriel
# non marqué. Placés en dernier : les plus proches de la phrase à traiter.
FEW_SHOT_TARGETED: list[tuple[str, list[str]]] = [
    # Passé implicite + ils -> EUX + directionnel -> NOUS + possessif « leurs ».
    ("Ils nous ont montré leurs résultats.",
     ["AVANT", "EUX", "RÉSULTAT", "EUX", "MONTRER-À-NOUS"]),
    # Directionnel -> MOI + possessif + modal après le verbe + question fermée.
    ("Tu peux me prêter ton stylo ?",
     ["TOI", "STYLO", "TOI", "PRÊTER-À-MOI", "POUVOIR", "QUESTION"]),
    # Directionnel -> LUI (« lui » complément, pas sujet).
    ("Le médecin lui explique le traitement.",
     ["MÉDECIN", "TRAITEMENT", "EXPLIQUER-À-LUI"]),
    # Futur proche -> BIENTÔT (pas ALLER) + directionnel -> VOUS.
    ("Je vais vous montrer une démo.", ["BIENTÔT", "MOI", "DÉMO", "MONTRER-À-VOUS"]),
    # Passé implicite + nombre après le nom + pluriel non marqué.
    ("Nous avons testé trois modèles.", ["AVANT", "NOUS", "MODÈLE", "TROIS", "TESTER"]),
    # Passé implicite + possessif (deux TOI : sujet et possessif).
    ("Tu as oublié ton sac.", ["AVANT", "TOI", "SAC", "TOI", "OUBLIER"]),
    # Passé implicite dans une question fermée.
    ("Vous avez compris l'exercice ?",
     ["AVANT", "VOUS", "EXERCICE", "COMPRENDRE", "QUESTION"]),
    # « live » -> DIRECT ; on -> NOUS ; question fermée.
    ("On regarde le live ensemble ?",
     ["NOUS", "DIRECT", "REGARDER", "ENSEMBLE", "QUESTION"]),
    # ils -> EUX ; temps explicite en tête ; modal après le verbe.
    ("Ils doivent travailler samedi.", ["SAMEDI", "EUX", "TRAVAILLER", "DEVOIR"]),
    # Pluriel non marqué + adjectif au masculin singulier après le nom.
    ("Les nouveaux étudiants arrivent.", ["ÉTUDIANT", "NOUVEAU", "ARRIVER"]),
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


def build_messages(text: str, language: str | None = None,
                   targeted: bool = False) -> list[dict]:
    """Assemble la liste de messages complète pour l'appel au LLM.

    Structure : [système] + [paires user/assistant d'exemples] + [phrase réelle].
    Fournir les exemples comme de vrais tours de dialogue (plutôt que dans le
    prompt système) ancre fortement le format de sortie JSON et le style des
    gloses — c'est la bonne pratique de few-shot avec une API de chat.

    `targeted` ajoute FEW_SHOT_TARGETED après les exemples de base.
    """
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    examples = FEW_SHOT_EXAMPLES + (FEW_SHOT_TARGETED if targeted else [])
    for source, glosses in examples:
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
    "FEW_SHOT_TARGETED",
    "build_user_prompt",
    "build_messages",
    "GLOSS_JSON_SCHEMA",
]
