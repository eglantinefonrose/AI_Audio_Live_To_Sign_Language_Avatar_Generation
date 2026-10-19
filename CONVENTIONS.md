# Conventions de glose LSF du projet

Référence **unique** appliquée partout : prompt (`lsf_translator/prompts.py`),
gold de l'éval (`eval/dataset.jsonl`), datasets d'entraînement
(`finetune/data/<hash>/`). Les règles vérifiables automatiquement sont contrôlées
par `lsf_translator/conventions.py` (`uv run python -m lsf_translator.conventions`).

> ⚠️ Ces conventions sont des **choix de projet** pour obtenir des données
> cohérentes et mesurables. Elles ne sont **pas validées par un locuteur LSF** ;
> la LSF admet d'autres ordres et d'autres signes. Si une personne compétente
> les révise, on change ce fichier, puis le gold et les datasets ensemble.

## 1. Forme des gloses

- Une glose = un signe, en **MAJUSCULES**, accents conservés (`ÉTUDIANT`, `ÇA-VA`).
- Plusieurs mots pour un seul signe → **traits d'union** (`ÇA-VA`, `TOUT-LE-MONDE`,
  `SEMAINE-PROCHAINE`, `D'ACCORD`).
- Mot de la phrase repris tel quel, mis en majuscules (pas de ré-orthographe).

## 2. Ordre canonique

```
[CONNECTEUR] → TEMPS → LIEU → SUJET → OBJET/COMPLÉMENT → VERBE → [MODAL] → [ADVERBE] → [NÉGATION] → [INTERROGATIF]
```

- **Sujet avant objet** (`MOI PHRASE COMPRENDRE NON`, `NOUS MATCH REGARDER DIRECT`).
- **Lieu** = endroit explicite (« à Paris », « au bureau », « chez moi », « dans la cuisine »),
  y compris la destination d'un verbe de déplacement (« aller à la réunion ») :
  il vient **avant le sujet** (`PARC NOUS CHIEN PROMENER`).
- **Topicalisation** seulement si la phrase source met elle-même un élément en
  avant (dislocation : « Le café, j'adore ça » → `CAFÉ MOI AIMER BEAUCOUP`).
- Sans sujet exprimé (impersonnel, passif) : le thème vient en tête
  (« Il pleut » → `PLEUVOIR` ; « Le concert est diffusé en live » →
  `CONCERT DIFFUSER DIRECT`).

## 3. Mots supprimés

- Articles, prépositions, conjonctions : `LE`, `DE`, `À`, `POUR`, `ET`, `QUE`…
  ne sont **jamais** des gloses.
- Copule et auxiliaires : `ÊTRE`/`AVOIR` ne sont pas glosés (« Ma voiture est
  bleue » → `VOITURE MOI BLEU` ; « J'ai deux chats » → `MOI CHAT DEUX`).
- Connecteurs de discours « du coup », « donc », « alors » : supprimés.
- « il y a » (existence) : supprimé (« À Paris, il y a beaucoup de touristes » →
  `PARIS TOURISTE BEAUCOUP`).

## 4. Pronoms et possession

- Pointages : je/moi → `MOI`, tu/toi → `TOI`, il/elle → `LUI`, nous → `NOUS`,
  vous → `VOUS`, ils/elles → `EUX`. Jamais `JE`, `TU`, `IL`, `ELLE`, `ON`.
- « on » = nous → `NOUS`.
- Possessif → pointage **après le nom** : « mon frère » → `FRÈRE MOI`,
  « ses projets » → `PROJET LUI`, « votre attention » → `ATTENTION VOUS`.
  Jamais `MON`, `SES`, `VOTRE`…
- Complément du nom : nom principal puis complément (« la détection dans un
  flux » → `DÉTECTION FLUX` ; « les résultats du modèle » → `RÉSULTAT MODÈLE`).

## 5. Verbes

- Toujours à l'**infinitif** (`ACHETER`, `PARTIR`), jamais conjugués.
- Participe employé comme **adjectif** : forme adjectivale masculin singulier
  (« modèles pré-entraînés » → `MODÈLE PRÉ-ENTRAÎNÉ`, « magasin ouvert » →
  `MAGASIN OUVERT`).
- **Modaux** (vouloir, pouvoir, devoir) **après** le verbe principal
  (« je ne veux pas partir » → `MOI PARTIR VOULOIR NON`). Seul, le modal est le
  verbe (« tu veux un café ? » → `TOI CAFÉ VOULOIR QUESTION`).
- « aller » + infinitif (futur proche) → `BIENTÔT` + verbe, pas de `ALLER`.
  « aller » de déplacement → `ALLER`.
- « parler de » → `PARLER` ; « expliquer » → `EXPLIQUER`.
- Expressions figées : « adorer » → `AIMER BEAUCOUP`, « détester » →
  `AIMER NON`, « avoir faim » → `FAIM`.

## 6. Verbes directionnels

- Le destinataire est incorporé : `VERBE-À-<PRONOM>` (`DONNER-À-TOI`,
  `EXPLIQUER-À-VOUS`, `REMERCIER-À-VOUS`, `MONTRER-À-MOI`).
- Le sujet reste exprimé devant : « je vous remercie » → `MOI REMERCIER-À-VOUS` ;
  « il me donne le livre » → `LUI LIVRE DONNER-À-MOI`.
- Jamais `REMERCIER-VOUS` (sans `À`) ni le pronom destinataire séparé.

## 7. Temps

- Le temps vient **en tête** (après un éventuel connecteur), du plus général au
  plus précis (`DEMAIN MATIN`, `HIER SOIR`).
- Un **temps explicite garde son signe** : `HIER`, `DEMAIN`, `AUJOURD'HUI`,
  `MAINTENANT`, `MATIN`, `SOIR`, `WEEK-END`, `SEMAINE-PROCHAINE`,
  `SEMAINE-DERNIÈRE`, `ANNÉE-PROCHAINE`…
- Passé **sans** indication de temps → `AVANT` en tête
  (« j'ai utilisé deux méthodes » → `AVANT MOI MÉTHODE DEUX UTILISER`).
- Futur **sans** indication de temps → `BIENTÔT` en tête.
- Jamais `AVANT` ou `BIENTÔT` en plus d'un temps explicite.
- « tout à l'heure » → `AVANT` (passé) ou `BIENTÔT` (futur) selon le temps du verbe.
- Jour + « dernier/prochain » → un seul signe (`LUNDI-DERNIER`, `MARDI-PROCHAIN`).
- Durée « dans trois semaines » → `SEMAINE TROIS` en tête (temps explicite).
- « déjà » est un adverbe après le verbe (`VOIR DÉJÀ`) et dispense de `AVANT`.

## 8. Négation

- « ne … pas », « ne … plus », « don't/not » → `NON`, **après le verbe** (et ses
  adverbes), avant un éventuel interrogatif.
- « ne … jamais » → `JAMAIS` ; « ne … rien » → `RIEN` ; « ne … personne » →
  `PERSONNE` (même position que `NON`).
- « pas du tout » → `NON` (l'intensif n'est pas glosé à part).

## 9. Questions

- Question ouverte : mot interrogatif **en dernier** (`QUOI`, `OÙ`, `QUI`, `QUAND`,
  `POURQUOI`, `COMMENT`, `COMBIEN`) ; « que » interrogatif → `QUOI`.
- Question fermée (oui/non) : `QUESTION` en dernier.
- **Jamais** un interrogatif **et** `QUESTION` ensemble.
- La négation reste avant l'interrogatif (`DEMAIN TOI VENIR NON POURQUOI`).
- Le marqueur est le `QUESTION` **final** d'une phrase interrogative ; ailleurs,
  `QUESTION` est le nom « question » (« je n'ai pas compris la question » →
  `AVANT MOI QUESTION COMPRENDRE NON`).

## 10. Noms, adjectifs, nombres

- Adjectif **après** le nom, au masculin singulier (`VOITURE ROUGE`,
  `ÉTUDIANT ÉTRANGER`).
- Nombre **après** le nom, en lettres (`MÉTHODE DEUX`, `HEURE HUIT`).
- Nom + adjectif + nombre : `NOM ADJECTIF NOMBRE` (« trois livres anciens » →
  `LIVRE ANCIEN TROIS`).
- Intensif d'un adjectif après lui (« très claire » → `CLAIR TRÈS`,
  « trop petit » → `PETIT TROP`).
- Pluriel non marqué : nom au singulier (`ÉLÈVE`, `RÉSULTAT`).
- Quantifieurs après le nom (`TOURISTE BEAUCOUP`, `HEURE PLUSIEURS`).
- Adverbes de manière / intensité après le verbe (`CUISINER BIEN`,
  `TRAVAILLER BEAUCOUP`, `REGARDER DIRECT`).

## 11. Phrases en anglais

Mêmes règles ; les gloses sont les **signes LSF en français** (« I don't
understand » → `MOI COMPRENDRE NON`), sauf noms propres et termes techniques
repris tels quels.

## 12. Lexique imposé

| Source | Glose |
| --- | --- |
| « en direct », « en live », « live » | `DIRECT` |
| « pour finir », « pour conclure », « en conclusion », « pour terminer » | `ENFIN` (en tout premier) |
| « du coup », « donc », « alors » (connecteurs) | supprimés |
| termes techniques / anglais (« dataset », « fine tuning », « random forest », « machine learning », « deep learning ») | repris à l'identique, mots reliés par des traits d'union (`DATASET`, `FINE-TUNING`, `RANDOM-FOREST`) |
| noms propres (« Paris », « Camembert ») | repris à l'identique (`PARIS`, `CAMEMBERT`) |
