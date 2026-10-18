# Évaluation du traducteur LSF

Harnais pour mesurer la qualité des gloses produites par `lsf_translator` et
**comparer objectivement** un changement de prompt ou de modèle (plutôt que de
juger « à l'œil » sur quelques phrases).

```
eval/
├── dataset.jsonl   # paires français -> glose de référence (gold)
├── evaluate.py     # scoring : appelle Ollama avec le prompt de PRODUCTION
├── compare.py      # diff entre deux runs archivés
├── RESULTS.md      # tableau de bord (une ligne par run --save)
├── results/        # records JSON immuables (un par run --save)
└── README.md       # ce fichier
```

## Lancer

Ollama doit tourner (`ollama serve`) et le modèle être installé.

```bash
# Modèle par défaut (config.py)
uv run python eval/evaluate.py

# Comparer un autre modèle
uv run python eval/evaluate.py --model qwen2.5:7b

# Agrégats seulement (sans le détail par phrase)
uv run python eval/evaluate.py --quiet
```

Le script utilise **exactement** `build_messages()` + `GLOSS_JSON_SCHEMA` de
`lsf_translator/prompts.py` : ce qu'il mesure est donc bien le comportement réel
du pipeline, pas un prompt approché.

## Métriques

| Mesure | Ce qu'elle capte |
| --- | --- |
| **exact** | séquence de gloses identique à la référence (ordre compris). Sévère. |
| **F1** | recouvrement lexical (multiset) : bon **choix** des gloses, indépendamment de l'ordre. Détecte les gloses en trop (mots-outils) ou manquantes. |
| **ordre** | plus longue sous-séquence commune / longueur de la référence : respect de l'**ordre**. |
| **fuites** | nb de phrases contenant une glose « mot-outil » interdite (`DE`, `DANS`, `POUR`, `LE`…), qui ne devrait jamais apparaître en LSF. |

Une **ventilation par phénomène** (étiquettes du dataset) indique quelle classe
de règle régresse ou progresse (ex. `negation`, `connecteur`, `lexique-direct`).

> ⚠️ `exact` est volontairement sévère : une phrase correcte mais dans un ordre
> LSF alternatif (voir ci-dessous) comptera comme non-exact. Regarde **F1** et
> la ventilation pour le détail, pas seulement `exact`.

## Suivi des résultats dans le temps

Pour comparer proprement un changement (amélioration du prompt **ou** modèle
fine-tuné), archive chaque run avec `--save` :

```bash
uv run python eval/evaluate.py --save --label "ajout lexique direct/enfin"
```

Cela produit **deux** choses :

1. **Un record immuable** `eval/results/<date>__<modèle>__<githash>.json` — il
   pince les **trois variables** qui déterminent un résultat :

   | Variable | Champ(s) du record |
   | --- | --- |
   | Modèle | `model`, `model_digest` (distingue deux versions d'un même tag) |
   | Prompt | `prompt_hash` (système + few-shot), `git_commit`, `git_dirty` |
   | Dataset | `dataset`, `dataset_sha`, `dataset_n` |

   Il contient aussi `metrics`, `by_phenomenon`, `prompt_tokens` (alerte si
   `> num_ctx`), et le **détail par phrase** (`per_sentence`) qui permet le diff.

2. **Une ligne dans `eval/RESULTS.md`** — tableau de bord lisible, versionné
   dans git : une ligne par run, avec un lien vers le record. Chaque
   amélioration devient ainsi un **diff visible** dans l'historique.

> `git*` (étoile) dans le tableau = le run a été lancé avec des modifications
> non commitées. Commite le prompt **et** le record ensemble pour garder le lien
> « cette version = ces résultats ».

### Comparer deux runs

```bash
uv run python eval/compare.py <runA.json> <runB.json>   # A = avant, B = après
# (un simple nom de fichier est résolu sous eval/results/)
```

Affiche les **deltas** de métriques, les phénomènes qui ont bougé, et surtout
les phrases dont la sortie a changé, classées **✅ corrigée / ❌ régressée /
↑ améliorée / ↓ dégradée / ~ modifiée**. Le script **alerte** si les deux runs
n'ont pas le même `dataset_sha` (comparaison non fiable) ou si prompt **et**
modèle sont identiques (écarts dus au seul non-déterminisme).

### Workflow type

```bash
# 1. modifie le prompt (lsf_translator/prompts.py) ou installe un modèle fine-tuné
# 2. mesure + archive
uv run python eval/evaluate.py --save --label "ce que j'ai changé"
# 3. compare au run précédent
uv run python eval/compare.py <ancien> <nouveau>
# 4. commite prompt + record + tableau ensemble
git add eval/ lsf_translator/prompts.py
git commit -m "prompt: ... (F1 0.87 -> 0.96)"
```

> **Fine-tune :** un modèle fine-tuné n'est qu'un autre `--model lsf-gemma2:v1`
> dans Ollama — le même harnais le mesure et le `model_digest` le distingue.
> Versionne explicitement les tags (`v1`, `v2`…), ne réécris jamais un tag.

## Format du dataset (`dataset.jsonl`)

Un objet JSON par ligne :

```json
{"text": "Hier, j'ai mangé une pomme.", "language": "fr",
 "glosses": ["AVANT", "MOI", "POMME", "MANGER"],
 "phenomena": ["temps-passe", "verbe-final"]}
```

- `text` : phrase source (fr/en).
- `glosses` : glose **de référence** attendue (en MAJUSCULES).
- `phenomena` : étiquettes libres servant à la ventilation (choisis des noms
  cohérents d'une ligne à l'autre).

## Conventions de glose retenues

Ce sont les règles appliquées **à la fois** dans le prompt (`prompts.py`) et dans
le gold de ce dataset, pour que la mesure soit cohérente :

1. Gloses en MAJUSCULES ; une notion = un signe, les composés reliés par
   trait d'union (`CA-VA`, `DONNER-À-TOI`).
2. Ordre `TEMPS → LIEU → SUJET → OBJET → VERBE` ; verbe en fin.
3. Suppression des mots-outils (articles, prépositions, conjonctions).
4. Négation `NON` **après** le verbe.
5. Question : mot interrogatif **en fin** ; `QUESTION` uniquement pour une
   question fermée **sans** mot interrogatif (jamais les deux).
6. Verbe à l'**infinitif** ; temporalité portée par une glose de temps
   (`AVANT`, `BIENTÔT`…), pas par un auxiliaire.
7. Adjectif et nombre **après** le nom ; pluriel non marqué.
8. Verbes directionnels incorporés (`EXPLIQUER-À-VOUS`) ; possession par
   pointage (`VOITURE MOI`).
9. Lexique imposé : `en direct/en live → DIRECT` ; `pour finir/pour conclure/
   en conclusion → ENFIN` ; termes techniques/anglais (`DATASET`, `FINE-TUNING`,
   `CAMEMBERT`) **repris à l'identique**.

## Limites à connaître (important)

- **Ces gloses de référence ne sont PAS validées par un locuteur LSF.** Elles
  reflètent une interprétation des règles ci-dessus. Elles doivent être
  **relues/corrigées par une personne compétente en LSF** avant de servir de
  vérité terrain fiable.
- **La LSF admet plusieurs ordres valides** (structure topic-commentaire, OSV
  fréquent). La métrique `ordre` peut donc pénaliser une sortie *correcte* qui
  ne suit pas l'ordre figé du gold. Ne sur-optimise pas `ordre` sans un gold
  validé.
- **Petit échantillon** (16 phrases) : utile pour détecter de grosses
  régressions, insuffisant pour des conclusions fines. Étendre à ~40-50 phrases
  variées améliorerait la stabilité des mesures.

## Étendre le dataset

Ajoute des lignes à `dataset.jsonl` (idéalement validées LSF). Pour éviter de
fausser la mesure, **n'y mets pas** les phrases déjà présentes comme exemples
few-shot dans `lsf_translator/prompts.py` (`FEW_SHOT_EXAMPLES`) : le modèle les a
déjà vues, elles gonfleraient artificiellement le score.
