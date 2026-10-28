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
├── datasets/       # copies figées des datasets évalués : <dataset_sha>/dataset.jsonl
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

Cela produit **trois** choses :

1. **Un record immuable** `eval/results/<date>__<modèle>__<githash>.json` — il
   pince les **trois variables** qui déterminent un résultat :

   | Variable | Champ(s) du record |
   | --- | --- |
   | Modèle | `model`, `model_digest` (distingue deux versions d'un même tag) |
   | Prompt | `prompt_hash` (système + few-shot), `git_commit`, `git_dirty` |
   | Dataset | `dataset`, `dataset_sha`, `dataset_snapshot`, `dataset_n` |

   Il contient aussi `metrics`, `by_phenomenon`, `prompt_tokens` (alerte si
   `> num_ctx`), et le **détail par phrase** (`per_sentence`) qui permet le diff.

2. **Une copie figée du dataset** `eval/datasets/<dataset_sha>/dataset.jsonl`
   (lecture seule, jamais réécrite ; créée au premier run sur ce contenu).
   `dataset.jsonl` peut ainsi évoluer librement : chaque record pointe
   (`dataset_snapshot`) vers le contenu exact sur lequel il a été mesuré. Si une
   copie existante ne correspond plus à son hash, `--save` s'arrête.

3. **Une ligne dans `eval/RESULTS.md`** — tableau de bord lisible, versionné
   dans git : une ligne par run, avec un lien vers le record. Deux colonnes
   de dataset à ne pas confondre : **Dataset éval** = hash du jeu de test
   (`dataset_sha`, renseigné pour **tous** les runs, lien vers sa copie figée) ;
   **Dataset train** = hash du `train.jsonl` d'entraînement (`training.train_sha`),
   renseigné **uniquement pour un modèle fine-tuné**, « — » pour un modèle de base,
   avec un lien vers son dossier `finetune/data/<train_sha>/`. Chaque
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
> Si le tag a été importé via `finetune/register_ollama.py`, le record contient
> aussi un champ `training` (date, durée, hyperparamètres, val loss) et la ligne
> de `RESULTS.md` affiche la date d'entraînement à côté du modèle.

### Post-traitement déterministe

`--postprocess` applique aux gloses du modèle les corrections de
`lsf_translator/postprocess.py` : règles de CONVENTIONS.md vérifiables à partir
de la phrase source (temps AVANT/BIENTÔT, QUESTION / interrogatif en fin,
négation après le verbe, ils/elles -> EUX, verbes directionnels, possessifs,
lexique imposé, orthographe des mots source, gloses inventées). Le modèle et le
prompt ne changent pas. La colonne « Méthode » de `RESULTS.md` affiche
`post-traitement (<hash>)` : le hash change dès qu'une règle ou le lexique
change, comme `prompt_hash` pour le prompt. Le record garde, par phrase, la
sortie brute du modèle (`raw`) et les règles appliquées (`rules`). La prod
utilise le même code (`config.py: OllamaConfig.postprocess`, `cli.py --postprocess`).

**Mise au point sur le jeu de DEV, jamais sur l'éval** : `eval/dev.jsonl` = le
split valid de `finetune/data/2184f8f3751e` (45 phrases, ni train ni éval).
`--out` écrit le record sans l'ajouter à `RESULTS.md` :

```bash
uv run python eval/evaluate.py --model lsf-gemma9b:v1 --postprocess \
  --dataset eval/dev.jsonl --out /tmp/dev_pp.json
```

Garde-fou sans LLM : `python -m lsf_translator.postprocess` applique les règles
aux gloses de RÉFÉRENCE (train + dev) et doit n'en modifier aucune. Le lexique
(`lsf_translator/lexicon.txt`) vient du seul split train :
`python -m lsf_translator.lexicon finetune/data/<sha>`.

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

## Conventions de glose

Le gold suit **[CONVENTIONS.md](../../CONVENTIONS.md)**, la référence unique partagée
avec le prompt (`prompts.py`) et les datasets d'entraînement. Avant de modifier ou
d'ajouter une phrase, vérifie-la :

```bash
uv run python -m lsf_translator.conventions          # gold + exemples du prompt
```

Le vérificateur contrôle ce qui est mécanique (forme, mots-outils, pronoms,
possessifs, négation, questions, lexique imposé, temps, directionnels) ; l'ordre
sujet/objet et le choix des signes restent à relire.

## Limites à connaître (important)

- **Ces gloses de référence ne sont PAS validées par un locuteur LSF.** Elles
  reflètent une interprétation des règles ci-dessus. Elles doivent être
  **relues/corrigées par une personne compétente en LSF** avant de servir de
  vérité terrain fiable.
- **La LSF admet plusieurs ordres valides** (structure topic-commentaire, OSV
  fréquent). La métrique `ordre` peut donc pénaliser une sortie *correcte* qui
  ne suit pas l'ordre figé du gold. Ne sur-optimise pas `ordre` sans un gold
  validé.
- **Échantillon encore modeste** (74 phrases, une phrase ≈ 1,4 point
  d'« exact ») : suffisant pour voir un écart de quelques points de F1, pas pour
  départager deux runs à 1-2 points près. Regarde aussi la ventilation par
  phénomène (3 à 15 phrases chacun).
- Les runs antérieurs au dataset `01f21c7fbf2d` (74 phrases) ont été mesurés sur
  l'ancien gold de 16 phrases (`3db947280fd3`) : **ne compare que des runs de
  même « Dataset éval »** (`compare.py` alerte sinon).

## Étendre le dataset

Ajoute des lignes à `dataset.jsonl` (idéalement validées LSF). Pour éviter de
fausser la mesure, **n'y mets pas** les phrases déjà présentes comme exemples
few-shot dans `lsf_translator/prompts.py` (`FEW_SHOT_EXAMPLES`) : le modèle les a
déjà vues, elles gonfleraient artificiellement le score. De même, garde-les **hors** des datasets
d'entraînement : `build_dataset.py` rejette toute paire qui recouvre l'éval.
