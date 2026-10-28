# Fine-tuning FR → glose LSF (LoRA sur Apple Silicon)

Recette pour fine-tuner un **petit modèle élève** sur le dataset synthétique
produit par `build_dataset.py`, avec **MLX-LM** (natif M1), puis l'évaluer avec
le harnais `eval/` et le servir via Ollama.

> ⚠️ Prérequis de bon sens
> - On fine-tune les **poids HuggingFace**, pas la copie Ollama.
> - Sur **M1 Pro 16 Go**, prends un **élève 3B** (pas 9B) + base **4-bit**.
> - L'**élève doit être ≤ le professeur** qui a généré les données, sinon aucun
>   gain. Choisis le professeur en le mesurant : `eval/evaluate.py --model …`.
> - Dataset **synthétique non validé** → run **expérimental** (monte la chaîne).

---

## Qu'est-ce que LoRA ? (Low-Rank Adaptation)

**Le problème.** Un LLM, c'est des milliards de poids. Les ré-entraîner **tous**
(« full fine-tuning ») demande de garder en mémoire les poids + leurs gradients +
les états de l'optimiseur : plusieurs fois la taille du modèle. Impensable sur
16 Go pour autre chose qu'un tout petit modèle.

**L'idée de LoRA.** On **gèle** les poids d'origine (on n'y touche pas) et, à côté
de certaines matrices du modèle, on ajoute **deux petites matrices** `A` et `B`
qu'on est seul à entraîner. Mathématiquement, au lieu d'apprendre une grosse
correction `ΔW` (de la taille du poids), on l'approxime par un produit de deux
matrices fines : `ΔW ≈ B · A`. Ces matrices sont de **rang faible** (« low-rank »,
d'où le nom) — quelques colonnes seulement — donc **minuscules** : on entraîne
~0,1–1 % des paramètres au lieu de 100 %.

```
          poids figé W  (des milliards de valeurs, NON entraînés)
                │
   entrée ──────┼───────────────► sortie
                │                    ▲
                └──►  A ──► B  ──────┘   ← LoRA : A et B, petites, SEULES entraînées
                   (rank r)
```

**Ce que ça change concrètement ici :**

- **Tient en 16 Go.** On n'optimise que `A`/`B` → empreinte mémoire réduite.
  C'est ce qui rend l'entraînement possible sur ton M1 Pro.
- **Sortie légère : l'« adaptateur ».** Le résultat de l'entraînement n'est pas
  un nouveau modèle de 6 Go, mais juste ces petites matrices → `finetune/adapters/`
  (**~8 Mo**). C'est pour ça qu'on publie l'adaptateur sur HF, pas les poids entiers.
- **Réversible / composable.** L'adaptateur se **branche** sur le modèle de base
  au moment de l'inférence (`--adapter-path`), ou se **fusionne** dedans une fois
  pour toutes (`mlx_lm.fuse`, étape 4) pour servir via Ollama. Le modèle de base
  reste intact ; tu peux entraîner plusieurs adaptateurs pour plusieurs tâches.
- **QLoRA = LoRA sur base quantifiée.** Ici la base est déjà en **4-bit**
  (`...-4bit`) : les poids figés prennent 4× moins de place, et seules les
  matrices LoRA sont en pleine précision. C'est la variante qui fait tenir un 3B
  confortablement en 16 Go.

**Les hyperparamètres LoRA de la commande (section 2) :**

| Flag | Rôle |
| --- | --- |
| `--num-layers 8` | nombre de couches (en partant de la fin) où l'on greffe des matrices LoRA. Plus = plus de capacité d'apprentissage, mais plus de mémoire et plus de risque de surapprentissage sur un petit dataset. |
| `--iters` | nombre de pas d'entraînement. Trop → overfitting (la *val loss* remonte). |
| `--learning-rate` | pas d'apprentissage des matrices `A`/`B`. |
| `--batch-size 1` | nb d'exemples par pas ; 1 pour ménager la mémoire. |
| `--mask-prompt` | la perte ne porte que sur la **réponse** (les gloses), pas sur le prompt (règles + phrase). |

> En clair : LoRA n'apprend pas un nouveau modèle, il apprend un **petit correctif
> greffé** sur un modèle figé. D'où : rapide, léger (~8 Mo), peu gourmand en
> mémoire — au prix d'une capacité d'adaptation moindre qu'un full fine-tuning
> (largement suffisante pour spécialiser le style/format des gloses).

---

## 0. Construire les données (si pas déjà fait)

```bash
# professeur = le meilleur modèle mesuré via eval/evaluate.py
uv run python finetune/build_dataset.py --teacher qwen2.5:14b --per-category 40
# produit finetune/data/<train_sha>/{train.jsonl, valid.jsonl, pairs.jsonl, manifest.json}
```

Chaque dataset est **rangé par hash** : `finetune/data/<train_sha>/` (hash de
`train.jsonl`, celui de la colonne « Dataset train » de `eval/RESULTS.md`), en
lecture seule, jamais réécrit. Une nouvelle génération crée un nouveau dossier ;
les anciennes versions restent intactes.

```
finetune/data/
├── fa6131f5e0ee/   # v1 (professeur gemma2:9b, 20 train)
└── e604d1cd6448/   # v2 (Claude Opus, 92 train)
```

MLX-LM lit automatiquement `train.jsonl` et `valid.jsonl` dans `--data <dir>`,
au format chat `{"messages":[…]}` — c'est exactement ce que le builder écrit.

## 1. Installer MLX-LM

```bash
uv pip install mlx-lm        # outil d'entraînement (dépendance de dev, pas runtime)
```

> Les scripts `mlx_lm.*` s'exécutent dans l'environnement du projet : préfixe-les
> par **`uv run`** (sinon `zsh: command not found: mlx_lm.lora`).

## 2. Lancer l'entraînement LoRA

L'élève proposé est **ungated** et déjà en 4-bit (pas de téléchargement gated,
pas d'étape de quantification). Passe par le wrapper, qui horodate le run :

```bash
./finetune/train.sh                          # défauts : dataset le plus récent, iters=300, num_layers=8, lr=1e-4
DATA=finetune/data/fa6131f5e0ee ./finetune/train.sh   # un dataset précis
ITERS=200 NUM_LAYERS=4 ./finetune/train.sh   # surcharge des hyperparamètres
```

Sans `DATA`, il prend le **dataset le plus récent** (`created` du manifeste). Il
vérifie que le dossier est intact (`snapshot_data.py`) ; un dossier hors de
`finetune/data/` est d'abord copié et rangé sous son hash. L'entraînement lit
toujours la version rangée : chaque modèle garde le lien (`train_snapshot`) vers
les données exactes qui l'ont produit.

Il écrit ensuite un log dans `finetune/logs/` et surtout **`finetune/adapters/training_meta.json`** :
date de début/fin, durée, hyperparamètres, sha du train, manifeste du dataset,
commit git, dernière val loss. Ce fichier suit ensuite le modèle (Ollama → éval → HF).

Équivalent brut (sans provenance) :

```bash
uv run mlx_lm.lora \
  --model mlx-community/Qwen2.5-3B-Instruct-4bit \
  --train \
  --data finetune/data/<train_sha> \
  --adapter-path finetune/adapters \
  --mask-prompt \
  --batch-size 1 \
  --num-layers 8 \
  --iters 400 \
  --learning-rate 1e-4 \
  --steps-per-eval 50
```

`--mask-prompt` : la perte ne porte que sur la **réponse** (les gloses), pas sur
le system+user — le modèle apprend à *produire* la glose sachant les règles en
contexte, pas à recopier le prompt (cf. section « règles » du README racine).

Hyperparamètres adaptés à **16 Go + petit dataset** (quelques centaines
d'exemples). `mlx_lm.lora --help` pour les options exactes de ta version.

> **Surveille l'overfitting** : avec peu de données, la *validation loss* (sur
> `valid.jsonl`) finit par remonter alors que la *train loss* descend. Arrête-toi
> avant (baisse `--iters`). Règle d'or : plus le dataset est petit, moins
> d'itérations et de `--num-layers`.

## 3. Tester vite fait l'adaptateur

```bash
uv run mlx_lm.generate \
  --model mlx-community/Qwen2.5-3B-Instruct-4bit \
  --adapter-path finetune/adapters \
  --prompt "Phrase à transcrire en gloses LSF : \"Demain je vais à Paris.\""
```

## 4. Fusionner + importer dans Ollama

⚠️ **Pas de GGUF pour qwen2** : `mlx_lm.fuse --export-gguf` échoue avec
`Model type qwen2 not supported for GGUF conversion`. On importe donc les
**safetensors** directement dans Ollama (qui gère l'architecture qwen2). Pour
cela il faut des poids **fp16** standard, d'où `--de-quantize` (le modèle de
base est en 4-bit MLX) :

```bash
# fusionne l'adaptateur dans les poids, repasse en fp16 (safetensors standard)
uv run mlx_lm.fuse \
  --model mlx-community/Qwen2.5-3B-Instruct-4bit \
  --adapter-path finetune/adapters \
  --save-path finetune/fused \
  --de-quantize          # -> finetune/fused/{model*.safetensors, config.json, tokenizer*}
```

> `--de-quantize` ne *restaure* pas la précision perdue par la quantification
> 4-bit ; il remet simplement les poids au format fp16 que Ollama sait lire.
> Compte ~6 Go dans `finetune/fused/`.

Le Modelfile (`finetune/Modelfile`, déjà généré) pointe sur le **dossier**
`./fused` : Ollama détecte l'architecture et convertit tout seul. Son `SYSTEM`
est le prompt **règles-seules** utilisé à l'entraînement (`TRAIN_SYSTEM`),
**sans** few-shot — le prompt d'inférence doit correspondre au prompt
d'entraînement.

```dockerfile
# finetune/Modelfile (extrait)
FROM ./fused
PARAMETER temperature 0
SYSTEM """<les règles LSF + lexique, SANS la mention d'exemples>"""
```

```bash
uv run python finetune/register_ollama.py lsf-qwen3b:v3
```

= `ollama create lsf-qwen3b:v3 -f finetune/Modelfile` **+** enregistrement de la
provenance dans `finetune/models/<tag>.json` (tag, digest Ollama, copie de
`training_meta.json`). Ollama ne conserve aucune métadonnée : c'est ce fichier
(versionné) qui relie un tag à son entraînement. `eval/evaluate.py` le lit pour
inscrire la date/durée d'entraînement dans le record d'éval (et alerte si le
digest ne correspond plus), et `publish_hf.py` les reprend dans la model card.
Le script refuse de réécrire un tag existant.

> Variante GGUF (si tu y tiens) : fuse avec `--de-quantize` seul, puis convertis
> `finetune/fused/` en GGUF avec `llama.cpp` (`convert_hf_to_gguf.py`). Plus long,
> inutile ici puisque Ollama lit les safetensors directement.

## 5. Évaluer l'élève (boucle fermée avec `eval/`)

```bash
uv run python eval/evaluate.py --model lsf-qwen3b:v1 --save --label "finetune v1 (qwen3b)"
uv run python eval/compare.py <record_prof_ou_prompt> <record_finetune_v1>
```

> ⚠️ **Alignement du prompt** : `evaluate.py` envoie aujourd'hui le prompt de
> prod **avec** few-shot (`build_messages`). Pour évaluer *correctement* un
> modèle entraîné **sans** few-shot, il faut une variante **règles-seules**
> (system = `TRAIN_SYSTEM`, pas d'exemples). Sinon tu mesures un modèle sur un
> prompt qu'il n'a jamais vu → sous-évaluation. Voir la tâche « option
> rules-only » côté `prompts.py`/`evaluate.py`.

## Variante 7B (même recette, élève plus gros)

Mêmes données et hyperparamètres que la v2 ; seuls la base et les dossiers changent
(`adapters-*/` et `fused-*/` sont ignorés par git). Sur 16 Go, active le gradient
checkpointing et quantifie à l'import (le fused fp16 pèse ~15 Go).

```bash
MODEL=mlx-community/Qwen2.5-7B-Instruct-4bit ADAPTERS=finetune/adapters-7b \
  GRAD_CHECKPOINT=1 ./finetune/train.sh
uv run mlx_lm.fuse --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --adapter-path finetune/adapters-7b --save-path finetune/fused-7b --de-quantize
sed 's#^FROM ./fused$#FROM ./fused-7b#' finetune/Modelfile > finetune/Modelfile.7b
uv run python finetune/register_ollama.py lsf-qwen7b:v1 --modelfile finetune/Modelfile.7b \
  --meta finetune/adapters-7b/training_meta.json --quantize q4_K_M
rm -r finetune/fused-7b          # régénérable ; libère ~15 Go une fois importé
uv run python eval/evaluate.py --model lsf-qwen7b:v1 --prompt rules-only --save \
  --label "fine-tune qwen7b v1 — règles seules"
```

Pour la model card HF, `publish_hf.py` lit la base dans `training_meta.json`
(passe `--adapters finetune/adapters-7b`).

> ⚠️ La fusion `--de-quantize` d'un 7B matérialise ~15 Go de poids fp16 : sur
> 16 Go de RAM, attends-toi à du swap. Ferme les autres applications.

## Variante Gemma 2 (9B)

Le chat template de Gemma 2 n'a **pas de rôle `system`** (« System role not
supported » dès le premier exemple). On dérive donc une variante du dataset où le
system est placé en tête du premier message user, rangée sous son propre hash :

```bash
uv run python finetune/gemma_variant.py finetune/data/2184f8f3751e
# -> eaa7c0947e75 finetune/data/eaa7c0947e75   (manifeste : variant + derived_from)
```

Les variantes ne sont jamais choisies par défaut par `train.sh` (passe `DATA=`).
Test court (20 itér.) pour mesurer vitesse et mémoire avant un vrai run sur 16 Go ;
`--steps-per-report 10` car `It/sec` est calculé sur `steps_per_report` pas,
donc faux si `iters` n'en est pas un multiple :

Sur M1 Pro 16 Go, sans `MLX_MAX_OPS_PER_BUFFER`/`MLX_MAX_MB_PER_BUFFER`, le premier
pas plante (`kIOGPUCommandBufferCallbackErrorImpactingInteractivity` : le chien de
garde GPU de macOS tue un command buffer trop long). Avec ces variables et
`--num-layers 4` : ~12 s/itér., pic mémoire 10,5 Go, ~9 s par lot de validation.

```bash
MLX_MAX_OPS_PER_BUFFER=10 MLX_MAX_MB_PER_BUFFER=10 caffeinate -i uv run mlx_lm.lora \
  --model mlx-community/gemma-2-9b-it-4bit \
  --train --data finetune/data/eaa7c0947e75 --adapter-path finetune/adapters-gemma-test \
  --mask-prompt --batch-size 1 --num-layers 4 --iters 20 --learning-rate 1e-4 \
  --steps-per-report 10 --steps-per-eval 20 --val-batches 2 --grad-checkpoint \
  2>&1 | tee finetune/logs/test_gemma_$(date +%Y%m%d-%H%M%S).log
```

Vrai run (~2 h ; `VAL_BATCHES=10` pour ne pas passer ~40 min en évaluations) :

```bash
MLX_MAX_OPS_PER_BUFFER=10 MLX_MAX_MB_PER_BUFFER=10 \
  MODEL=mlx-community/gemma-2-9b-it-4bit DATA=finetune/data/eaa7c0947e75 \
  ADAPTERS=finetune/adapters-gemma9b ITERS=500 NUM_LAYERS=4 GRAD_CHECKPOINT=1 \
  VAL_BATCHES=10 caffeinate -i ./finetune/train.sh
```

Import Ollama : pas de safetensors direct (Ollama 0.40 quantifie les safetensors via
son moteur MLX, qui refuse `Gemma2ForCausalLM`). Fusion fp16, puis GGUF avec
llama.cpp (`brew install llama.cpp` pour `llama-quantize` ; `convert_hf_to_gguf.py`
vient du dépôt llama.cpp, même version). Le convertisseur gemma exige
`tokenizer.model`, absent du repo mlx-community : le copier depuis
`unsloth/gemma-2-9b-it` (même vocabulaire). Le `TEMPLATE` de
`finetune/Modelfile.gemma9b` place le system dans le 1er message user, comme à
l'entraînement.

```bash
uv run mlx_lm.fuse --model mlx-community/gemma-2-9b-it-4bit \
  --adapter-path finetune/adapters-gemma9b --save-path finetune/fused-gemma9b --de-quantize
uv run --with <llama.cpp>/gguf-py --with sentencepiece --with "protobuf<5" \
  python <llama.cpp>/convert_hf_to_gguf.py finetune/fused-gemma9b --outtype bf16 \
  --outfile finetune/fused-gemma9b-gguf/lsf-gemma9b-v1-bf16.gguf
llama-quantize finetune/fused-gemma9b-gguf/lsf-gemma9b-v1-bf16.gguf \
  finetune/fused-gemma9b-gguf/lsf-gemma9b-v1-Q4_K_M.gguf Q4_K_M
uv run python finetune/register_ollama.py lsf-gemma9b:v1 --modelfile finetune/Modelfile.gemma9b \
  --meta finetune/adapters-gemma9b/training_meta.json
```

Publication HF : repo séparé (licence **Gemma**, déduite de la base par
`publish_hf.py`), avec le GGUF et le Modelfile pour servir sans refaire la chaîne :

```bash
uv run python finetune/publish_hf.py --model-repo <user>/lsf-gemma9b --version v1 \
  --adapters finetune/adapters-gemma9b \
  --eval eval/results/<record prompt prod>.json --eval eval/results/<record règles seules>.json \
  --modelfile finetune/Modelfile.gemma9b \
  --gguf finetune/fused-gemma9b-gguf/lsf-gemma9b-v1-Q4_K_M.gguf \
  --dataset-repo <user>/lsf-fr-gloss
```

## Variante sans Ollama

Au lieu des étapes 4-5, tu peux servir directement l'adaptateur en MLX :

```bash
uv run mlx_lm.server --model mlx-community/Qwen2.5-3B-Instruct-4bit --adapter-path finetune/adapters
# API compatible OpenAI sur http://localhost:8080
```

…et pointer un client dessus. Plus simple (pas de GGUF), mais hors du circuit
Ollama/`evaluate.py` actuel.

## Alternatives d'élève

| Élève | Licence | Note |
| --- | --- | --- |
| `mlx-community/Qwen2.5-3B-Instruct-4bit` | Apache-2.0 | **recommandé** : ungated, 4-bit prêt, 16 Go OK |
| `mlx-community/gemma-2-2b-it-4bit` | Gemma (gated) | lignée Gemma ; nécessite token HF + accepter la licence |
| `mlx-community/Qwen2.5-7B-Instruct-4bit` | Apache-2.0 | plus capable mais plus tendu en 16 Go à l'entraînement |

## Rappels honnêtes

- Quelques centaines d'exemples **synthétiques** : suffisant pour **valider la
  chaîne**, pas pour un modèle fiable. Vise 1-2k paires **validées LSF** pour un
  vrai gain, et garde `eval/dataset.jsonl` strictement **hors** du train.
- Les noms de flags MLX-LM évoluent entre versions — vérifie `--help`.
