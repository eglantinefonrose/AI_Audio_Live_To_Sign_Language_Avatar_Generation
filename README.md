# LSF Live — Transcription temps réel Parole → Glossaire LSF

Application Python **modulaire** qui transcrit la parole en continu et la
convertit en **gloses de la Langue des Signes Française (LSF)**, prête pour
l'intégration d'un moteur de rendu par **avatar 3D**.

```
Micro ──▶ STT (Kyutai/Moshi + VAD) ──▶ Traducteur LSF (LLM via Ollama) ──▶ File de gloses ──▶ Consommateurs (CLI, avatar 3D…)
```

---

## Architecture

Le système repose sur **deux composants métier indépendants**, reliés par une
interface neutre et orchestrés (mais jamais couplés) par le pipeline :

| Module | Rôle |
| --- | --- |
| `config.py` | Configuration centralisée (modèles, endpoints, audio) via dataclasses. |
| `contracts.py` | **Interface neutre** : `TranscriptEvent` (phrase) et `GlossEvent` (gloses). Ni l'un ni l'autre composant ne dépend de son voisin — seulement de ces types. |
| `stt_engine/` | **Composant 1 — transcription.** `Transcriber` (façade) au-dessus de `engine.py` (Kyutai STT + VAD) ; sources `audio_capture.py` (micro) et `file_source.py` (vidéo/audio via ffmpeg). |
| `lsf_translator/` | **Composant 2 — traduction LSF.** `translator.py` (client Ollama) + `prompts.py` (règles LSF + schéma JSON). |
| `pipeline/` | `orchestrator.py` : chaîne les deux composants via des `asyncio.Queue`. `events.py` ré-exporte les contrats. |
| `avatar_adapter/` | `AvatarRendererInterface` — classe de base abstraite pour brancher l'avatar 3D. |
| `cli.py` | Outil ligne de commande : affiche `[STT Raw] → [LSF Gloss]` en direct. |

### Deux composants, réellement séparés

```
                 contracts.py (interface neutre)
                 ┌───────────────┴───────────────┐
   [source audio] ─▶ Transcriber ──TranscriptEvent──▶ LSFTranslator ──GlossEvent──▶ renderers
                   (composant 1)                     (composant 2)
```

- **`Transcriber`** transforme une source audio en flux de phrases (`TranscriptEvent`),
  la découpe étant assurée par la VAD de Kyutai. Utilisable seul :
  ```python
  for phrase in Transcriber(cfg.stt, cfg.audio).transcribe(source):
      print(phrase.text)
  ```
- **`LSFTranslator`** convertit une phrase en gloses LSF. Utilisable seul :
  ```python
  gloss_event = await LSFTranslator(cfg.ollama).translate_event(phrase)
  ```
- Les deux composants **ne s'importent pas l'un l'autre** : ils ne connaissent
  que `contracts`. L'orchestrateur est le seul à les relier, via des
  `asyncio.Queue` (la transcription, bloquante, tourne dans un thread dédié et
  publie de façon thread-safe). On peut tester, remplacer ou réutiliser chaque
  composant indépendamment.

---

## Prérequis système (macOS Apple Silicon / M1)

### 1. PortAudio (pour `sounddevice`)

```bash
brew install portaudio
```

### 2. Ollama (backend LLM local)

```bash
# Installation
brew install ollama            # ou https://ollama.com/download

# Démarrage du serveur (laisser tourner dans un terminal)
ollama serve

# Téléchargement du modèle par défaut (dans un autre terminal)
ollama pull qwen2.5:7b         # défaut sûr (~4.7 Go), tient sur M1 16 Go avec le STT
# Qualité supérieure à taille comparable (à comparer via --model) :
#   ollama pull gemma2:9b      # ~5.4 Go — excellent en français, fidèle au texte
#   ollama pull qwen3:8b       # ~5.2 Go — génération plus récente (voir note *thinking*)
```

> Le modèle configuré doit être **réellement installé** (`ollama list`). Sinon
> l'application s'arrête au démarrage avec la liste des modèles disponibles.
> Choisissez-en un autre avec `--model <nom>`.

Endpoint par défaut : `http://localhost:11434` (modifiable via `--ollama-host`).

### 3. ffmpeg (pour transcrire un fichier vidéo/audio)

Nécessaire uniquement pour le mode fichier (pas pour le micro) :

```bash
brew install ffmpeg
```

---

## Installation Python (uv)

> La gestion des dépendances se fait avec [**uv**](https://docs.astral.sh/uv/).
> Python **3.10–3.12** requis (contrainte `requires-python` dans `pyproject.toml`).

```bash
# Crée le .venv et installe toutes les dépendances depuis pyproject.toml
uv sync

# Variante Apple Silicon avec le backend MLX (voir plus bas)
uv sync --extra mlx
```

`uv sync` génère un `uv.lock` (versions figées, reproductibles) et provisionne
le `.venv` automatiquement — pas besoin de l'activer à la main, `uv run` s'en
charge.

Les poids du modèle Kyutai `kyutai/stt-1b-en_fr` sont **téléchargés
automatiquement** depuis Hugging Face au premier lancement (quelques Go, mis en
cache dans `~/.cache/huggingface`). Pour pré-télécharger :

```bash
uv run huggingface-cli download kyutai/stt-1b-en_fr
```

> **Repli pip** : un `requirements.txt` est conservé pour les environnements
> sans uv (`pip install -r requirements.txt`). Pour le régénérer depuis le
> lock : `uv export --no-hashes --format requirements-txt > requirements.txt`.

---

## Utilisation

`uv run` exécute la commande dans l'environnement du projet (sans activation
manuelle du `.venv`).

```bash
# Lister les micros disponibles
uv run python cli.py --list-devices

# Lancer la transcription temps réel (micro par défaut)
uv run python cli.py

# Transcrire un fichier vidéo/audio présent dans le projet
uv run python cli.py videos/demo.mp4
uv run python cli.py --language fr media/reunion.mov

# Options courantes
uv run python cli.py --language fr --model qwen2.5:7b --stt-device cpu
uv run python cli.py --input-device 1 --verbose
```

### Mode fichier vs micro

| | Source | Fin |
| --- | --- | --- |
| **Micro** (aucun argument) | `AudioCapture` (sounddevice) | `Ctrl+C` |
| **Fichier** (chemin en argument) | `FileAudioSource` (ffmpeg) | automatique en fin de média |

Le mode fichier accepte tout format géré par ffmpeg (mp4, mov, mkv, webm, wav,
mp3…) : la piste audio est extraite, rééchantillonnée en 24 kHz mono, puis
transcrite par le même pipeline. La segmentation en phrases (VAD) est identique.

#### Cadence : `--realtime`

Le pipeline est **toujours en streaming** (STT, traduction et affichage tournent
en parallèle). Mais par défaut, en mode fichier, le STT traite la vidéo **aussi
vite que le matériel le permet** (bien plus vite que le temps réel) : le STT
« termine » donc la vidéo avant que les traductions LLM, plus lentes, ne
défilent.

Pour un rendu **live** (phrases cadencées au rythme réel de la vidéo) :

```bash
uv run python cli.py videos/demo.mp4 --realtime
```

> `--realtime` ralentit seulement si le STT est plus rapide que le temps réel ;
> si le STT est déjà le goulot d'étranglement, aucun délai n'est ajouté. Au
> micro, la cadence est intrinsèquement temps réel (option sans effet).

### Tester uniquement la transcription (`--stt-only`)

Pour valider le STT **sans Ollama** et visualiser la découpe en phrases qui
alimente le traducteur LSF :

```bash
# Sur un fichier (recommandé pour des tests reproductibles)
uv run python cli.py videos/demo.mp4 --stt-only

# Au micro
uv run python cli.py --stt-only

# Régler le seuil de pause VAD qui valide une phrase (défaut 1.0 s)
uv run python cli.py videos/demo.mp4 --stt-only --vad-pause 1.5
```

Sortie (chaque bloc est ce qui serait transmis tel quel au LLM) :

```
🎬  STT seul sur « videos/demo.mp4 » (pause VAD = 1.0s, pas de traduction)…

  ┌─ phrase #01  →  [envoyée au LLM]
  └─ 'bonjour tout le monde'

  ┌─ phrase #02  →  [envoyée au LLM]
  └─ 'je vais à paris demain'

✅  Terminé — 2 phrase(s) détectée(s).
```

> Ce mode ne requiert **que** les dépendances STT (`torch`, `moshi`) — Ollama
> peut être éteint. Augmentez `--vad-pause` si les phrases sont coupées trop
> tôt, diminuez-le si plusieurs phrases sont fusionnées.

Exemple de sortie :

```
🎙️  Parlez… (Ctrl+C pour quitter)

[STT Raw] bonjour comment ça va aujourd'hui
       -> [LSF Gloss] ['AUJOURD-HUI', 'TOI', 'CA-VA', 'COMMENT']

[STT Raw] je ne veux pas aller à paris demain
       -> [LSF Gloss] ['DEMAIN', 'PARIS', 'MOI', 'ALLER', 'NON']
```

### Options CLI

| Option | Description |
| --- | --- |
| `--list-devices` | Liste les périphériques d'entrée et quitte. |
| `--model` | Modèle Ollama (défaut `qwen2.5:7b`). |
| `--ollama-host` | URL de l'instance Ollama. |
| `--hf-repo` | Dépôt HF du modèle STT (défaut `kyutai/stt-1b-en_fr`). |
| `--stt-device` | `auto` \| `cpu` \| `cuda` \| `mps`. |
| `--language` | `auto` \| `fr` \| `en` (défaut `auto`). |
| `--input-device` | Index du micro. |
| `-v, --verbose` | Journalisation `DEBUG`. |

---

## Segmentation par VAD

Le modèle `kyutai/stt-1b-en_fr` embarque une **VAD sémantique** : il émet ses
jetons texte au rythme de la parole et des jetons de remplissage pendant les
silences. Le moteur (`stt_engine/engine.py`) exploite ce signal — une absence de
nouveau mot pendant `vad_pause_seconds` (défaut **1 s**) valide une **phrase
complète**, plutôt que des découpages temporels fixes. Réglable dans
`config.py` (`STTConfig.vad_pause_seconds`).

---

## Étendre avec un avatar 3D

Il suffit d'implémenter `AvatarRendererInterface` et de l'enregistrer auprès du
pipeline — **aucune modification du STT ni du traducteur** n'est nécessaire :

```python
from avatar_adapter.interface import AvatarRendererInterface
from pipeline.events import GlossEvent

class Unity3DAvatar(AvatarRendererInterface):
    async def start(self):
        ...  # ouvrir une connexion WebSocket vers le moteur 3D

    async def render(self, event: GlossEvent):
        await self._socket.send_json({"glosses": event.glosses})

    async def stop(self):
        await self._socket.close()
```

```python
# Dans votre point d'entrée :
pipeline = Pipeline(CONFIG, renderers=[Unity3DAvatar()])
```

Le renderer devient un simple **consommateur de la file de gloses**.

---

## Gestion des erreurs

L'application échoue tôt et explicitement, avec des codes de sortie distincts :

| Code | Cause |
| --- | --- |
| `2` | Ollama injoignable (lancez `ollama serve`). |
| `3` | Aucun périphérique audio / micro indisponible. |
| `4` | `torch`/`moshi` non installés. |

Toutes les erreurs non fatales (réponse LLM invalide, trame audio perdue,
renderer défaillant) sont **journalisées** sans interrompre le flux temps réel.

---

## Variante Apple Silicon (MLX, optionnelle)

Pour de meilleures performances sur M1/M2/M3, Kyutai fournit un backend **MLX**.
Installez l'extra (`uv sync --extra mlx`), puis adaptez `STTEngine.load()`
pour charger via `moshi_mlx` (voir le dépôt
[kyutai-labs/delayed-streams-modeling](https://github.com/kyutai-labs/delayed-streams-modeling)).
Le reste du pipeline (files, traducteur, avatar) reste inchangé grâce au
découplage.
```
