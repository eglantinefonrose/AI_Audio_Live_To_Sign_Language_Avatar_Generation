Développer une application Python modulaire permettant la transcription en temps réel de la parole/audio en jetons (tokens) de glossaire pour la Langue des Signes Française (LSF).

## Exigences architecturales
Le système doit être conçu avec un découplage strict et une séparation nette des composants, en vue de l'intégration ultérieure d'un module de rendu par avatar 3D.

Structure architecturale :
1. `stt_engine/` : Capture audio et module STT de Kyutai.
2. `lsf_translator/` : Convertisseur basé sur un LLM (via Ollama en local) pour générer le glossaire LSF.
3. `pipeline/` : Orchestrateur reliant le STT au traducteur LSF, puis à une file d'attente d'événements de sortie.
4. `config.py` : Configuration centralisée (modèles, points de terminaison API, paramètres audio).

---

## Spécifications techniques

### 1. Composant STT (`stt_engine`)
- **Modèle** : Utiliser `kyutai/stt-1b-en_fr` (ou le pipeline STT de Moshi).
- **Intégration VAD** : Utiliser le mécanisme natif de détection d'activité vocale (VAD) / segmentation de la parole de Kyutai. Cela permet d'obtenir des phrases complètes basées sur les pauses dans le discours, plutôt que sur des segments temporels fixes (par ex. des intervalles de 10 secondes).
- **Sortie** : Dès qu'une phrase complète est transcrite et validée par le VAD, émettre un événement ou transmettre la chaîne de caractères vers une file d'attente asynchrone pour traitement.

### 2. Composant de traduction en glossaire LSF (`lsf_translator`)
- **Backend** : Ollama exécuté en local (cible : macOS M1, point de terminaison par défaut `http://localhost:11434`).
- **Entrée** : Phrases en français ou en anglais provenant du flux STT.
- **Sortie** : Une structure Python de type `list[str]` ou un tableau JSON représentant les glosses LSF dans l'ordre chronologique (en respectant la grammaire LSF : Temps -> Lieu -> Sujet -> Objet -> Verbe, ainsi que les structures spatiales le cas échéant).
- **Ingénierie de prompt (Prompt Engineering)** : Inclure un prompt système dédié aux règles de transcription en LSF (par ex. utilisation de majuscules pour les jetons de glossaire, indicateurs spécifiques pour la négation ou les interrogations).
- **Performance** : Optimiser pour une faible latence en temps réel (utiliser des schémas JSON concis ou imposer un formatage strict). ## 3. Pipeline modulaire et extension future
- Implémenter un modèle de type « écouteur d'événements » (Event Listener) / file d'attente asynchrone (`asyncio.Queue`).
- Créer une interface ou une classe de base abstraite `AvatarRendererInterface` dans `avatar_adapter/`, afin qu'un système d'avatar 3D puisse être facilement intégré ultérieurement en tant que consommateur de la file d'attente de glossaire LSF.
- Fournir un outil en ligne de commande (CLI) simple qui affiche en temps réel : `[STT Raw] -> [LSF Gloss Array]`.

---

## Livrables attendus
1. Une structure de répertoire propre avec des modules isolés.
2. Un fichier `requirements.txt` listant toutes les dépendances nécessaires (torch, transformers/moshi, ollama, asyncio, sounddevice, etc.).
3. Un fichier `README.md` complet incluant des instructions de configuration (exécution locale d'Ollama, commandes de téléchargement des modèles, préparation de l'environnement pour macOS M1).
4. Gestion des erreurs et journalisation (logging) en cas de périphériques audio manquants ou d'instance Ollama non opérationnelle.

Veuillez générer la base de code, étape par étape.