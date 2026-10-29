# Génération de données pour l'avatar

Données de signes servant à animer l'avatar (l'animation elle-même est dans
`src/sign_language_3d_avatar/avatar_animation/`) :

- **sous-ensemble LSFB** (`lsfb_subset.py`) : vraies captures de mouvement, en LSFB ;
- **lexiques SiGML** (`sigml/*.sigml`) : descriptions de 50 signes en notation
  HamNoSys, brouillons à valider (outil de validation : `sigml_validation.py`).

## Sous-ensemble LSFB

`lsfb_subset.py` télécharge depuis **LSFB-ISOL** (Université de Namur) uniquement les
signes isolés des gloses choisies : poses MediaPipe par défaut, vidéos en option.

## ⚠️ LSFB ≠ LSF

LSFB = **langue des signes de Belgique francophone**. Ce n'est pas la LSF de France,
alors que le traducteur produit des gloses LSF. Les deux langues sont proches,
mais beaucoup de signes diffèrent : le lexique, les nombres, les signes-noms de
lieux et de personnes, ainsi que certaines configurations.
Conséquences pour l'avatar :

- un avatar animé avec ces données **signe en LSFB**. Pour un public français,
  il faut valider chaque signe avec un locuteur LSF, ou le remplacer par une
  source LSF ;
- une étiquette identique (`MAISON` ↔ `MAISON`) veut dire « même mot français »,
  pas « même signe » ;
- les gloses du projet ne sont pas validées par un locuteur LSF non plus
  (cf. `CONVENTIONS.md`).

## Source et licence

- Site : https://lsfb.info.unamur.be/#dataset. Le téléchargement suppose d'accepter
  la licence sur ce site.
- Fichiers : `https://lsfb.info.unamur.be/static/datasets/lsfb_v2/isol/`. Ils sont
  publics, sans compte. Ce sont les mêmes que ceux de la librairie officielle
  [`lsfb-dataset`](https://pypi.org/project/lsfb-dataset/), qui ne sait pas filtrer
  par glose.
- **Licence CC BY-NC-SA 4.0** :
  - pas d'usage commercial ;
  - partage dans les mêmes conditions ;
  - citation obligatoire : *Meurant, Laurence (2015), Corpus LSFB. Un corpus informatisé
    en libre accès de vidéos et d'annotations de la langue des signes de Belgique
    francophone (LSFB). LSFB-Lab, FRS-FNRS et Université de Namur*, plus l'article
    LSFB-ISOL : https://ieeexplore.ieee.org/abstract/document/9534336.
  - Les données vont dans `src/sign_language_3d_avatar/dataset_generation/data/`, ignoré par git. Ne pas les commiter.

## Utilisation

```bash
uv run python src/sign_language_3d_avatar/dataset_generation/lsfb_subset.py map
```

Cette commande télécharge les index (~12 Mo) et écrit `src/sign_language_3d_avatar/dataset_generation/lsfb_mapping.csv` :
glose projet → étiquette LSFB, statut, nombre d'instances, autres variantes, note.

```bash
uv run python src/sign_language_3d_avatar/dataset_generation/lsfb_subset.py download --words-file src/sign_language_3d_avatar/dataset_generation/lsfb_words.txt --dry-run
```

`--dry-run` affiche, glose par glose, le nombre d'instances et la taille estimée
(requêtes HEAD sur un échantillon, extrapolées à la durée), sans rien télécharger.

```bash
uv run python src/sign_language_3d_avatar/dataset_generation/lsfb_subset.py download --words-file src/sign_language_3d_avatar/dataset_generation/lsfb_words.txt
```

Options utiles :
- `--words A,B` : liste directe ; `lsfb:ÉTIQUETTE` force une étiquette LSFB.
- `--max-par-glose N` : 10 par défaut, 0 = toutes. Le choix tourne sur les signeurs.
- `--landmarks pose,left_hand,right_hand,face` : `face` pèse environ 15 fois plus.
- `--raw` : poses non lissées.
- `--videos` : clips .mp4.
- `--inclure-incertains` : prend aussi les statuts à valider.

**Reprise** : relancer la même commande. Les fichiers complets sont sautés et les
`.part` reprennent là où ils s'étaient arrêtés (en-tête HTTP Range).

**Manifeste** : `src/sign_language_3d_avatar/dataset_generation/data/lsfb/manifest.csv`, avec pour chaque ligne la glose,
l'étiquette, l'id d'instance, le signeur, le début et la fin (ms), les fichiers,
la source et la licence. Il est complété à chaque exécution.

Poses : un `.npy` float16 de forme `(frames, landmarks, 3)` par instance et par
partie (pose = 33 points, chaque main = 21, face = 468), en coordonnées normalisées
MediaPipe. Les versions nettoyées sont interpolées et lissées (Savitzky-Golay).

## Correspondance des gloses (`lsfb_mapping.csv`)

LSFB note sans accents (`ECOLE`), avec des traits d'union (`AUJOURD-HUI`, `C-EST`),
des nombres en chiffres (`2`) et des variantes `BASE.PRÉCISION` (`GRAND.HAUTEUR`,
`PLUS.P`). Statuts :

| statut | sens | téléchargé par défaut |
|---|---|---|
| `exact` | même étiquette après normalisation (accents, `'` → `-`, nombres) | oui |
| `directionnel` | `VERBE-À-X` → forme de citation `VERBE` ; l'orientation vers X est à produire par l'avatar | oui |
| `variante` | seulement des `BASE.X` : choisir le bon signe | non |
| `synonyme` | équivalent proposé à la main (`PÈRE`→`PAPA`, `REMERCIER`→`MERCI`) | non |
| `décomposé` | composé absent, signes séparés (`LUNDI+DERNIER`) | non |
| `nom-propre` | signe-nom belge `NS:PARIS` ; l'épellation est peut-être préférable | non |
| `incertain` | homographe sans accent (`OÙ` → `OU`, qui veut aussi dire « ou ») | non |
| `pointage` | `MOI/TOI/…` → `PT:PRO1…6` : annotés dans le corpus mais **aucun clip servi** dans ISOL ; à générer | — |
| `absent` | pas dans LSFB (termes techniques, noms français, etc.) | — |

Même `exact` ne garantit pas le sens : `AIMER` a aussi `AIMER.PLAISIR`, et le
clip sans suffixe peut mélanger plusieurs signes annotés pareil. Il faut regarder
quelques vidéos (`--videos --max-par-glose 3`) avant de figer un signe.

`src/sign_language_3d_avatar/dataset_generation/lsfb_words.txt` contient la liste proposée : les gloses sûres sont actives,
celles à valider sont commentées. Pour régénérer la table après une mise à jour de
`lexicon.txt` ou de `eval/dataset.jsonl`, relancer `map`.

## Lexiques SiGML (`sigml/`)

Descriptions de signes en SiGML (notation HamNoSys en XML, balises de la DTD
officielle `sigmlhns.dtd` de l'UEA). Ce sont des **brouillons rédigés par une IA,
non validés** par un locuteur LSF ; chaque signe porte un commentaire avec sa
description en français et un niveau de confiance (bonne / moyenne / faible).

| Fichier | Gloses |
|---|---|
| `lexique_brouillon.sigml` | MOI, TOI, MERCI, MAISON, DONNER-À-TOI, DONNER-À-MOI |
| `pronoms.sigml` | LUI, NOUS, VOUS, EUX |
| `temps_et_negation.sigml` | AVANT, BIENTÔT, HIER, DEMAIN, NON |
| `verbes_et_quotidien.sigml` | MANGER, BOIRE, VOIR, REGARDER, PENSER, VENIR, BIEN |
| `nombres.sigml` | DEUX, TROIS, QUATRE, CINQ |
| `questions.sigml` | QUOI, OÙ, QUI, POURQUOI, COMMENT, COMBIEN |
| `verbes_et_objets.sigml` | MAINTENANT, MATIN, SOIR, AIMER, VOULOIR, POUVOIR, PARLER, PARTIR, COMPRENDRE, APPRENDRE, TRAVAILLER, FINIR, AIDER-À-TOI, AIDER-À-MOI, VOITURE, BEAUCOUP, ACHETER, CAFÉ |

Pour ajouter un signe, il suffit de créer ou compléter un `.sigml` dans ce dossier :
tous les fichiers sont chargés. Si une glose est définie deux fois, la dernière
définition (par ordre alphabétique des fichiers) gagne.
QUESTION (marqueur de question fermée) n'a pas de signe manuel : en LSF, la
question se marque surtout par le visage, que l'animation ne gère pas. Ils sont joués par `src/sign_language_3d_avatar/avatar_animation/sigml_blender.py` (conventions

### Validation par un locuteur LSF (pilote de 50 signes)

```bash
uv run python src/sign_language_3d_avatar/dataset_generation/sigml_validation.py --ouvrir
```

Ce script lit le commentaire de chaque signe (description et niveau de confiance)
et produit deux fichiers :
- `sigml/validation_pilote.csv` : une ligne par signe, avec les colonnes `verdict`,
  `correction` et `commentaire` à remplir. Les verdicts déjà saisis sont conservés
  quand on relance le script ;
- `avatar_animation/data/sigml/validation.html` : une page locale qui montre chaque
  vidéo à côté de sa description, avec le formulaire de verdict et un bouton
  « Exporter le CSV ».

Les vidéos doivent avoir été rendues avant, avec `sigml_blender.py`.

Niveaux de confiance actuels : 2 bonne, 1 bonne pour la forme manuelle, 23 moyenne,
24 faible. Le but du pilote est de mesurer le taux réel d'erreur et le temps de
relecture avant de viser ~500 signes.
et balises gérées décrites dans `src/sign_language_3d_avatar/avatar_animation/README.md`).
