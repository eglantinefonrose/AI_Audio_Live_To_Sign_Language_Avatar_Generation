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
| `décomposé` | composé absent, signes séparés joués à la suite (`LUNDI+DERNIER`) | oui |
| `nom-propre` | signe-nom belge `NS:PARIS` ; l'épellation est peut-être préférable | non |
| `incertain` | homographe sans accent (`OÙ` → `OU`, qui veut aussi dire « ou ») | non |
| `pointage` | `MOI/TOI/…` → `PT:PRO1…6` : annotés dans le corpus mais **aucun clip servi** dans ISOL ; à générer | — |
| `validé` | étiquette choisie à la main sur la page de relecture (`lsfb_candidats.csv`) | oui |
| `épeler` | relu : nom propre ou terme technique, épelé avec l'alphabet manuel LSFB (`GPU` → `FS:G+FS:P+FS:U`) | oui |
| `absent` | pas dans LSFB (termes techniques, noms français, etc.) | — |

Les réponses de la page de relecture (`lsfb_candidats.csv`, voir plus bas) priment sur
la proposition automatique à chaque `map`.

Même `exact` ne garantit pas le sens : `AIMER` a aussi `AIMER.PLAISIR`, et le
clip sans suffixe peut mélanger plusieurs signes annotés pareil. Il faut regarder
quelques vidéos (`--videos --max-par-glose 3`) avant de figer un signe.

`src/sign_language_3d_avatar/dataset_generation/lsfb_words.txt`, réécrit par `map`, contient la liste de
téléchargement : les gloses sûres (`exact`, `directionnel`, `validé`) sont actives, celles à
valider sont commentées. Pour régénérer les deux fichiers après une mise à jour de
`lexicon.txt`, de `eval/dataset.jsonl`, de `eval/dev.jsonl` ou de `lsfb_candidats.csv`, relancer `map`.

## Relire les gloses sans signe sûr (`candidats_absents.py`)

Les gloses `absent`, `variante`, `incertain`, `synonyme` et `nom-propre` ne sont pas
dans le dictionnaire de l'avatar. La page de relecture propose, pour chacune, jusqu'à 4
étiquettes LSFB (synonymes proposés à la main, lemme, racine, variantes, orthographe
voisine en dernier recours), avec 2 clips par étiquette :

```bash
uv run python src/sign_language_3d_avatar/dataset_generation/candidats_absents.py --ouvrir
```

Au clavier : <kbd>1</kbd>–<kbd>4</kbd> ce signe, <kbd>A</kbd> aucun, <kbd>E</kbd> à épeler,
<kbd>I</kbd> incertain ; une autre étiquette LSFB peut être tapée à la main (elle devient
un candidat, avec sa vidéo, à la relance). **Exporter le CSV**, le copier à la place de
`lsfb_candidats.csv`, puis relancer `lsfb_subset.py map` : les signes choisis passent en
`validé` (téléchargés et indexés comme `exact`), les noms à épeler en `épeler` (suite de
lettres `FS:A`…`FS:Z`, les 26 lettres existent dans LSFB-ISOL). Ensuite,
reconstruire le dictionnaire ([avatar_animation/README.md](../avatar_animation/README.md#dictionnaire-de-signes-lsfb)).

**Relecture ciblée** (`--bloquantes`) : seulement les gloses encore sans signe qui empêchent
de jouer une phrase du banc d'essai (21 gloses, 21 phrases au 11/11/2026), même si elles
ont déjà reçu « aucun ». Jusqu'à 8 candidats : de nouvelles pistes écrites à la main
(`PISTES` : synonymes, suites de signes `A+B` comme `AVANT+ENTRAINEMENT`, jouées l'une
après l'autre dans la vidéo), toutes les variantes LSFB du mot, puis les candidats déjà
refusés. La page `data/lsfb/candidats_bloquants.html` rappelle la réponse précédente ; son
export contient tout le CSV (les gloses non relues ici gardent leur réponse) et remplace
`lsfb_candidats.csv`. Une suite `A+B` choisie ou tapée devient le statut `décomposé`.

```bash
uv run python src/sign_language_3d_avatar/dataset_generation/candidats_absents.py --bloquantes --ouvrir
```

## Choisir le meilleur exemple de chaque signe (`select_examples.py`)

Sans relecture : parmi les ~20 exemples téléchargés par signe, le script garde celui que
le **reconnaisseur** de l'évaluation (`evaluation/features.py`) reconnaît le mieux parmi les
exemples des autres signeurs : le plus typique du signe. Les personnes filmées sont
partagées en deux moitiés fixes (`features.signer_half`) : la moitié « choix » sert ici,
la moitié « eval » sert seulement à noter l'avatar, pour que l'évaluation ne note jamais
des clips choisis avec ses propres exemples. À égalité, ou pour les signes sans autre
exemple dans la moitié « choix », le score technique départage (0,5 × taux d'images où
les deux mains sont détectées, mesuré sur les poses **brutes** ; 0,3 × faible
tremblement ; 0,2 × durée proche de la médiane du signe ; seul critère avec
`--sans-reconnaisseur`). Prérequis : `download --max-par-glose 20` (≈ 10 exemples par moitié).
Résultat : `selection.csv` (versionné, identifiants et scores seulement). Suite de la
chaîne : [avatar_animation/README.md](../avatar_animation/README.md#dictionnaire-de-signes-lsfb).

### Choisir les clips à l'œil (`revue_clips.py`)

Le reconnaisseur se laisse piéger : deux clips atypiques qui se ressemblent se classent
1ers l'un grâce à l'autre (OU, QUAND.8, ENFIN, DEVOIR…). Les corriger avec les scores de
l'évaluation reviendrait à régler sur l'évaluation : on les choisit donc à l'œil. La page
montre les signes qui font échouer le plus de phrases du banc (dernier record du
dictionnaire actuel), avec tous leurs exemples, sans aucun score :

```bash
uv run python src/sign_language_3d_avatar/dataset_generation/revue_clips.py --ouvrir
```

Les lettres de l'alphabet manuel (`FS:N`…) se relisent à part, avec `--lettres` (page
`data/lsfb/revue_lettres.html`) : chaque carte indique les mots épelés où la lettre fait
échouer une phrase. L'export contient aussi les réponses déjà présentes dans le CSV.

**Lettres d'après une vidéo ou une image de référence** (`lettres_images.py`) : déposer
une vidéo par lettre (`A.mp4`…, de face, main bien visible, la lettre tenue ~1 s entre deux
repos) ou à défaut une photo (`N.jpg`, `P.png`…) dans `data/alphabet/` (ignoré par git),
nommée par la lettre ; la vidéo l'emporte. Pour les lettres aux doigts repliés cachés par
la main (K, Q, X, Z…), ajouter une vidéo de profil `K-profil.mp4`, prioritaire : seuls les
doigts sont repris, l'orientation vient du clip. Une source jugée correcte sur la page de
vérification est gardée. Puis :

```bash
uv run --with mediapipe python src/sign_language_3d_avatar/dataset_generation/lettres_images.py
```

MediaPipe extrait les 21 points de la main (`N.json`) et dessine une image de contrôle
(`N_controle.png`) à vérifier. `clean_signs.py` garde ensuite le bras, la position et
l'orientation du clip LSFB de la lettre, et remplace seulement les doigts par ceux de
l'image. Supprimer `N.json` revient au clip seul. MediaPipe ne lit que des photos de vraies
mains, assez grandes : les dessins au trait ne sont pas détectés.

**Vérifier les lettres sur l'avatar** (`verif_lettres.py`) : l'avatar signe chaque lettre
(rendu Blender, `avatar_animation/demo_lettres_blender.py`), en pied et en gros plan sur la
main, à côté du clip LSFB copié et de l'image de référence s'il y en a une. On répond
correcte / douteuse / incorrecte (touches C / D / I) ; l'export `verif_lettres.csv` est à
copier dans `dataset_generation/`. Page : `data/lsfb/verif_lettres.html`.

```bash
uv run python src/sign_language_3d_avatar/dataset_generation/verif_lettres.py --ouvrir
```

Exporter le CSV, le copier dans `dataset_generation/choix_clips.csv` (versionné), puis
relancer `select_examples.py` : chaque clip choisi passe avant les critères automatiques
(critère « relecture »). « Autre variante » ne change pas le clip : ces signes se relisent
ensuite avec `candidats_absents.py --variantes` (pistes comme MODELE.EXEMPLE ou
OBLIGER.DEVOIR, toutes les variantes LSFB du mot, le signe actuel en dernier) ; son export
remplace `lsfb_candidats.csv`, puis `lsfb_subset.py map` et la suite de la chaîne.

```bash
uv run python src/sign_language_3d_avatar/dataset_generation/candidats_absents.py --variantes --ouvrir
```

## Étiqueter LSFB → LSF (`lsf_comparaison.py`)

`lsfb_mapping.csv` rapproche des **mots**, pas des **gestes** : « exact » veut dire
même étiquette française, pas forcément même signe en LSF de France. La page
d'étiquetage sert à le noter, signe par signe.

1. Télécharger une vidéo par glose (≈ 80 Mo, une seule fois) :
   ```bash
   uv run python src/sign_language_3d_avatar/dataset_generation/lsfb_subset.py download --words-file src/sign_language_3d_avatar/dataset_generation/lsfb_words.txt --videos --landmarks pose --max-par-glose 1
   ```
2. Générer et ouvrir la page :
   ```bash
   uv run python src/sign_language_3d_avatar/dataset_generation/lsf_comparaison.py --ouvrir
   ```
   Les clips LSFB sont en MPEG-4 Part 2, que les navigateurs ne lisent pas : le script en
   fait des copies H.264 (ffmpeg) dans `data/lsfb/videos_h264/`, sans toucher aux originaux.
3. Pour chaque signe, regarder le clip et répondre au clavier :
   <kbd>1</kbd> identique · <kbd>2</kbd> variante · <kbd>3</kbd> différent ·
   <kbd>4</kbd> je ne sais pas. La page passe seule au suivant ; <kbd>←</kbd>/<kbd>→</kbd>
   pour naviguer, <kbd>Espace</kbd> pour rejouer, <kbd>L</kbd> pour le ralenti,
   <kbd>N</kbd> pour une note, <kbd>0</kbd> pour effacer une réponse.
   - Ordre : gloses les plus fréquentes dans les datasets d'abord.
   - Les formes d'un même verbe (`AIDER`, `AIDER-À-MOI`, `AIDER-À-TOI`) ne font qu'une
     carte : le verdict vaut pour toutes.
4. Les réponses sont gardées dans le navigateur. **Exporter le CSV** (le bouton devient
   rouge toutes les 25 réponses) puis remplacer le fichier du projet :
   ```bash
   mv ~/Downloads/lsfb_lsf_comparaison.csv src/sign_language_3d_avatar/dataset_generation/lsfb_lsf_comparaison.csv
   ```
   Relancer le script reprend les verdicts du CSV (le plus récent l'emporte). Seuls les
   signes **identiques** sont à utiliser tels quels pour l'avatar.

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
