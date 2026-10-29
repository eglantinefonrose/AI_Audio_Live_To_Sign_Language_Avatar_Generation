# Avatar 3D en langue des signes : essais

Ce dossier rassemble les essais faits pour qu'un avatar 3D **signe** les gloses LSF
produites par le traducteur (`src/lsf_translator/`). Rien n'est encore branché sur
le pipeline en direct : la prise prévue pour ça est `avatar_adapter/`, qui ne fait
pour l'instant qu'afficher les gloses dans la console.

| Dossier | Rôle | Doc détaillée |
|---|---|---|
| `dataset_generation/` | données de signes : sous-ensemble LSFB, lexique SiGML | [dataset_generation/README.md](dataset_generation/README.md) |
| `avatar_animation/` | scripts Blender qui produisent les animations | [avatar_animation/README.md](avatar_animation/README.md) |
| `avatar_adapter/` | interface entre le pipeline et un avatar (`AvatarRendererInterface`), avatar console | `avatar_adapter/interface.py` |

Les poses LSFB téléchargées et les vidéos (`*.mp4`) sont ignorées par git ; les
exports `.glb`/`.bvh` du retargeting (`avatar_animation/data/retarget/`) sont
commités. Ce sont des dérivés de LSFB, donc sous CC BY-NC-SA 4.0 eux aussi.

## Tester une phrase

```bash
uv run python src/sign_language_3d_avatar/tester_phrase_sigml.py
```

Tapez une phrase en gloses (`TOI DONNER-À-MOI MERCI`) : l'avatar la signe et la
vidéo s'ouvre. Seules les gloses des lexiques SiGML (tous les `.sigml` de
`dataset_generation/sigml/`) sont disponibles, soit 50 gloses (liste dans
`dataset_generation/README.md`). Les
autres sont ignorées avec un avertissement. Le rendu prend quelques secondes et les vidéos sont écrites dans
`avatar_animation/data/sigml/phrases/`.

Pour une phrase sans mode interactif :

```bash
uv run python src/sign_language_3d_avatar/tester_phrase_sigml.py --phrase "MOI MAISON"
```

Si Blender n'est pas dans `/Applications`, passez `--blender` ou la variable `BLENDER`.

> ⚠️ Aucun des signes produits ici n'a été validé par un locuteur LSF. Les gloses du
> projet ne le sont pas non plus (cf. `CONVENTIONS.md`).

---

## Résumé

| # | Essai | Résultat | Verdict |
|---|---|---|---|
| 1 | Récupérer des signes réels dans le dataset LSFB (UNamur) | 354 gloses couvertes, 2 925 clips de poses | données réelles, mais **LSFB ≠ LSF** et clips tirés de conversations |
| 2 | Retargeting des poses LSFB vers un squelette Blender | animation fidèle aux poses | techniquement OK, **non conforme** aux signes attendus |
| 3 | Chercher un dataset de signes LSF isolés | rien d'exploitable et d'accessible | LSF-ANIMAL non publié |
| 4 | Faire générer les signes par une IA | écarté | trop d'erreurs invérifiables |
| 5 | Décrire les signes (HamNoSys/SiGML) et les animer dans Blender | 6 signes animés | **piste retenue pour la suite** |

---

## 1. Dataset LSFB de l'Université de Namur

### Source et conditions

- Site : https://lsfb.info.unamur.be/#dataset. Corpus LSFB, constitué depuis 2012 par le
  LSFB-Lab de l'Université de Namur : 90 h de vidéo, dont 25 h annotées.
- **Langue : LSFB**, la langue des signes de Belgique francophone. Ce n'est **pas** la
  LSF de France : le lexique, les nombres et les signes-noms diffèrent souvent.
- **Licence : CC BY-NC-SA 4.0.**
  - Pas d'usage commercial, partage dans les mêmes conditions.
  - Citation obligatoire : *Meurant, Laurence (2015), Corpus LSFB. Un corpus
    informatisé en libre accès de vidéos et d'annotations de la langue des signes de
    Belgique francophone (LSFB). LSFB-Lab, FRS-FNRS et Université de Namur*.
  - Pour LSFB-ISOL, citer aussi l'article
    https://ieeexplore.ieee.org/abstract/document/9534336.
- **Accès** : pas de compte. Le site demande de cocher l'acceptation de la licence, ce
  qui a été fait par le porteur du projet le 2026-10-27. Les fichiers sont ensuite
  publics en HTTP sous `https://lsfb.info.unamur.be/static/datasets/lsfb_v2/`.

### Ce que contient le dataset

| Variante | Contenu | Poses | Vidéos |
|---|---|---|---|
| **LSFB-ISOL** | clips d'un signe, découpés dans les conversations | 10 Go | 25 Go |
| **LSFB-CONT** | conversations complètes entre deux signeurs, 85 signeurs | 31 Go | ~400 Go |

- Un index `instances.csv` (id, glose, signeur, début, fin) et une table des gloses.
  ISOL compte 4 657 étiquettes, 120 740 clips et 99 signeurs.
- Les **poses** sont des points MediaPipe, en `.npy` float16 de forme
  `(images, points, 3)`, à 50 images/s :
  - 33 points pour le corps, 21 par main, 468 pour le visage ;
  - x et y donnent la position dans l'image (fiable), z une profondeur estimée (très
    approximative) ;
  - deux versions : « nettoyée » (interpolée et lissée) ou brute.
- La librairie officielle `lsfb-dataset` (MIT) filtre seulement par split et par
  signeur, **pas par glose**. D'où le script maison `dataset_generation/lsfb_subset.py`,
  qui lit l'index et ne télécharge que les clips des gloses choisies. Il a un mode
  `--dry-run`, sait reprendre un téléchargement interrompu et tient un manifeste avec
  la licence.

### Correspondance avec les gloses du projet

Les 533 gloses du projet (`src/lsf_translator/lexicon.txt` et `src/eval/dataset.jsonl`) ont été
comparées aux étiquettes LSFB, qui sont sans accents, avec des nombres en chiffres et
des variantes `BASE.X`. Résultat dans `dataset_generation/lsfb_mapping.csv` :

| Statut | Nombre | Exemple |
|---|---|---|
| exact | 325 | `ÉCOLE` → `ECOLE`, `DEUX` → `2` |
| directionnel | 29 | `AIDER-À-TOI` → `AIDER` (orientation à produire par l'avatar) |
| synonyme (à valider) | 20 | `PÈRE` → `PAPA`, `REMERCIER` → `MERCI` |
| variante (à valider) | 14 | `GRAND` → `GRAND.HAUTEUR` / `.LARGEUR` |
| nom propre (à valider) | 7 | `PARIS` → `NS:PARIS` (signe belge) |
| décomposé / incertain | 3 | `LUNDI-DERNIER` → `LUNDI+DERNIER` ; `OÙ` → `OU` (homographe) |
| pointage | 6 | `MOI`…`EUX` → `PT:PRO1…6` : annotés, mais **aucun clip disponible** |
| absent | 129 | termes techniques, noms français, `MAINTENANT`, `TRÈS`… |

**Téléchargé** : les poses (corps et mains) des 354 gloses sûres (exactes et
directionnelles), 10 clips au plus par glose. Cela fait 2 925 clips distincts de
89 signeurs, soit 8 775 fichiers (≈ 32 Mo de données, 48 Mo sur disque). Ni vidéos ni
visage. Deux coupures réseau ont été rattrapées par la reprise.

```bash
uv run python src/sign_language_3d_avatar/dataset_generation/lsfb_subset.py download \
    --words-file src/sign_language_3d_avatar/dataset_generation/lsfb_words.txt --max-par-glose 10
```

---

## 2. Retargeting des poses LSFB vers Blender

Script : `avatar_animation/retarget_blender.py`, lancé par Blender 5.0 sans interface.

**Méthode**
- Un squelette de 41 os est construit : torse, cou, tête, bras, mains à 5 doigts de
  3 phalanges.
- Chaque os vise le point MediaPipe suivant (contrainte « Damped Track »), avec des
  longueurs d'os fixes. L'animation est ensuite convertie en rotations
  d'articulations.
- Exports `.glb` et `.bvh`, plus des aperçus `.mp4` de face et de profil.

**Corrections nécessaires**
- La profondeur des bras donnée par MediaPipe est trop bruitée : elle est reconstruite
  à partir de longueurs de bras plausibles.
- De profil, la tête partait en avant : torse redressé, tête placée au-dessus.
- L'image source est en 4:3 (estimé par la stabilité des longueurs d'os de la main).

**Résultat** (MAISON, AIDER, MERCI, DEUX) : l'animation suit fidèlement le tracé 2D
des poses d'origine.

**Pourquoi le résultat n'est pas conforme aux signes attendus**
- Les clips ISOL sont **découpés dans des conversations spontanées**. Les signes y
  sont rapides, réduits et influencés par les signes voisins, et le découpage peut
  couper un mouvement.
- C'est de la **LSFB**, pas de la LSF.
- Les données sont surtout **2D**, avec une profondeur devinée, et la version
  nettoyée est lissée.
- Il n'y a **pas de visage**, alors que les expressions font partie du signe.
- Chaque clip est isolé, sans transition, et l'avatar prend les proportions du
  signeur.

---

## 3. Recherche d'un dataset de signes LSF isolés

| Ressource | Nature | Exploitable ? |
|---|---|---|
| [LSF-ANIMAL](https://aclanthology.org/2020.lrec-1.736) (Naert, Larboulette, Gibet, LREC 2020) | capture de mouvement LSF faite pour animer des avatars, 2 signeurs, 200 Hz, doigts et visage. Environ 90 signes isolés (11 avec changement de configuration, 9 interrogatifs, 47 noms d'animaux, 25 descripteurs), alphabet, phrases | **Non** : l'article indique que le corpus n'est pas encore mis à disposition, et aucune page de téléchargement n'a été trouvée. Piste : contacter les autrices (IRISA, Université Bretagne Sud) |
| Mediapi-RGB, Matignon-LSF, STVD-LSF, Signes en famille | LSF **continue** (vidéo) | pas de signes isolés ; accès sur accord ou compte Ortolang |
| Rosetta-LSF, STK LSF | capture de mouvement LSF en **phrases** | utiles plus tard pour la génération de phrases |
| ASL Citizen, SignAvatar/ASL3DWord, SignAvatars, MySign | signes isolés, parfois en 3D (SMPL-X) | **pas en LSF** (américaine, malaisienne…) |
| Dictionnaires vidéo LSF (Elix, Spreadthesign…) | signes isolés propres | pas des datasets ; conditions d'utilisation à vérifier |

## 4. Générer les signes par une IA

Écarté. Un modèle de langue (Claude compris) ne connaît pas la LSF de façon fiable. Il
confond avec d'autres langues des signes, invente des signes plausibles mais faux et
oublie le visage. Ces erreurs ne sont pas repérables sans signeur, et les générateurs
de vidéo ou de mouvement manquent de données LSF.

L'IA reste utile pour **rédiger des brouillons** (essai 5) et pour l'outillage :
enregistrement, extraction de poses, rendu.

---

## 5. Notation des signes (HamNoSys / SiGML) → animation Blender

### Le principe

Un signe se décrit par ses paramètres : configuration de la main, orientation,
emplacement, mouvement, une ou deux mains, et non-manuel (visage). L'intérêt est
qu'un signe décrit par paramètres **se modifie par programme**. `DONNER-À-TOI` et
`DONNER-À-MOI` partagent la même description, seule la direction change. De même,
les pointages ne diffèrent que par l'emplacement visé. C'est ce que les clips
enregistrés ne permettent pas.

Notations examinées :
- **AZee** (LISN, conçu pour la LSF) : grammaire formelle animée par l'avatar
  **Paula** (DePaul University). Aucun outil public, export ou lexique LSF n'a été
  trouvé. Piste : contacter l'équipe AZee du LISN.
- **HamNoSys** et sa version XML **SiGML**, lue par l'avatar **CWASA/JASigning** de
  l'université d'East Anglia :
  - noms de balises vérifiés dans la DTD officielle `sigmlhns.dtd`
    (https://www.visicast.cmp.uea.ac.uk/sigml/) ;
  - conditions de CWASA : copyright UEA, utilisation sans modification
    (https://vh.cmp.uea.ac.uk/index.php/JASigning_Conditions) ;
  - CWASA ne sait pas exporter vers Blender.

### Le prototype

- **Lexiques** : 7 fichiers dans `dataset_generation/sigml/`, 50 signes en tout. Ce
  sont des **brouillons rédigés par une IA**, non validés, avec un niveau de
  confiance noté pour chaque signe.

| Fichier | Gloses |
|---|---|
| `lexique_brouillon.sigml` | MOI, TOI, MERCI, MAISON, DONNER-À-TOI, DONNER-À-MOI |
| `pronoms.sigml` | LUI, NOUS, VOUS, EUX |
| `temps_et_negation.sigml` | AVANT, BIENTÔT, HIER, DEMAIN, NON |
| `verbes_et_quotidien.sigml` | MANGER, BOIRE, VOIR, REGARDER, PENSER, VENIR, BIEN |
| `nombres.sigml` | DEUX, TROIS, QUATRE, CINQ |
| `questions.sigml` | QUOI, OÙ, QUI, POURQUOI, COMMENT, COMBIEN |
| `verbes_et_objets.sigml` | MAINTENANT, MATIN, SOIR, AIMER, VOULOIR, POUVOIR, PARLER, PARTIR, COMPRENDRE, APPRENDRE, TRAVAILLER, FINIR, AIDER-À-TOI, AIDER-À-MOI, VOITURE, BEAUCOUP, ACHETER, CAFÉ |

- **Interprète** : `avatar_animation/sigml_blender.py`, un interprète maison qui
  n'utilise aucun fichier de JASigning. Il lit un sous-ensemble des balises HamNoSys,
  construit un avatar simple et l'anime :
  - un calcul de coude pour les bras ;
  - une pose de doigts par configuration ;
  - affichage de la glose à l'écran ;
  - sorties `.mp4` par signe, plus un enchaînement en `.mp4` et `.glb`.
- **Corrections faites en regardant les rendus** :
  1. les maillages se détachaient du squelette : il fallait les attacher en position
     de repos ;
  2. le coude partait vers le haut : angle de la cible du coude de la cinématique
     inverse (« pole ») réglé à 90° ;
  3. la main pointant vers l'avant rentrait dans le torse : quand les doigts pointent
     loin du corps, c'est la base de la main qui va à l'emplacement ;
  4. les mains de MAISON ne se rejoignaient pas : symétrie et `hamtouch` en espace
     neutre placent les mains au centre ;
  5. les murs de MAISON descendaient trop bas : mouvement réduit ;
  6. avec les signes près du visage (VOIR, BOIRE, PENSER), le coude se levait
     encore et l'avant-bras traversait la tête. La cinématique inverse de Blender a
     été remplacée par un calcul direct du coude (coude vers le bas, longueurs du
     bras conservées). Le poignet est placé un peu devant l'emplacement quand les
     doigts pointent vers l'avant. PENSER arrive maintenant par le côté de la tête.
  7. lot de 50 signes : vérification contre la DTD officielle. `hampalmi` et
     `hamarco` n'existent pas en HamNoSys, car la paume et les arcs n'ont que
     u/d/l/r, relatifs à la direction des doigts ou du mouvement. L'interprète
     applique maintenant cette lecture relative pour les doigts pointés sur le
     côté et pour les balayages latéraux. POURQUOI, PARLER, BOIRE, PENSER et SOIR
     ont été réorientés pour que le coude ne monte pas au-dessus de l'épaule.

Un **pilote de validation** de ces 50 signes est prêt (page de relecture et CSV) :
voir `dataset_generation/README.md`.

```bash
/Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \
    --python src/sign_language_3d_avatar/avatar_animation/sigml_blender.py -- \
    src/sign_language_3d_avatar/dataset_generation/sigml
```

**Limites**
- Pas de visage. Animation raide, distances et durées réglées à l'œil.
- Seul un sous-ensemble des balises est géré.
- L'orientation de la paume suit une convention propre au script (doigts verticaux :
  `hampalmu` = vers l'avant, `hampalmd` = vers soi). Elle reste à confronter à
  JASigning.

---

## Suites envisagées

1. **Faire valider** les brouillons SiGML par un signeur LSF, idéalement en lui
   montrant le rendu. Commencer par une vingtaine de gloses pour mesurer le taux
   d'erreur et le temps de relecture.
2. **Enregistrer un signeur LSF** pour les gloses les plus fréquentes : webcam, puis
   extraction MediaPipe dans le même format que LSFB. `retarget_blender.py`
   fonctionnerait tel quel, et les droits appartiendraient au projet (avec l'accord
   écrit de la personne filmée).
3. **Contacter** les autrices de LSF-ANIMAL et l'équipe AZee pour un accès dans un
   cadre de recherche.
4. **Brancher** l'animation sur le pipeline : un `SiGMLAvatarRenderer` qui implémente
   `AvatarRendererInterface` (`avatar_adapter/`) et fait jouer chaque glose reçue.
