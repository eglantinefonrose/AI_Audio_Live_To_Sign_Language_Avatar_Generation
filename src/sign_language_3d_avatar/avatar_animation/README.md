# Animation de l'avatar

Scripts Blender qui produisent les animations (sorties dans `src/sign_language_3d_avatar/avatar_animation/data/`,
ignoré par git). Les données d'entrée viennent de `src/sign_language_3d_avatar/dataset_generation/`.

## Dictionnaire de signes LSFB

Chaîne entièrement automatique, des clips LSFB-ISOL jusqu'à un avatar qui signe des
phrases. Tout ce qu'elle produit va dans `data/dictionnaire/` (ignoré par git, dérivé de
LSFB : CC BY-NC-SA 4.0).

| Étape | Script | Ce qu'il fait |
|---|---|---|
| 1. Choix | `dataset_generation/select_examples.py` | parmi ~10 exemples par signe, garde le meilleur : mains détectées (poses brutes), tremblement, durée proche de la médiane → `selection.csv` |
| 2. Nettoyage | `clean_signs.py` | droitier (clips gauchers retournés), sauts corrigés, lissage léger, immobilités du début et de la fin coupées, articulations 3D → `signes/*.npz` |
| 3. Index | `sequence.py index` | chaque glose du projet → signe, pointage généré ou mot affiché → `index.json` |
| 4. Avatar | `build_dictionary_blender.py` | une action Blender par signe sur l'avatar → `dictionnaire_<avatar>.blend` et `.glb` |
| 5. Phrase | `demo_phrase_blender.py` (via `tester_phrase_lsfb.py`) | repos, transitions de 0,24 s, signes, pointages, sous-titres → `.mp4` |

```bash
# poses (nettoyées + brutes) de 10 exemples par signe (≈ 30 Mo)
uv run python src/sign_language_3d_avatar/dataset_generation/lsfb_subset.py download --words-file src/sign_language_3d_avatar/dataset_generation/lsfb_words.txt --max-par-glose 10 --landmarks pose,left_hand,right_hand --raw
uv run python src/sign_language_3d_avatar/dataset_generation/select_examples.py
uv run python src/sign_language_3d_avatar/avatar_animation/clean_signs.py
uv run python src/sign_language_3d_avatar/avatar_animation/sequence.py index
/Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup --python src/sign_language_3d_avatar/avatar_animation/build_dictionary_blender.py -- [--avatar mon_avatar.vrm]
uv run python src/sign_language_3d_avatar/tester_phrase_lsfb.py --gloses "MOI MAISON MERCI" [--avatar mon_avatar.vrm]
```

**Transfert sur l'avatar** (`avatar_rig.py`) : chaque os humanoïde est orienté d'après
la *direction* entre deux articulations (mains, torse et tête par deux vecteurs pour le
roulis). Le résultat ne dépend donc ni de la taille ni des proportions de l'avatar.

**Avatar VRoid Studio** : dans VRoid Studio, *Exporter → Exporter en VRM* (VRM 1.0 ou
0.x), puis passer le fichier avec `--avatar`. Aucune extension Blender n'est nécessaire :
le `.vrm` est importé comme un glTF et la table des os humanoïdes est lue dans le
fichier ; les modèles VRM 0.x (tournés vers l'arrière) sont remis face caméra. Les
matériaux MToon sont simplifiés (rendu Workbench), suffisant pour des démos.

**Ce qui est généré ou manque** :
- pointages `MOI`, `TOI`, `LUI` (fixes) et `NOUS`, `VOUS`, `EUX` (balayage) : générés, main
  droite index tendu (LSFB-ISOL ne fournit pas de clip de pointage) ;
- verbes directionnels (`AIDER-À-TOI`) : forme de base du verbe, sans orientation ;
- 173 gloses sans signe (termes techniques, noms propres, synonymes non retenus) : mot
  affiché en sous-titre, mains au repos. LSFB-ISOL n'a pas d'alphabet manuel complet,
  donc pas d'épellation ;
- pas de visage dans cette version.

**Limites** : clips tirés de conversations spontanées (gestes parfois bas ou réduits),
mains MediaPipe imparfaites, et signes en **LSFB**, pas en LSF de France.

## Retargeting des poses LSFB (`retarget_blender.py`)

Il fait jouer les poses MediaPipe téléchargées par `src/sign_language_3d_avatar/dataset_generation/lsfb_subset.py` à un squelette Blender (41 os, doigts
compris). Pour chaque exemple, il exporte `.glb`, `.bvh` et un aperçu `.mp4` dans
`src/sign_language_3d_avatar/avatar_animation/data/retarget/<GLOSE>/` :

```bash
/Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \
    --python src/sign_language_3d_avatar/avatar_animation/retarget_blender.py -- --words MAISON,MERCI --vues face,profil
```

La profondeur des bras et de la tête est reconstruite, car celle de MediaPipe n'est
pas fiable. Les clips viennent de conversations spontanées en LSFB : ce ne sont pas
des signes de dictionnaire.

## Animation à partir de SiGML (`sigml_blender.py`) — prototype

SiGML est le format XML de la notation HamNoSys, lu par l'avatar CWASA/JASigning de
l'UEA (https://vh.cmp.uea.ac.uk/index.php/CWASA). Les noms de balises suivent la
DTD officielle `sigmlhns.dtd`. Ce script est **un interprète maison** (ce n'est pas
JASigning) : il anime dans Blender un sous-ensemble des balises. Le coude est
calculé directement (bras à deux segments, coude vers le bas) et les doigts prennent
une pose par configuration.

```bash
/Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \
    --python src/sign_language_3d_avatar/avatar_animation/sigml_blender.py -- src/sign_language_3d_avatar/dataset_generation/sigml [--gloses MERCI,TOI]
```

Avec `--phrase "TOI DONNER-À-MOI MERCI" --sortie chemin.mp4`, le script rend une
seule vidéo, avec les gloses dans l'ordre donné (répétitions permises). C'est ce
qu'utilise `src/sign_language_3d_avatar/tester_phrase_sigml.py`.

Sorties dans `src/sign_language_3d_avatar/avatar_animation/data/sigml/` : un `.mp4` par signe, plus `_enchainement.mp4`
et `_enchainement.glb` (tous les signes à la suite).

- **Lexiques** : le script prend des fichiers `.sigml` ou des dossiers (tous leurs
  `.sigml`). `src/sign_language_3d_avatar/dataset_generation/sigml/` contient 50
  signes, qui sont des **brouillons rédigés par une IA, non validés** par un
  locuteur LSF.
- **Balises gérées** :
  - configurations `hamflathand`, `hamfist`, `hamfinger2`, `hamfinger23(spread)`,
    `hamfinger2345`, `hampinch12`, `hampinchall`, `hamcee12`, `hamceeall`, et les
    modificateurs de pouce et de flexion ;
  - directions des doigts `hamextfinger*` et de la paume `hampalm*` ;
  - emplacements `hamchin`, `hamforehead`, `hameyes`, `hamchest`, `hamneutralspace`…, avec
    `hamtouch` / `hamclose` ;
  - mouvements `hammove*`, `hamarc*`, `hamcircle*`, `hamsmallmod` / `hamlargemod`,
    `hamfast` / `hamslow`, `hamrepeat*` (`hamrepeatreverse` = aller-retour),
    `hamreplace` ;
  - symétrie `hamsymmlr` / `hamsymmpar`.

  Les autres balises sont ignorées, et le visage (non-manuel) n'est pas animé.
- **Conventions propres à ce script, à confronter à JASigning** :
  - main dominante = droite ;
  - orientation de la paume relative aux doigts : doigts verticaux, `hampalmu` =
    vers l'avant et `hampalmd` = vers soi ; doigts vers la gauche du signeur,
    `hampalml` = vers soi et `hampalmr` = vers l'avant (et inversement vers la droite) ;
  - arc `hamarcu` sur un balayage latéral = arc bombé vers l'avant ;
  - doigts vers l'avant : le poignet est placé un peu devant l'emplacement ; sinon,
    c'est le bout des doigts qui va à l'emplacement ;
  - `hamneutralspace` + `hamtouch` en symétrie = mains jointes au centre ;
  - distances et durées fixées à la main (25 images/s).
- **Licence de CWASA/JASigning** : copyright UEA, utilisable sans modification
  (conditions sur https://vh.cmp.uea.ac.uk/index.php/JASigning_Conditions). Ce
  script n'en utilise aucun fichier.
