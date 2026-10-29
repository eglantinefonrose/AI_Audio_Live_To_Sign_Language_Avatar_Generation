# Animation de l'avatar

Scripts Blender qui produisent les animations (sorties dans `src/sign_language_3d_avatar/avatar_animation/data/`,
ignoré par git). Les données d'entrée viennent de `src/sign_language_3d_avatar/dataset_generation/`.

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
