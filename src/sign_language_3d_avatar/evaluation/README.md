# Évaluation de l'avatar

Question posée : **l'avatar signe-t-il bien des phrases déjà en gloses LSFB ?** La
traduction et la transcription n'interviennent pas : les phrases du banc d'essai sont des
gloses de référence (par défaut les 74 phrases de `src/eval/dataset.jsonl` et les 45 de
`src/eval/dev.jsonl`, figées par hash).

```bash
uv run python src/sign_language_3d_avatar/evaluation/evaluate_avatar.py                 # mannequin
uv run python src/sign_language_3d_avatar/evaluation/evaluate_avatar.py \
  --avatar src/sign_language_3d_avatar/avatar_animation/data/lsfb_avatar.vrm --save --label "…"
```

Durée : ~2 min la première fois (galerie de vrais signeurs, mise en cache), puis ~5 min
(reconnaissance). Blender n'est relancé que si l'avatar, le dictionnaire, le banc ou le code
changent. `--save` archive le run : JSON dans `results/` et ligne dans
[`RESULTS.md`](RESULTS.md). Évite de le lancer pendant un entraînement : Blender et la
reconnaissance chargent la machine.

## Comment ça marche

1. **L'avatar joue** chaque signe du dictionnaire (isolé) et chaque phrase du banc dans
   Blender, avec le même transfert que pour la vidéo (`export_avatar_blender.py`). On relève
   la position **réelle** de ses os, image par image : épaules, coudes, poignets, 21 points
   par main. Pas de vidéo ni de MediaPipe : la détection des mains sur un avatar stylisé
   serait elle-même une source d'erreur.
2. **Un reconnaisseur** (`features.py`) compare chaque signe à une galerie d'exemples
   **réels** de LSFB-ISOL (~9 signeurs par signe, 330 signes) : descripteurs indépendants
   de la taille du corps (directions des bras dans le plan de l'image, orientation et
   configuration des mains, main dominante comptée double), alignement temporel par DTW,
   plus proche exemple par signe.
3. **Pas de triche** : le signeur du clip copié par l'avatar est retiré de la galerie,
   sinon on mesurerait seulement que l'avatar ressemble à son clip.
4. **Référence** : le même test sur le clip source, joué par le vrai signeur. L'écart
   avatar / clip source est ce que perdent le transfert et l'enchaînement en phrase.

## Les mesures

| Mesure | Question | Lecture |
|---|---|---|
| **Score** (chiffre de tête) | l'avatar est-il aussi reconnaissable que le signeur qu'il copie, une fois les signes enchaînés en phrase ? | (centile en contexte − 50 %) ÷ (centile des clips sources − 50 %). 100 % = aussi reconnaissable que le vrai signeur ; 0 % = hasard. |
| **Phrases** | la phrase **entière** est-elle bien signée ? | **jouables** : % de phrases dont chaque glose a un vrai signe ou un pointage (aucun mot sans signe). **Reconnues au seuil 80 % / 90 %** : jouables, et chaque vrai signe, en contexte, a un rang centile ≥ 80 % (top 20 %) ou ≥ 90 %. Référence : la même phrase avec les clips sources joués par les vrais signeurs. |
| Couverture | chaque glose du banc est-elle jouée par un vrai signe ? | % signes réels / pointages générés / mots sans signe |
| Reconnaissance isolée | un signe seul de l'avatar est-il reconnu ? | rang centile, top-1 / 5 / 10 |
| Reconnaissance en contexte | et une fois dans une phrase ? On connaît le début et la fin exacts de chaque signe (c'est nous qui générons la chronologie) : on découpe et on reconnaît chaque morceau | idem |
| Qualité du mouvement | vitesses ou à-coups anormaux, main enfoncée dans le torse | % d'images sans défaut ; seuils = 99,5e centile des vrais signeurs |
| Fidélité | l'avatar reproduit-il son clip source ? | écart angulaire médian bras / doigts (degrés) |

**Rang centile** : part des autres signes que le bon signe devance (100 % = classé
premier, 50 % = hasard). Le top-1 est bas même pour de vrais signeurs (~10 % sur 330
signes : clips de 0,5 s tirés de conversations, ~9 exemples par signe) ; le centile,
continu, reste informatif et se mesure de la même façon pour l'avatar et les humains.

Le rapport liste aussi :
- les signes **perdus par l'avatar** (clip source nettement mieux reconnu que l'avatar) :
  problème de transfert ou de transition ;
- les **clips sources à changer** (mal reconnus alors qu'un autre signeur du même signe
  l'est bien) : à corriger dans `dataset_generation/select_examples.py`.

## Contrôles et limites

Contrôles de sensibilité sur 80 signes (avatar dégradé exprès, rang centile) : tel quel
88 %, doigts figés 80 %, **bras figés 87 %**, mauvais signe 42 %. La mesure repère un mauvais
signe et une configuration de main perdue, mais elle est **presque aveugle au mouvement
des bras** : même sur de vrais signeurs, figer les bras ne coûte que 0,4 point (clips
courts, trajectoires peu discriminantes). Le mouvement des bras est donc surveillé par la
**fidélité** et la **qualité du mouvement**, pas par le score.

Autres limites :
- les pointages (MOI, TOI…) et les mots sans signe ne sont pas reconnaissables par la
  galerie : ils ne comptent que dans la couverture ;
- la « main dans le torse » utilise un torse approximatif (cylindre sous les épaules) :
  à lire par rapport à la valeur des vrais signeurs (6,6 %), pas dans l'absolu ;
- un reconnaisseur n'est pas un signeur : seul un panel de signeurs LSFB sourds valide la
  compréhension. La prochaine étape utile serait de vérifier que ce score suit leur
  jugement sur une trentaine de phrases.

Pistes pour un reconnaisseur plus fort : télécharger plus d'exemples par signe (la
galerie n'en garde que 10 par glose) ou entraîner un classifieur de poses sur LSFB-ISOL.

## Premiers résultats (dictionnaire `7913fcd77c66`, banc `214ff0d9f655`)

| Avatar | Score | Phrases jouables | Phrases reconnues 80 % (avatar / vrais signeurs) | Centile contexte / clips sources | Mouvement sans défaut | Fidélité bras / doigts |
|---|---|---|---|---|---|---|
| mannequin | 100 % | 51 % | 29 % / 29 % | 89 % / 89 % | 89 % (main dans le torse 10,8 %) | 0° / 2° |
| VRoid (`lsfb_avatar.vrm`) | 100 % | 51 % | 29 % / 29 % | 89 % / 89 % | 95 % (main dans le torse 3,8 %) | 0° / 3° |

**Par phrase**, une phrase sur deux seulement est jouable en entier : il suffit d'un mot
sans signe (QUAND, UTILISER, AMÉLIORER, DATASET, GPU…). Parmi les phrases jouables,
l'avatar fait jeu égal avec les vrais signeurs (29 % / 29 % au seuil 80 %) : la perte
vient de la couverture et du choix des clips (ex. POUVOIR, mal reconnu même sur le
signeur filmé), pas du transfert. Les lignes de RESULTS.md antérieures à la mesure par
phrase ont été complétées à partir de leurs records (`par_phrase`), sans relancer Blender.

Les deux avatars reproduisent fidèlement leurs clips : rien n'est perdu au transfert ni
aux transitions. Le point faible est en amont, dans le **choix des clips** : 12 clips
sources sont mal reconnus alors qu'un autre signeur l'est bien (NOIR, PORTER, IDEE,
VIDEO, 16, BESOIN…). Couverture du banc : 63 % de signes réels, 20 % de pointages, 17 %
de mots sans signe.

## Fichiers

| Fichier | Rôle |
|---|---|
| `evaluate_avatar.py` | point d'entrée : galerie, lancement de Blender, mesures, rapport, archivage |
| `export_avatar_blender.py` | dans Blender : joue signes et phrases sur l'avatar, relève ses articulations |
| `features.py` | descripteurs, DTW, galerie, rang centile, écart angulaire |
| `results/`, `RESULTS.md` | runs archivés (étiquettes et scores seulement) |

Caches (dérivés de LSFB, CC BY-NC-SA 4.0, ignorés par git) :
`avatar_animation/data/dictionnaire/evaluation/`.
