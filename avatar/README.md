# Sous-ensemble LSFB pour l'avatar

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
  - Les données vont dans `avatar/data/`, ignoré par git. Ne pas les commiter.

## Utilisation

```bash
uv run python avatar/lsfb_subset.py map
```

Cette commande télécharge les index (~12 Mo) et écrit `avatar/lsfb_mapping.csv` :
glose projet → étiquette LSFB, statut, nombre d'instances, autres variantes, note.

```bash
uv run python avatar/lsfb_subset.py download --words-file avatar/lsfb_words.txt --dry-run
```

`--dry-run` affiche, glose par glose, le nombre d'instances et la taille estimée
(requêtes HEAD sur un échantillon, extrapolées à la durée), sans rien télécharger.

```bash
uv run python avatar/lsfb_subset.py download --words-file avatar/lsfb_words.txt
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

**Manifeste** : `avatar/data/lsfb/manifest.csv`, avec pour chaque ligne la glose,
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

`avatar/lsfb_words.txt` contient la liste proposée : les gloses sûres sont actives,
celles à valider sont commentées. Pour régénérer la table après une mise à jour de
`lexicon.txt` ou de `eval/dataset.jsonl`, relancer `map`.
