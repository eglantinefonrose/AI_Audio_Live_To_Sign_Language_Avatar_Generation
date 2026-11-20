<!--
Pour les développeurs (ce commentaire est retiré de la notice copiée dans l'archive).
Outil : serveur.py (serveur local), index.html (la page), lanceurs Installer.* et
« Avatar LSFB.* » ; ceux de racine_archive/ sont mis à la racine de l'archive et appellent
ceux de ce dossier. Archive : bash fabriquer_archive.sh -> dist/avatar-lsfb.zip (ignoré par
git : contient le dictionnaire dérivé de LSFB et l'avatar). Lanceurs Windows non testés.
Avis envoyés au tableau Google Sheets de config.json (url_avis) : mise en place dans
GOOGLE_SHEETS.md ; tant que url_avis est vide, l'étape 9 propose seulement l'e-mail.
Avant de publier l'archive (Release GitHub), réexporter avatar_sign_4.vrm depuis VRoid Studio
en autorisant la redistribution et l'usage par tout le monde (aujourd'hui : auteur seulement,
redistribution interdite).
-->

# Avatar LSFB : faire signer une phrase par l'avatar

Cet outil fabrique une **vidéo dans laquelle un avatar signe une phrase en langue des signes
de Belgique francophone (LSFB)**. Vous écrivez la phrase, l'outil vérifie les mots, puis crée
la vidéo. Vous pouvez la regarder et l'enregistrer sur votre ordinateur.

Tout se passe **sur votre ordinateur** : rien n'est envoyé sur Internet.

## Ce qu'il vous faut

- Un ordinateur **Mac** (macOS 12 ou plus récent) ou **Windows** (10 ou 11 ; version Windows
  pas encore testée).
- Environ **3 Go d'espace libre**.
- Une connexion Internet, **seulement pendant l'installation**.

## 1. Installer Blender (une seule fois)

Blender est un logiciel gratuit de 3D. L'outil s'en sert pour fabriquer la vidéo : vous
n'aurez jamais à l'ouvrir vous-même.

1. Allez sur **blender.org** et cliquez sur **Download**.
2. Ouvrez le fichier téléchargé :
   - **Mac** : faites glisser l'icône Blender dans le dossier **Applications** ;
   - **Windows** : double-cliquez sur le fichier et suivez l'installation, en laissant les
     choix proposés.

## 2. Installer l'outil (une seule fois)

1. Téléchargez le fichier **`avatar-lsfb.zip`** depuis la page des versions du projet :
   **github.com/eglantinefonrose/AI_Audio_Live_To_Sign_Language_Avatar_Generation/releases**.
2. Double-cliquez dessus pour le décompresser. Un dossier **`avatar-lsfb`** apparaît.
3. Dans ce dossier (il contient aussi cette notice), double-cliquez sur :
   - **Mac** : `Installer.command`. Si un message dit que le fichier vient d'Internet,
     faites un clic droit sur le fichier → **Ouvrir** → **Ouvrir** ;
   - **Windows** : `Installer.bat`. Si Windows affiche « Windows a protégé votre
     ordinateur », cliquez sur **Informations complémentaires** → **Exécuter quand même**.
4. Une fenêtre (Terminal sur Mac) s'ouvre et installe ce qui manque. Attendez le message
   **« Installation terminée »**, ce qui peut prendre quelques minutes, puis appuyez sur
   **Entrée** (Mac) ou sur une touche (Windows) : la fenêtre se ferme ou peut être fermée.

## 3. Lancer l'outil

Dans le dossier `avatar-lsfb`, double-cliquez sur **`Avatar LSFB.command`** (Mac) ou
**`Avatar LSFB.bat`** (Windows). La première fois, le même message de sécurité qu'à l'étape 2
peut apparaître : faites de même.

- Une fenêtre (Terminal sur Mac) s'ouvre : **laissez-la ouverte** tant que vous utilisez
  l'outil, c'est elle qui fait le travail.
- Votre navigateur (Safari, Chrome, Edge…) affiche la page de l'outil. Si elle ne s'ouvre pas
  toute seule, tapez dans la barre d'adresse l'adresse affichée dans la fenêtre (en général
  **`http://localhost:8000`**).

## 4. Écrire la phrase en gloses

L'avatar ne lit pas le français directement : il lit des **gloses**. Une glose est un mot en
MAJUSCULES qui représente un signe, écrit dans l'ordre de la langue des signes.

| Phrase en français | Phrase en gloses |
|---|---|
| Ce matin, mon frère a acheté une voiture rouge. | `MATIN FRÈRE MOI VOITURE ROUGE ACHETER` |
| Tu veux un café ? | `TOI CAFÉ VOULOIR QUESTION` |
| Je ne comprends pas cette phrase. | `MOI PHRASE COMPRENDRE NON` |

Les règles principales :

- **Majuscules**, mots séparés par des espaces.
- **Pas de petits mots** : pas de « le », « de », « à », « est »…
- **Le moment d'abord**, puis le lieu, la personne, l'objet, et **le verbe à la fin**. Le verbe
  est toujours à l'**infinitif** (ACHETER, pas « a acheté »).
- **Je, tu, il** s'écrivent **MOI, TOI, LUI** (et NOUS, VOUS, EUX).
- **Négation** : **NON** après le verbe. **Question** : le mot de question à la fin (QUOI, OÙ,
  QUAND…), ou **QUESTION** pour une question à laquelle on répond par oui ou non.

## 5. Vérifier les mots

Tapez votre phrase dans la case. Sous chaque mot, un symbole apparaît **tout de suite** :

| Symbole | Signification |
|---|---|
| ✅ | le signe existe, l'avatar le fera |
| 👉 | l'avatar pointera (MOI, TOI, LUI…) |
| 🔤 | l'avatar épellera le mot lettre par lettre (noms propres, mots techniques) |
| ❌ | pas de signe pour ce mot : il sera affiché en texte. Essayez un autre mot si l'outil en propose un |

## 6. Générer la vidéo

1. Choisissez si le mot signé doit s'écrire en haut à gauche de la vidéo : case
   **« Afficher le mot signé… »**, cochée par défaut. Décochée, la vidéo ne montre que
   l'avatar ; seuls les mots sans signe restent écrits.
2. Cliquez sur **« Générer la vidéo »**.
3. Patientez, en général **de quelques secondes à une minute**. Une barre indique la
   progression.

## 7. Regarder l'aperçu

La vidéo apparaît dans la page. Si la case est cochée, le mot signé s'affiche en haut à
gauche de l'image. Cliquez sur
**lecture** pour la revoir autant de fois que vous le voulez.

## 8. Télécharger la vidéo

Cliquez sur **« Télécharger »**. La vidéo (fichier **`.mp4`**) est enregistrée dans votre
dossier **Téléchargements**. Vous pouvez la lire avec n'importe quel lecteur vidéo, et la
partager dans le respect de la licence (voir plus bas).

## 9. Donner votre avis sur les signes (facultatif, mais très utile)

Sous la vidéo, la section **« 4. Votre avis sur les signes »** liste chaque signe de la phrase.

1. Cliquez sur **▶** devant un signe pour le revoir seul.
2. Dites s'il est **✅ Correct**, **⚠️ À revoir** ou **❌ Mauvais**. Cliquez à nouveau pour
   annuler.
3. S'il n'est pas correct, cochez ce qui ne va pas (forme de la main, emplacement,
   mouvement, orientation de la paume, mauvais signe…) et, si possible, précisez ce qu'il
   faudrait changer.
4. Ajoutez si vous le voulez un commentaire sur la phrase entière, puis cliquez sur
   **« Enregistrer et envoyer mon avis »**.

Votre avis est gardé sur votre ordinateur et **envoyé à l'équipe du projet** pour l'aider à
corriger l'avatar (connexion Internet nécessaire). Il contient la phrase, vos réponses et vos
commentaires, mais ni votre nom ni votre adresse.

Si l'envoi échoue (pas de connexion Internet, par exemple), la page vous propose
**« ✉️ Envoyer cet avis par e-mail »** : votre messagerie s'ouvre avec un message déjà rédigé
pour **feedback@aispeechtosignlanguage.fr**. Si aucune messagerie ne s'ouvre (webmail),
écrivez directement à cette adresse. En bas de la page, **« Télécharger tous les avis
(CSV) »** récupère tous vos avis dans un fichier qui s'ouvre dans un tableur.

## 10. Fermer l'outil

Fermez l'onglet du navigateur, puis la fenêtre ouverte à l'étape 3.

## En cas de problème

| Problème | Que faire |
|---|---|
| La page ne s'ouvre pas | Vérifiez que la fenêtre de l'étape 3 est ouverte, puis tapez dans le navigateur l'adresse qu'elle affiche (en général `http://localhost:8000`) |
| « Avatar LSFB est déjà ouvert » | L'outil tourne déjà dans une autre fenêtre : utilisez la page déjà ouverte, et fermez cette fenêtre-ci |
| « Blender introuvable » | Installez Blender (étape 1). Sur Mac, il doit être dans le dossier **Applications** |
| Un mot est marqué ❌ | Ce signe n'existe pas encore dans l'outil : reformulez avec un autre mot |
| La vidéo ne se crée pas | Fermez la fenêtre de l'étape 3 et le navigateur, puis relancez l'outil (étape 3) |

## Licence et crédits

- Les mouvements des signes viennent du **corpus LSFB-ISOL (Université de Namur)**, sous
  licence **CC BY-NC-SA 4.0**. Vous pouvez partager les vidéos produites **gratuitement**,
  **en citant la source** et **sous la même licence**. Elles ne doivent **jamais** être
  utilisées dans un but commercial.
- Crédit à indiquer avec chaque vidéo partagée : *« Avatar : Eglantine Fonrose (VRoid Studio) —
  signes : corpus LSFB (Université de Namur), CC BY-NC-SA 4.0 »*.
- Cet outil est un projet expérimental. Il ne remplace pas un interprète ni une personne qui
  signe : faites relire les vidéos importantes par une personne qui connaît la LSFB.
