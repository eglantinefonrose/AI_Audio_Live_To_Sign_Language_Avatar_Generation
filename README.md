# Un avatar qui signe en langue des signes

Ce projet transforme **la parole en langue des signes**, jouée par un **personnage 3D**
(un avatar). Une personne parle ; quelques instants plus tard, l'avatar signe ce qu'elle a dit.

Le but : rendre accessibles aux personnes sourdes et malentendantes des contenus qui ne sont
aujourd'hui que parlés (vidéos, conférences, annonces…), quand aucun interprète n'est
disponible.

> 🧑‍💻 Vous êtes développeur ou développeuse ? La documentation technique est dans
> [README-tech.md](README-tech.md).

## Comment ça marche, en trois étapes

1. **De la parole au texte.** Un programme de reconnaissance vocale écoute le micro (ou une
   vidéo) et écrit ce qui est dit, phrase par phrase.
2. **Du texte aux gloses.** Une intelligence artificielle, qui tourne sur l'ordinateur sans
   rien envoyer sur Internet, traduit chaque phrase en **gloses**.
3. **Des gloses aux signes.** L'avatar joue les signes correspondant aux gloses, l'un après
   l'autre, avec des transitions naturelles entre eux.

### C'est quoi, une glose ?

La langue des signes n'est pas du français « mimé » : elle a son propre vocabulaire et sa propre
grammaire. Une **glose** est un mot écrit en MAJUSCULES qui représente un signe, placé dans
l'ordre de la langue des signes. Par exemple :

| En français | En gloses |
|---|---|
| Ce matin, mon frère a acheté une voiture rouge. | `MATIN FRÈRE MOI VOITURE ROUGE ACHETER` |
| Je ne comprends pas cette phrase. | `MOI PHRASE COMPRENDRE NON` |

Le moment vient d'abord, le verbe à la fin, la négation après le verbe, et les petits mots
(« le », « de », « à »…) disparaissent.

### Pourquoi la langue des signes de Belgique (LSFB) ?

Pour apprendre à l'avatar à signer, il faut des vidéos de vraies personnes qui signent. Le
projet utilise le **corpus LSFB** de l'Université de Namur : des milliers de vidéos de signeurs
en **langue des signes de Belgique francophone**. L'avatar signe donc en LSFB, proche de la
langue des signes française (LSF) mais pas identique.

Les mouvements de chaque signe sont relevés sur ces vidéos, nettoyés, puis transférés sur
l'avatar. Quand un mot n'a pas de signe connu, l'avatar l'épelle avec l'alphabet manuel (noms
propres, mots techniques) ; les pronoms (moi, toi, lui…) sont des gestes de pointage.

## L'outil « Avatar LSFB » : faire signer une phrase

Un outil simple permet de **faire signer une phrase par l'avatar et d'en récupérer la vidéo**,
sans rien connaître à la programmation :

1. on écrit la phrase en gloses dans une page web ;
2. la page indique, mot par mot, si l'avatar sait le signer (✅ signe, 👉 pointage,
   🔤 épelé, ❌ pas de signe) et propose parfois un autre mot ;
3. on clique sur « Générer la vidéo » : quelques secondes plus tard, la vidéo de l'avatar
   apparaît, avec ou sans le mot signé écrit dans un coin ;
4. on la regarde, puis on la télécharge sur son ordinateur ;
5. on peut donner son avis sur chaque signe (correct, à revoir, mauvais, et ce qui ne va
   pas) et nous l'envoyer par e-mail à **feedback@aispeechtosignlanguage.fr** : ces retours
   aident à corriger l'avatar.

Tout se passe sur votre ordinateur, rien n'est envoyé sur Internet. L'outil fonctionne sur Mac
(Windows : en cours de test) et demande d'installer le logiciel gratuit Blender, qui fabrique
la vidéo.

👉 **Mode d'emploi pas à pas, de l'installation à la vidéo :
[NOTICE.md](src/sign_language_3d_avatar/local_website/NOTICE.md)**

> L'archive à télécharger (`avatar-lsfb.zip`) décrite dans la notice n'est pas encore
> publiée. En attendant, l'outil se lance depuis le dossier du projet
> (`src/sign_language_3d_avatar/local_website/`), et l'archive se fabrique avec
> `bash src/sign_language_3d_avatar/local_website/fabriquer_archive.sh`.

## Où en est le projet ?

- **L'avatar sait jouer 426 signes LSFB**, l'alphabet manuel et les pointages. Toutes les
  phrases de notre jeu de test (119 phrases) peuvent être signées en entier.
- **La qualité des signes** est vérifiée de deux façons : un programme compare l'avatar aux
  vraies personnes qui signent, et chaque signe douteux est revu à l'œil.
- **La traduction automatique en gloses** et **la chaîne complète en direct** (parole →
  avatar) sont encore expérimentales.

## Les limites à connaître

- **Ce projet ne remplace pas un interprète.** Faites relire les vidéos importantes par une
  personne qui connaît la LSFB.
- **Le vocabulaire est limité** à ce que contient le corpus : certains mots n'ont pas encore de
  signe et sont seulement écrits à l'écran.
- **Certaines dimensions de la langue des signes ne sont pas encore rendues** : les
  expressions du visage (qui portent une partie de la grammaire), l'utilisation de l'espace
  autour du signeur, le rythme naturel de la conversation.
- L'avatar signe en **LSFB** (Belgique), pas en LSF (France).

## Crédits et licence

- **Signes** : corpus **LSFB-ISOL**, Université de Namur, sous licence
  **CC BY-NC-SA 4.0**. Les vidéos produites par l'avatar en dérivent : elles peuvent être
  partagées **gratuitement**, **en citant la source** et **sous la même licence**, mais
  **jamais dans un but commercial**.
- **Avatar** : Eglantine Fonrose, créé avec VRoid Studio.
- **Reconnaissance vocale** : modèle Kyutai ; **traduction** : modèle de langage local via
  Ollama.
