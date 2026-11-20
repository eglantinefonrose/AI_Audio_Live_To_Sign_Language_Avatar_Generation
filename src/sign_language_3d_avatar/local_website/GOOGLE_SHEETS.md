# Recevoir les avis des utilisateurs dans un tableau Google Sheets

Quand un utilisateur clique sur « Enregistrer et envoyer mon avis », l'outil envoie son avis à
un petit programme Google (Apps Script) relié à **votre** tableau : une ligne par signe jugé
(phrase, signe, verdict, problèmes cochés, commentaires, étiquette LSFB et clip copié). Sans
cette mise en place, les avis restent sur l'ordinateur de l'utilisateur, qui peut les envoyer
par e-mail.

À faire **une seule fois**, avec votre compte Google (une dizaine de minutes).

## 1. Créer le tableau

1. Ouvrez **sheets.new** (connecté à votre compte Google) : un tableau vide s'ouvre.
2. Nommez-le, par exemple **« Avis Avatar LSFB »** (en haut à gauche).

## 2. Ajouter le programme

1. Dans le tableau : menu **Extensions → Apps Script**. Un éditeur s'ouvre dans un nouvel
   onglet.
2. Effacez le contenu du fichier `Code.gs`, puis collez **tout** le contenu de
   [`google_sheets/avis.gs`](google_sheets/avis.gs).
3. Cliquez sur l'icône **Enregistrer** (disquette).

## 3. Le publier comme « application Web »

1. En haut à droite : **Déployer → Nouveau déploiement**.
2. À côté de « Sélectionner le type », cliquez sur la roue dentée → **Application Web**.
3. Réglez :
   - **Description** : `Avis Avatar LSFB` ;
   - **Exécuter en tant que** : **Moi** (votre adresse) ;
   - **Qui a accès** : **Tout le monde**. Indispensable : les utilisateurs de l'outil n'ont
     pas de compte Google à fournir.
4. Cliquez sur **Déployer**, puis **Autoriser l'accès** et choisissez votre compte.
5. Google affiche « Google n'a pas validé cette application » : c'est normal pour un programme
   que vous avez écrit vous-même. Cliquez sur **Paramètres avancés**, puis sur **Accéder à …
   (non sécurisé)**, puis **Autoriser**. Le programme demande seulement à écrire dans ce
   tableau.
6. Copiez l'**URL de l'application Web**. Elle ressemble à
   `https://script.google.com/macros/s/AKfy…/exec`.

**Vérification** : collez cette URL dans votre navigateur. La page doit afficher
`{"ok":true,"service":"avis Avatar LSFB"}`.

## 4. Donner l'URL à l'outil

Ouvrez le fichier [`config.json`](config.json) de ce dossier avec un éditeur de texte et
collez l'URL entre les guillemets :

```json
{
  "url_avis": "https://script.google.com/macros/s/AKfy…/exec"
}
```

Relancez l'outil (`Avatar LSFB.command`) : la fenêtre doit afficher
« Avis : envoyés au tableau de l'équipe », et le bouton de la page devient
« Enregistrer et envoyer mon avis ». Envoyez un avis de test : une ligne apparaît dans
l'onglet **Avis** du tableau.

Refaites ensuite l'archive (`bash fabriquer_archive.sh`) : `config.json` y est copié, les
utilisateurs enverront donc leurs avis à votre tableau.

## 5. Être prévenu des nouveaux avis

Dans le tableau : **Outils → Règles de notification** (ou « Paramètres de notification ») →
**Toute modification apportée** → **Par e-mail, une fois par jour** (ou immédiatement).

## À savoir

- **Modifier le programme plus tard** : après avoir changé le code, utilisez **Déployer →
  Gérer les déploiements → ✏️ (modifier) → Version : Nouvelle version → Déployer**. L'URL ne
  change pas. Un « Nouveau déploiement » créerait une autre URL.
- **Qui peut écrire dans le tableau ?** Toute personne qui connaît l'URL, puisqu'elle est
  incluse dans l'outil distribué. Le programme ne fait qu'ajouter des lignes : personne ne
  peut lire, modifier ni supprimer vos données par cette URL. Si des envois indésirables
  arrivent, il suffit de faire un nouveau déploiement (nouvelle URL) et de mettre à jour
  `config.json`.
- **Ce qui est envoyé** : la phrase, les réponses et les commentaires de l'utilisateur, la
  date. Ni nom, ni adresse e-mail, ni adresse IP dans le tableau.
- **Limites gratuites** de Google : largement suffisantes pour des avis (plusieurs milliers
  d'envois par jour).
