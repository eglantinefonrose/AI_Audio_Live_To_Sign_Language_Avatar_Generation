"""Page de relecture des gloses SANS signe sûr : choisir, parmi des étiquettes LSFB
proposées automatiquement, celle dont le signe peut jouer la glose du projet.

Gloses relues : celles dont la correspondance automatique (`lsfb_subset.py`) n'est pas
sûre, donc absentes du dictionnaire de l'avatar :
- « absent » : aucune étiquette LSFB du même nom ;
- « variante » : plusieurs signes LSFB pour ce mot (QUAND.1, QUAND.8…), lequel ?
- « incertain » : étiquette ambiguë (OÙ / OU) ;
- « synonyme » : équivalent proposé à la main dans `lsfb_subset.py`, jamais vérifié ;
- « nom-propre » : signe-nom belge (NS:PARIS), ou épellation.
Une glose déjà relue reste sur la page avec sa réponse.

Pour une glose « absent », le script propose jusqu'à 4 candidats, dans cet ordre :
1. synonymes proposés à la main (`SYNONYMES` ci-dessous : TRÈS -> BEAUCOUP…) ;
2. formes proches : lemme (TERMINÉ -> TERMINER), racine commune (BRUITÉ -> BRUIT),
   parties d'une glose composée (ANNÉE-DERNIÈRE -> ANNEE, DERNIER) ;
3. orthographe voisine (difflib), en dernier recours : souvent un faux ami (AGIR -> AIR).
Pour les autres statuts, les candidats sont les étiquettes LSFB trouvées (variantes
comprises). Rien n'est retenu sans relecture : seule la personne qui regarde les vidéos
sait si le signe a le bon sens. `lsfb_subset.py map` reporte ensuite les réponses dans
`lsfb_mapping.csv` (statut « validé » ou « épeler »).

Sur la page, une carte par glose (les plus fréquentes d'abord) : la glose, une phrase
du projet où elle apparaît, avec sa traduction en gloses (la glose relue surlignée), et
un clip de chaque candidat (« autre signeur » pour en voir
un second). Réponses au clavier : 1-4 = ce candidat, A = aucun ne convient, E = à épeler
(nom propre, terme technique), I = incertain, ou une autre étiquette LSFB tapée à la main.

Sorties :
- `lsfb_candidats.csv` (versionné) : une ligne par glose, colonnes `choix`, `verdict`,
  `note`, `date` ; les réponses déjà présentes sont conservées à chaque relance ;
- `data/lsfb/candidats.html` (ignoré par git, à côté des vidéos) : la page. Les réponses
  restent dans le navigateur ; « Exporter le CSV » les rapatrie, à copier à la place de
  `lsfb_candidats.csv`. Une étiquette tapée à la main (« autre ») devient un candidat,
  avec sa vidéo, à la relance suivante.

Les vidéos des candidats (2 signeurs par étiquette) sont téléchargées depuis LSFB-ISOL
(CC BY-NC-SA 4.0) dans `data/lsfb/videos/`, puis copiées en H.264 pour le navigateur.

Relecture ciblée (`--bloquantes`) : seulement les gloses encore sans signe qui empêchent
de jouer une phrase du banc d'essai de l'avatar (`evaluation/banc_avatar.jsonl`), même déjà
relues. Les candidats sont de nouvelles pistes (`PISTES` ci-dessous : synonymes, signes
composés A+B joués à la suite), puis toutes les variantes LSFB du mot, puis les candidats
déjà refusés. La page (`data/lsfb/candidats_bloquants.html`) rappelle la réponse précédente ;
son export contient tout `lsfb_candidats.csv` (les gloses non relues ici gardent leur
réponse) et le remplace comme d'habitude. Une suite « A+B », choisie ou tapée, devient
le statut « décomposé » dans `lsfb_mapping.csv`.

Toutes les gloses sans signe (`--absentes`) : même relecture ciblée, pour les gloses du
projet encore absentes du dictionnaire, banc ou non (celles qui bloquent le banc d'abord).
S'y ajoutent les signes LSFB de la même famille (PLUIE pour PLEUVOIR, FATIGUER pour
FATIGUÉ…) : en langue des signes, nom, verbe et adjectif sont souvent un seul signe.
Page `data/lsfb/candidats_absentes.html`, même export.

Changement de variante (`--variantes`) : les signes pour lesquels aucun clip ne convenait
(verdict « autre variante » dans `choix_clips.csv`, page `revue_clips.py`). Une carte par
glose du projet jouée par ce signe, avec les pistes, toutes les variantes LSFB du mot, et
le signe actuel en dernier. Page `data/lsfb/candidats_variantes.html`, même export.

Usage :
    uv run python src/sign_language_3d_avatar/dataset_generation/candidats_absents.py [--ouvrir] [--sans-video] [--bloquantes | --absentes | --variantes]
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import subprocess
import sys
import urllib.parse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import simplemma

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lsf_comparaison import DATA, GEN, SRC, gloss_frequencies, to_h264  # noqa: E402
from lsfb_subset import SOURCE, LsfbLabels, fetch, load_index, norm, pick, project_glosses  # noqa: E402

sys.path.insert(0, str(GEN.parent / "avatar_animation"))
from sequence import load_index as load_dictionary, resolve  # noqa: E402

CSV_PATH = GEN / "lsfb_candidats.csv"
HTML_PATH = DATA / "candidats.html"
HTML_BLOCKING = DATA / "candidats_bloquants.html"
HTML_VARIANTS = DATA / "candidats_variantes.html"
HTML_ABSENT = DATA / "candidats_absentes.html"
CLIP_CHOICES = GEN / "choix_clips.csv"
SELECTION = GEN / "selection.csv"
BENCH = [GEN.parent / "evaluation" / "banc_avatar.jsonl"]     # banc de l'évaluation de l'avatar
MAX_BLOCKING = 8
FIELDS = ["glose", "frequence", "candidats", "choix", "verdict", "note", "date"]
MAX_CANDIDATES = 4
REVIEW_STATUSES = ("absent", "variante", "incertain", "synonyme", "nom-propre")
CONTEXTE = {
    "absent": "Aucune étiquette LSFB de ce nom : un de ces signes peut-il la jouer ?",
    "variante": "LSFB a plusieurs signes pour ce mot : lequel a le sens de la glose ?",
    "incertain": "Étiquette LSFB ambiguë : vérifie que le signe a bien ce sens.",
    "synonyme": "Équivalent proposé à la main, jamais vérifié : convient-il ?",
    "nom-propre": "Signe-nom belge : le garder, ou épeler le nom ?",
}
CLIPS_PER_LABEL = 2
# Synonymes proposés à la main (étiquettes LSFB, sans accents) : seulement des pistes, le
# sens exact reste à vérifier dans la vidéo. Les étiquettes absentes de LSFB sont ignorées.
SYNONYMES = {
    "TRÈS": ["BEAUCOUP", "VRAIMENT", "FORT"], "TROP": ["BEAUCOUP", "EXAGERER"],
    "UTILISER": ["SERVIR", "UTILE", "PRENDRE"], "MAINTENANT": ["AUJOURD'HUI", "DEJA"],
    "PARDON": ["DESOLE"], "STYLO": ["CRAYON", "BIC", "ECRIRE"],
    "PATRON": ["CHEF", "DIRECTEUR", "RESPONSABLE"], "FRAUDE": ["TRICHER", "FAUX", "VOLER"],
    "FONCTIONNER": ["MARCHER", "ROULER", "TOURNER"], "NÉCESSAIRE": ["BESOIN", "FALLOIR", "OBLIGER"],
    "AUGMENTER": ["MONTER", "GRANDIR", "PLUS"], "ANNIVERSAIRE": ["FETE", "GATEAU"],
    "MESSAGE": ["SMS", "LETTRE"], "TESTER": ["ESSAYER", "EXAMEN"], "VIEUX": ["AGE", "ANCIEN"],
    "BEAU": ["BELLE"], "CIEL": ["BLEU", "NUAGE"], "DRÔLE": ["RIRE", "COMIQUE"],
    "RETROUVER": ["TROUVER", "RENCONTRER"], "CHAPITRE": ["PARTIE", "LIVRE"],
    "COUR": ["RECREATION", "JARDIN"], "COURSE": ["COURIR", "COURSES"], "DONNÉE": ["CHIFFRE"],
    "LENTEMENT": ["LENT", "DOUCEMENT"], "FRÉQUENT": ["SOUVENT", "REGULIER"],
    "PAROLE": ["PARLER", "MOT", "VOIX"], "PUBLIER": ["PARTAGER", "DIFFUSER"],
    "QUARTIER": ["RUE", "VILLE", "ENDROIT"], "OUVERT": ["OUVRIR"], "TERMINÉ": ["TERMINER", "FINIR", "FINI"],
    "SIGNER": ["SIGNE", "LS"], "ENTRAÎNER": ["ENTRAINEMENT", "APPRENDRE"],
    "INFORMATIQUE": ["ORDINATEUR"], "CONTENIR": ["INTERIEUR"], "GARER": ["PARKING", "VOITURE"],
    "LANCER": ["JETER", "COMMENCER"], "DÉTECTER": ["TROUVER", "DECOUVRIR", "VOIR"],
    "DÉTECTION": ["TROUVER", "DECOUVRIR"], "TRAITER": ["TRAVAILLER", "SOIGNER"],
    "TRAITEMENT": ["SOIGNER", "MEDICAMENT"], "VÉRITÉ": ["VRAI"], "ROMAN": ["LIVRE", "HISTOIRE"],
    "RÈGLE": ["REGLES", "LOI"], "PRÉSENTATION": ["PRESENTER"], "AGIR": ["FAIRE"], "AIDE": ["AIDER"],
    "ANIMER": ["ANIMATEUR"], "ANNOTER": ["NOTER", "ECRIRE"], "ANNOTÉ": ["NOTER", "ECRIRE"],
    "ACCÉLÉRER": ["VITE", "RAPIDE"], "ADORABLE": ["MIGNON", "GENTIL"], "CLASSER": ["RANGER", "ORGANISER"],
    "COMMENTER": ["DIRE", "EXPLIQUER"], "CONCERT": ["MUSIQUE", "CHANTER"],
    "CONVERGER": ["REJOINDRE", "ENSEMBLE"], "COUCHE": ["NIVEAU", "ETAGE"], "DÉBAT": ["DISCUTER"],
    "DÉCOUPER": ["COUPER"], "DÉFILÉ": ["MANIFESTATION"], "DÉMO": ["MONTRER", "EXEMPLE"],
    "ÉMISSION": ["TELEVISION", "PROGRAMME"], "ENTRETIEN": ["INTERVIEW", "RENDEZ-VOUS"],
    "ENTRÉE": ["ENTRER", "PORTE"], "FORMULAIRE": ["PAPIER", "REMPLIR"], "FÊTER": ["FETE"],
    "MÉDECINE": ["MEDECIN"], "MATHS": ["CALCULER"], "NEURONE": ["CERVEAU"],
    "PERFORMANCE": ["RESULTAT", "REUSSIR"], "POLLUÉ": ["POLLUTION", "SALE"], "POSITIF": ["POSITIVE"],
    "PRÉDIRE": ["DEVINER", "AVENIR"], "RÉDUIRE": ["DIMINUER", "MOINS"],
    "RETRANSMETTRE": ["DIFFUSER", "TELEVISION"], "TOURISTE": ["TOURISME"], "VALIDATION": ["ACCEPTER"],
    "ÉVALUER": ["EVALUATION", "NOTER"], "ENCOURAGEANT": ["ENCOURAGER"], "ENNUYEUX": ["ENNUYER"],
    "BOULANGERIE": ["BOULANGER", "PAIN"], "BRUITÉ": ["BRUIT"], "CHANGEMENT": ["CHANGER"],
    "LÉGUME": ["LEGUMES"], "VOISINE": ["VOISIN"], "TRANSCRIPTION": ["ECRIRE", "TRADUIRE"],
    "LOCAL": ["SALLE"], "CHEZ-MOI": ["MAISON", "CHEZ"], "ALGORITHME": ["PROGRAMME", "ORDINATEUR"],
}
# Nouvelles pistes pour les gloses qui bloquent une phrase du banc (relecture ciblée) :
# étiquettes LSFB exactes, « A+B » = signes joués à la suite. Toujours à vérifier en vidéo.
PISTES = {
    "QUAND": ["MOMENT", "DATE", "QUELLE+MOMENT", "QUELLE+DATE"],
    "MODÈLE": ["MODELE.EXEMPLE", "EXEMPLE"],
    "ENFIN": ["ENFIN.TRANQUILLE", "FIN", "FINIR"],
    "DEVOIR": ["OBLIGER.DEVOIR", "FALLOIR", "OBLIGER"],
    "OÙ": ["ENDROIT", "QUELLE+ENDROIT"],
    "PRÉ-ENTRAÎNÉ": ["AVANT+ENTRAINEMENT", "DEJA+ENTRAINEMENT", "ENTRAINEMENT"],
    "GARER": ["VOITURE+METTRE", "METTRE", "METTRE.05"],
    "TOURNER": ["MARCHER", "TRAVAIL.ACTIF", "ROULER"],
    "PRÉSENTATION": ["EXPLIQUER.COMMENTAIRE", "MONTRER", "EXPLIQUER"],
    "POLLUÉ": ["SALE", "POLLUTION+SALE"],
    "FORMULAIRE": ["FEUILLE.F", "PAPIER.BIEN"],
    "COMMENTER": ["EXPLIQUER.COMMENTAIRE", "EXPLIQUER.8", "PARLER"],
    "COURSE": ["COURIR", "MATCH"],
    "FRAUDE": ["FAUX.TRICHER", "VOLER.PRENDRE", "FAUX"],
    "CONSEILLER-À-VOUS": ["CONSEIL", "PROPOSER"],
    "SUITE": ["SUIVANT", "APRES", "CONTINUER", "PROCHAINE"],
    "OCCUPÉ": ["TRAVAIL.ACTIF", "CHARGE"],
    "DÉCISION": ["DECIDER", "CHOISIR"],
    "ÉLECTION": ["VOTER", "VOTE"],
    "DURER": ["LONGTEMPS.DUREE", "LONGTEMPS", "TEMPS"],
    "INDUSTRIE": ["ENTREPRISE", "FABRIQUER", "TRAVAIL+FABRIQUER"],
    "GENS": ["PERSONNE.HUMAIN", "HUMAIN", "PUBLIC", "PERSONNE"],
    # Même famille (nom / verbe / adjectif), souvent un seul signe en langue des signes.
    "FATIGUÉ": ["FATIGUER"],
    "PLEUVOIR": ["PLUIE"],
    "NEIGER": ["NEIGE"],
    "AIDE": ["AIDER"],
    "CUISINER": ["CUISINE"],
    "PAROLE": ["PARLER", "MOT"],
    "CHANGEMENT": ["CHANGER", "CHANGER.DEVENIR"],
    "MÉDECINE": ["MEDECIN"],
    "SIGNER": ["SIGNE"],
    "FÊTER": ["FETE", "FETE.PARTY"],
    "OUVERT": ["OUVRIR"],
    "MATHS": ["MATHEMATIQUE", "CALCULER"],
    "INFORMATIQUE": ["ORDINATEUR"],
    "COÛTER": ["PRIX"],
    "CHEZ-MOI": ["MAISON"],
    "TOUT-LE-MONDE": ["TOUT+PERSONNE", "TOUT"],
    "ANNÉE-DERNIÈRE": ["ANNEE-1", "ANNEE+DERNIER"],
    "SEMAINE-DERNIÈRE": ["SEMAINE-1", "SEMAINE+DERNIER"],
}


# ------------------------------------------------------------------- données
def load_labels() -> tuple[dict[str, list[dict]], dict[str, list[str]]]:
    """Instances LSFB-ISOL par étiquette, et étiquettes par forme de base (AIDER -> AIDER,
    AIDER.2…), les plus fréquentes d'abord."""
    by_label: dict[str, list[dict]] = defaultdict(list)
    with open(DATA / "instances.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by_label[r["sign"]].append(r)
    bases: dict[str, list[str]] = defaultdict(list)
    for label in sorted(by_label, key=lambda l: -len(by_label[l])):
        bases[label.split(".")[0]].append(label)
    return by_label, bases


def candidates(gloss: str, bases: dict[str, list[str]], extra: list[str]) -> list[dict]:
    """Jusqu'à MAX_CANDIDATES étiquettes LSFB pour une glose sans signe, avec la raison."""
    out: list[dict] = []

    def add(base: str, why: str, variants: int = 2) -> None:
        for label in bases.get(base, [])[:variants]:
            if label not in {c["etiquette"] for c in out}:
                out.append({"etiquette": label, "raison": why})

    for label in extra:                                   # tapées à la main lors d'une relecture
        if label.split(".")[0] in bases:
            out.append({"etiquette": label, "raison": "proposée à la relecture"})
    for base in SYNONYMES.get(gloss, []):
        add(norm(base), "synonyme proposé", variants=1)
    parts = [p for p in gloss.split("-") if p]
    for p in parts:
        add(norm(p), "partie de la glose" if len(parts) > 1 else "même mot")
        lem = norm(simplemma.lemmatize(p.lower(), lang="fr"))
        if lem != norm(p):
            add(lem, "lemme")
    n = norm(gloss)
    if len(n) >= 5:     # racine commune : BRUITÉ / BRUIT, BOULANGERIE / BOULANGER
        for base in bases:
            k = max(5, min(len(base), len(n)) - 2)
            if base != n and len(base) >= 5 and base.startswith(n[:k]) and n.startswith(base[:k]):
                add(base, "racine commune")
    if len(out) < MAX_CANDIDATES:
        for base in difflib.get_close_matches(n, list(bases), n=3, cutoff=0.75):
            add(base, "orthographe voisine (souvent un faux ami)")
    return out[:MAX_CANDIDATES + len(extra)]


def candidates_for(row: dict, bases: dict[str, list[str]], extra: list[str]) -> list[dict]:
    """Candidats selon le statut automatique : étiquettes LSFB trouvées (variantes
    comprises), complétées par la recherche des gloses absentes."""
    found = [l for l in [row["etiquette_lsfb"], *row["autres_etiquettes"].split()]
             if l and "+" not in l and l.split(".")[0] in bases]
    if row["statut"] == "absent" or not found:
        return candidates(row["glose_projet"], bases, extra)
    why = {"variante": "variante LSFB", "incertain": "étiquette ambiguë",
           "synonyme": "synonyme proposé à la main", "nom-propre": "signe-nom belge"}[row["statut"]]
    out = [{"etiquette": l, "raison": "proposée à la relecture"} for l in extra if l.split(".")[0] in bases]
    out += [{"etiquette": l, "raison": why} for l in found if l not in extra]
    return out[:MAX_CANDIDATES + len(extra)]


def blocking_candidates(gloss: str, bases: dict[str, list[str]], by_label: dict[str, list[dict]],
                        refused: list[str], extra: list[str], current: str = "") -> list[dict]:
    """Relecture ciblée : pistes, variantes LSFB du mot, candidats déjà refusés, puis le
    signe actuel (`current`, quand aucun de ses clips ne convenait)."""
    out: list[dict] = []

    def add(label: str, why: str) -> None:
        if label not in {c["etiquette"] for c in out} and all(p in by_label for p in label.split("+")):
            out.append({"etiquette": label, "raison": why})

    for label in extra:
        add(label, "proposée à la relecture")
    for label in PISTES.get(gloss, []):
        add(label, "signes joués à la suite" if "+" in label else "nouvelle piste")
    for c in candidates(gloss, bases, []):                # lemme, racine commune…
        if not c["raison"].startswith("orthographe"):
            add(c["etiquette"], "même famille" if c["raison"] in ("lemme", "racine commune") else c["raison"])
    for p in [p for p in gloss.split("-") if p]:
        for label in bases.get(norm(p), []):
            if label != current:
                add(label, "variante LSFB du mot")
    for label in refused:
        if label != current:
            add(label, "déjà refusé à la 1re relecture")
    out = out[:MAX_BLOCKING - bool(current)]
    if current:
        add(current, "signe actuel : aucun de ses clips ne convenait")
    return out


def blocking_glosses(only: set[str] | None = None) -> dict[str, list[str]]:
    """Gloses sans signe dans le dictionnaire de l'avatar (ou les gloses `only`) -> phrases
    du banc où elles apparaissent."""
    index = load_dictionary()
    out: dict[str, list[str]] = defaultdict(list)
    for path in BENCH:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                for g in dict.fromkeys(r["glosses"]):
                    if (g in only) if only is not None else resolve(g, index)["type"] == "absent":
                        out[g].append(r["text"])
    return out


def project_pairs() -> list[dict]:
    """Phrases françaises du projet et leur traduction en gloses (éval, dev, dataset d'entraînement)."""
    paths = [*BENCH, SRC / "eval" / "dataset.jsonl", SRC / "eval" / "dev.jsonl",
             *sorted((SRC / "finetune" / "data").glob("*/pairs.jsonl"))]
    return [json.loads(line) for path in paths
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def example_sentences() -> dict[str, list[str]]:
    """Phrases françaises du projet où chaque glose apparaît."""
    out: dict[str, list[str]] = defaultdict(list)
    for r in project_pairs():
        for g in set(r["glosses"]):
            if len(out[g]) < 3 and r["text"] not in out[g]:
                out[g].append(r["text"])
    return out


def read_csv() -> dict[str, dict]:
    if not CSV_PATH.exists():
        return {}
    with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
        return {r["glose"]: r for r in csv.DictReader(f)}


def write_csv(cards: list[dict], old: dict[str, dict]) -> None:
    """Réécrit le CSV dans l'ordre de la page sans perdre aucune réponse."""
    rows = [{"glose": c["glose"], "frequence": c["frequence"],
             "candidats": " ".join(x["etiquette"] for x in c["candidats"])}
            | {k: old.get(c["glose"], {}).get(k, "") for k in FIELDS[3:]} for c in cards]
    listed = {c["glose"] for c in cards}
    rows += [{k: v.get(k, "") for k in FIELDS} for g, v in old.items() if g not in listed]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def attach_videos(cards: list[dict], by_label: dict[str, list[dict]], download: bool) -> int:
    """Clips de chaque candidat (CLIPS_PER_LABEL signeurs ; pour une suite A+B, un clip par
    signe, joués à la suite) : téléchargés si besoin, puis copiés en H.264. Renvoie le nombre
    de clips indisponibles."""
    def clips(label: str) -> list[str]:
        if "+" in label:
            return [pick(by_label[p], 1)[0]["id"] for p in label.split("+")]
        return [r["id"] for r in pick(by_label[label], CLIPS_PER_LABEL)]
    wanted = {c["etiquette"]: clips(c["etiquette"]) for card in cards for c in card["candidats"]}
    todo = [i for ids in wanted.values() for i in ids if not (DATA / "videos" / f"{i}.mp4").exists()]
    if todo and download:
        print(f"Téléchargement de {len(todo)} clips LSFB-ISOL (CC BY-NC-SA 4.0)…", flush=True)

        def get(i: str) -> bool:
            try:
                fetch(f"{SOURCE}/{urllib.parse.quote(f'videos/{i}.mp4')}", DATA / "videos" / f"{i}.mp4")
                return True
            except Exception:
                return False
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(get, todo))
    rows = [{"video": f"videos/{i}.mp4"} for ids in wanted.values() for i in ids
            if (DATA / "videos" / f"{i}.mp4").exists()]
    if rows:
        to_h264(rows)
    ready = {Path(r["video"]).stem: r["video"] for r in rows if r["video"].startswith("videos_h264/")}
    missing = 0
    for card in cards:
        for c in card["candidats"]:
            c["videos"] = [ready[i] for i in wanted[c["etiquette"]] if i in ready]
            c["suite"] = "+" in c["etiquette"]
            c["exemples"] = min(len(by_label[p]) for p in c["etiquette"].split("+"))
            if c["suite"] and len(c["videos"]) < len(wanted[c["etiquette"]]):
                c["videos"] = []
            missing += not c["videos"]
    return missing


# ---------------------------------------------------------------------- page
PAGE = r"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITRE__</title>
<style>
:root { --bg:#f6f5f2; --card:#fff; --ink:#1d1d1f; --mute:#6b6b70; --line:#e2e0da; --accent:#2457c5;
        --ok:#1e6b3a; --mid:#a15c00; --bad:#b3261e; --none:#5f5f66; }
@media (prefers-color-scheme: dark) { :root { --bg:#17171a; --card:#222226; --ink:#ececf0; --mute:#a0a0a8;
        --line:#34343a; --accent:#8ab4ff; --ok:#81c995; --mid:#ffb74d; --bad:#ff8a80; --none:#b0b0b8; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.45 system-ui, sans-serif; }
header { position:sticky; top:0; z-index:1; background:var(--bg); display:flex; flex-wrap:wrap;
         gap:8px 16px; align-items:center; padding:10px 16px; border-bottom:1px solid var(--line); }
header h1 { font-size:17px; margin:0; }
#stats { color:var(--mute); font-size:14px; margin-right:auto; }
#stats b { color:var(--ink); }
button, select, input, textarea { font:inherit; font-size:14px; border-radius:6px; border:1px solid var(--line);
         background:var(--card); color:var(--ink); }
button, select, input { padding:5px 10px; }
button { cursor:pointer; }
#exportBtn.due { border-color:var(--bad); color:var(--bad); font-weight:600; }
main { max-width:1000px; margin:0 auto; padding:16px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px;
        display:flex; flex-direction:column; gap:12px; }
#gloss { font-size:30px; font-weight:700; margin:0; }
.meta { font-size:13px; color:var(--mute); }
#sentences { margin:0; padding-left:18px; }
#sentences li { margin:2px 0; }
#sentences .gl { font-family:ui-monospace,monospace; font-size:.9em; opacity:.8; }
#sentences .gl b { background:#ffe58a; color:#000; padding:0 3px; border-radius:3px; }
.question { font-size:16px; font-weight:600; }
.ctx { font-size:14px; color:var(--mid); }
#cands { display:grid; grid-template-columns:repeat(auto-fill, minmax(220px, 1fr)); gap:10px; }
.cand { border:1px solid var(--line); border-radius:10px; padding:8px; display:flex; flex-direction:column; gap:6px; }
.cand.on { outline:3px solid var(--ok); border-color:var(--ok); }
.cand video { width:100%; aspect-ratio:4/3; object-fit:contain; border-radius:6px; background:#000; }
.cand .label { font-weight:700; font-size:16px; }
.cand .why { font-size:12px; color:var(--mute); }
.cand .why.risky { color:var(--mid); }
.cand .row { display:flex; gap:6px; flex-wrap:wrap; }
.cand button.pick { font-weight:600; }
.novideo { font-size:13px; color:var(--bad); padding:20px 0; text-align:center; }
.row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
.verdicts { display:grid; grid-template-columns:repeat(3, 1fr); gap:8px; }
.verdicts button { padding:10px; text-align:left; }
.verdicts button small { display:block; color:var(--mute); font-size:12px; }
.verdicts button.on { outline:2px solid var(--accent); }
.v-candidat { color:var(--ok); } .v-autre { color:var(--ok); } .v-aucun { color:var(--bad); }
.v-épeler { color:var(--mid); } .v-incertain { color:var(--none); }
kbd { font:12px ui-monospace, monospace; border:1px solid var(--line); border-bottom-width:2px;
      border-radius:4px; padding:0 5px; color:var(--mute); }
textarea { width:100%; padding:6px; background:var(--bg); resize:vertical; }
.help { font-size:12px; color:var(--mute); }
@media (max-width: 560px) { .verdicts { grid-template-columns:1fr; } }
</style></head><body>
<header>
  <h1>__TITRE__</h1>
  <span id="stats"></span>
  <label>Afficher <select id="filter">
    <option value="todo">à relire</option><option value="all">toutes</option>
    <option value="candidat">signe choisi</option><option value="aucun">aucun</option>
    <option value="épeler">à épeler</option><option value="incertain">incertain</option>
  </select></label>
  <input id="jump" list="glosses" placeholder="Aller à une glose…" size="16">
  <datalist id="glosses"></datalist>
  <button id="exportBtn" title="Télécharge le CSV à copier dans le projet">Exporter le CSV</button>
</header>
<main><section class="card">
  <div class="row" style="justify-content:space-between">
    <h2 id="gloss"></h2><span id="pos" class="meta"></span>
  </div>
  <div class="meta" id="freq"></div>
  <div class="ctx" id="ctx"></div>
  <div class="meta" id="ancien"></div>
  <ul id="sentences" class="meta"></ul>
  <div class="question">Lequel de ces signes LSFB peut jouer cette glose ? <span id="current"></span></div>
  <div id="cands"></div>
  <div class="verdicts">
    <button data-v="aucun"><b>Aucun ne convient</b> <kbd>A</kbd><small>pas de signe pour l'instant</small></button>
    <button data-v="épeler"><b>À épeler</b> <kbd>E</kbd><small>nom propre, terme technique</small></button>
    <button data-v="incertain"><b>Je ne sais pas</b> <kbd>I</kbd><small>à revoir plus tard</small></button>
  </div>
  <div class="row">
    <input id="other" list="labels" placeholder="Autre étiquette LSFB…" size="24">
    <datalist id="labels"></datalist>
    <button id="otherBtn">Retenir cette étiquette</button>
    <span class="help">sa vidéo apparaîtra comme candidat à la prochaine génération de la page</span>
  </div>
  <div class="row">
    <button id="prev">← Précédente <kbd>←</kbd></button>
    <button id="slow"><span id="speed">Ralenti</span> <kbd>L</kbd></button>
    <button id="next">Suivante <kbd>→</kbd></button>
  </div>
  <textarea id="note" rows="2" placeholder="Note facultative (N pour écrire, Échap pour revenir)"></textarea>
  <div class="help"><kbd>1</kbd>–<kbd>__MAXK__</kbd> choisir ce candidat · <kbd>A</kbd> aucun · <kbd>E</kbd> à épeler ·
    <kbd>I</kbd> incertain · <kbd>0</kbd> effacer · <kbd>←</kbd> <kbd>→</kbd> naviguer · <kbd>L</kbd> ralenti ·
    <kbd>N</kbd> note. Une suite de signes s'écrit A+B dans « Autre étiquette ». Choisir un signe veut dire : l'avatar jouera ce signe à chaque fois que la glose apparaît.
    Les réponses sont gardées dans ce navigateur : exporte le CSV régulièrement.</div>
</section></main>
<script>
const CARDS = __CARDS__;
const LABELS = __LABELS__;
const GLOSES = __GLOSES__;     // phrase -> sa traduction en gloses
const KEY = __KEY__;
const AUTRES = __AUTRES__;   // lignes du CSV absentes de cette page, recopiées à l'export
const EXPORT_EVERY = 20;
let store = { answers: {}, sinceExport: 0 };
try { store = Object.assign(store, JSON.parse(localStorage.getItem(KEY) || "{}")); } catch (e) {}
// Le CSV du projet fait foi s'il est plus récent que ce que le navigateur a gardé.
for (const c of CARDS) {
  const local = store.answers[c.glose];
  if (c.verdict && (!local || (c.date || "") > (local.date || "")))
    store.answers[c.glose] = { verdict: c.verdict, choix: c.choix || "", note: c.note || "", date: c.date || "" };
}
function save() { try { localStorage.setItem(KEY, JSON.stringify(store)); } catch (e) {} }

const $ = id => document.getElementById(id);
let order = [], idx = 0, rate = 1;
const a = g => store.answers[g] || {};
const kind = vd => vd === "autre" ? "candidat" : vd;
function visible() {
  const f = $("filter").value;
  return CARDS.map((c, i) => i).filter(i => {
    const vd = kind(a(CARDS[i].glose).verdict);
    return f === "all" || (f === "todo" ? !vd : vd === f);
  });
}
function stats() {
  const n = { candidat: 0, aucun: 0, "épeler": 0, incertain: 0 };
  for (const c of CARDS) { const vd = kind(a(c.glose).verdict); if (vd in n) n[vd]++; }
  const done = Object.values(n).reduce((x, y) => x + y, 0);
  $("stats").innerHTML = `<b>${done}</b> / ${CARDS.length} · <span class="v-candidat">${n.candidat} signes choisis</span> · ` +
    `<span class="v-aucun">${n.aucun} aucun</span> · <span class="v-épeler">${n["épeler"]} à épeler</span> · ` +
    `<span class="v-incertain">${n.incertain} incertains</span>`;
  const due = store.sinceExport >= EXPORT_EVERY;
  $("exportBtn").classList.toggle("due", due);
  $("exportBtn").textContent = due ? `Exporter le CSV (${store.sinceExport} réponses non exportées)` : "Exporter le CSV";
}
function esc(s) { return String(s).replace(/[&<>"]/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch])); }
function show() {
  order = visible();
  if (!order.length) {
    $("gloss").textContent = "Rien à afficher";
    for (const id of ["freq", "ctx", "current", "pos"]) $(id).textContent = "";
    $("sentences").innerHTML = ""; $("cands").innerHTML =
      $("filter").value === "todo" ? "<p>Toutes les gloses sont relues : pense à exporter le CSV.</p>" : "";
    stats(); return;
  }
  idx = Math.max(0, Math.min(idx, order.length - 1));
  const c = CARDS[order[idx]], cur = a(c.glose);
  $("gloss").textContent = c.glose;
  $("freq").innerHTML = (c.bloque ? `empêche de jouer <b>${c.bloque}</b> phrase(s) du banc · ` : "") +
    `<b>${c.frequence}</b> occurrence(s) dans les datasets du projet · statut automatique : <b>${esc(c.statut)}</b>`;
  const old = c.ancien || {};
  $("ancien").innerHTML = old.verdict ? `Relecture précédente : <b class="v-${esc(old.verdict)}">${esc(old.choix || old.verdict)}</b>` +
    (old.candidats ? ` parmi ${esc(old.candidats)}` : "") + (old.note ? ` — « ${esc(old.note)} »` : "") : "";
  $("ctx").textContent = c.contexte;
  $("sentences").innerHTML = c.phrases.map(p => `<li>« ${esc(p)} »` + (GLOSES[p] ? `<div class="gl">→ ` +
    GLOSES[p].map(g => g === c.glose ? `<b>${esc(g)}</b>` : esc(g)).join(" ") + `</div>` : "") + `</li>`).join("");
  $("cands").innerHTML = c.candidats.length ? c.candidats.map((x, k) => `
    <div class="cand${cur.choix === x.etiquette ? " on" : ""}" data-k="${k}">
      <div class="label">${k + 1}. ${esc(x.etiquette)}</div>
      ${x.videos.length ? `<video autoplay ${x.suite ? "" : "loop "}muted playsinline src="${x.videos[0]}"></video>`
                        : `<div class="novideo">vidéo indisponible</div>`}
      <div class="why${x.raison.startsWith("orthographe") ? " risky" : ""}">${esc(x.raison)} · ${x.exemples} exemple(s) LSFB</div>
      <div class="row">
        <button class="pick">Choisir <kbd>${k + 1}</kbd></button>
        ${x.videos.length > 1 && !x.suite ? `<button class="swap">Autre signeur</button>` : ""}
      </div>
    </div>`).join("") : `<p class="meta">Aucun candidat trouvé automatiquement : « Aucun », « À épeler », ou tape une autre étiquette ci-dessous.</p>`;
  document.querySelectorAll(".cand").forEach(el => {
    const x = c.candidats[+el.dataset.k], vid = el.querySelector("video");
    if (vid) vid.playbackRate = rate;
    el.querySelector(".pick").onclick = () => answer("candidat", x.etiquette);
    const sw = el.querySelector(".swap");
    if (vid && x.suite) { let i = 0; vid.onended = () => { i = (i + 1) % x.videos.length; vid.src = x.videos[i]; vid.playbackRate = rate; vid.play().catch(() => {}); }; }
    if (sw) { let i = 0; sw.onclick = () => { i = (i + 1) % x.videos.length; vid.src = x.videos[i]; vid.playbackRate = rate; vid.play().catch(() => {}); }; }
  });
  const label = cur.verdict === "candidat" || cur.verdict === "autre" ? `signe ${cur.choix}` : cur.verdict;
  $("current").innerHTML = cur.verdict ? `— déjà répondu : <span class="v-${cur.verdict}">${esc(label)}</span>` : "";
  document.querySelectorAll(".verdicts button").forEach(b => b.classList.toggle("on", b.dataset.v === cur.verdict));
  $("note").value = cur.note || "";
  $("pos").textContent = `${idx + 1} / ${order.length}`;
  stats();
}
function localNow() {   // heure locale, au format AAAA-MM-JJTHH:MM:SS
  const d = new Date(), p = n => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}
const current = () => order.length ? CARDS[order[idx]] : null;
function answer(vd, choix = "") {
  const c = current(); if (!c) return;
  if (vd) store.answers[c.glose] = { verdict: vd, choix, note: $("note").value.trim(), date: localNow() };
  else delete store.answers[c.glose];
  store.sinceExport++; save();
  if (!(vd && $("filter").value === "todo")) idx += vd ? 1 : 0;   // en « à relire », la suivante prend la place
  show();
}
function move(d) { idx = Math.max(0, Math.min(idx + d, order.length - 1)); show(); }

document.querySelectorAll(".verdicts button").forEach(b => b.addEventListener("click", () => answer(b.dataset.v)));
$("otherBtn").onclick = () => {
  const l = $("other").value.trim().toUpperCase(); if (!l) return;
  if (!l.split("+").every(p => LABELS.includes(p)) && !confirm(`« ${l} » n'est pas une étiquette LSFB-ISOL connue. La retenir quand même ?`)) return;
  $("other").value = ""; answer("autre", l);
};
$("prev").onclick = () => move(-1);
$("next").onclick = () => move(1);
$("slow").onclick = () => {
  rate = rate === 1 ? 0.5 : 1;
  document.querySelectorAll("video").forEach(v => v.playbackRate = rate);
  $("speed").textContent = rate === 1 ? "Ralenti" : "Vitesse normale";
};
$("filter").onchange = () => { idx = 0; show(); };
$("note").addEventListener("input", () => {
  const c = current(); if (!c || !store.answers[c.glose]) return;
  store.answers[c.glose].note = $("note").value.trim(); save();
});
$("jump").addEventListener("change", () => {
  const g = $("jump").value.trim().toUpperCase(); $("jump").value = "";
  if (!CARDS.some(c => c.glose === g)) return;
  $("filter").value = "all"; order = visible();
  idx = order.findIndex(i => CARDS[i].glose === g); show(); $("jump").blur();
});
$("glosses").innerHTML = CARDS.map(c => `<option value="${esc(c.glose)}">`).join("");
$("labels").innerHTML = LABELS.map(l => `<option value="${esc(l)}">`).join("");
$("exportBtn").onclick = () => {
  const cols = ["glose", "frequence", "candidats", "choix", "verdict", "note", "date"];
  const q = x => '"' + String(x ?? "").replace(/"/g, '""') + '"';
  const lines = [cols.join(",")].concat(CARDS.map(c => {
    const cur = a(c.glose);
    if (!cur.verdict && c.ancien)      // pas relue ici : la réponse précédente est gardée
      return [c.glose, c.frequence, c.ancien.candidats, c.ancien.choix, c.ancien.verdict, c.ancien.note, c.ancien.date].map(q).join(",");
    return [c.glose, c.frequence, c.candidats.map(x => x.etiquette).join(" "), cur.choix, cur.verdict, cur.note, cur.date]
      .map(q).join(",");
  }), AUTRES.map(r => cols.map(k => q(r[k])).join(",")));
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob(["﻿" + lines.join("\n") + "\n"], { type: "text/csv" }));
  link.download = "lsfb_candidats.csv"; link.click();
  store.sinceExport = 0; save(); stats();
};
document.addEventListener("keydown", e => {
  if (["TEXTAREA", "INPUT", "SELECT"].includes(document.activeElement.tagName)) {
    if (e.key === "Escape") document.activeElement.blur();
    return;
  }
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key.toLowerCase(), c = current();
  if (c && /^[1-9]$/.test(k) && c.candidats[+k - 1]) answer("candidat", c.candidats[+k - 1].etiquette);
  else if (k === "a") answer("aucun");
  else if (k === "e") answer("épeler");
  else if (k === "i") answer("incertain");
  else if (k === "0") answer(null);
  else if (k === "arrowright") move(1);
  else if (k === "arrowleft") move(-1);
  else if (k === "l") $("slow").click();
  else if (k === "n") $("note").focus();
  else return;
  e.preventDefault();
});
show();
</script></body></html>
"""


MODES = {   # mode -> (titre, page, clé du navigateur)
    "relecture": ("Gloses à relire (LSFB)", HTML_PATH, "lsfb-candidats-absents"),
    "bloquantes": ("Gloses qui bloquent le banc (LSFB)", HTML_BLOCKING, "lsfb-candidats-bloquants"),
    "variantes": ("Variantes à changer (LSFB)", HTML_VARIANTS, "lsfb-candidats-variantes"),
    "absentes": ("Gloses sans signe (LSFB)", HTML_ABSENT, "lsfb-candidats-absentes"),
}


def write_html(cards: list[dict], labels: list[str], mode: str = "relecture", others: list[dict] = ()) -> Path:
    dump = lambda x: json.dumps(x, ensure_ascii=False).replace("</", "<\\/")
    title, path, key = MODES[mode]
    shown = {t for c in cards for t in c["phrases"]}
    glosses = {r["text"]: r["glosses"] for r in project_pairs() if r["text"] in shown}   # phrase -> gloses
    page = (PAGE.replace("__CARDS__", dump(cards)).replace("__LABELS__", dump(labels))
            .replace("__GLOSES__", dump(glosses))
            .replace("__AUTRES__", dump(list(others))).replace("__TITRE__", title)
            .replace("__KEY__", dump(key))
            .replace("__MAXK__", str(MAX_CANDIDATES if mode == "relecture" else MAX_BLOCKING)))
    path.write_text(page, encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ouvrir", action="store_true", help="ouvrir la page dans le navigateur")
    ap.add_argument("--sans-video", action="store_true", help="ne télécharge aucun clip (seulement ceux déjà présents)")
    ap.add_argument("--variantes", action="store_true",
                    help="changer de variante LSFB pour les signes « autre variante » de choix_clips.csv")
    ap.add_argument("--bloquantes", action="store_true",
                    help="relecture ciblée : gloses sans signe qui bloquent une phrase du banc, avec de nouvelles pistes")
    ap.add_argument("--absentes", action="store_true",
                    help="toutes les gloses sans signe, avec les signes LSFB de la même famille (PLUIE pour PLEUVOIR…)")
    args = ap.parse_args()
    if not (DATA / "instances.csv").exists():
        sys.exit("Index LSFB absent : lance d'abord lsfb_subset.py map (voir l'aide).")
    by_label, bases = load_labels()
    labels = LsfbLabels(*load_index(DATA))       # proposition automatique, sans les réponses
    old = read_csv()
    freq, sentences = gloss_frequencies(), example_sentences()
    if args.bloquantes or args.variantes or args.absentes:
        return main_blocking(args, by_label, bases, labels, old, freq, sentences)
    cards = []
    for gloss in project_glosses():
        row = labels.propose(gloss)
        if row["statut"] not in REVIEW_STATUSES and gloss not in old:
            continue
        prev = old.get(gloss, {})
        extra = [prev["choix"]] if prev.get("verdict") == "autre" and prev.get("choix") else []
        cards.append({"glose": gloss, "frequence": freq.get(gloss, 0), "phrases": sentences.get(gloss, []),
                      "statut": row["statut"], "contexte": CONTEXTE.get(row["statut"], ""),
                      "candidats": candidates_for(row, bases, extra),
                      **{k: prev.get(k, "") for k in ("choix", "verdict", "note", "date")}})
    cards.sort(key=lambda c: (-c["frequence"], c["glose"]))
    missing = attach_videos(cards, by_label, download=not args.sans_video)
    write_csv(cards, old)
    write_html(cards, sorted(by_label))
    n_cand = sum(len(c["candidats"]) for c in cards)
    done = Counter(c["verdict"] or "à relire" for c in cards)
    kinds = Counter(c["statut"] for c in cards)
    print(f"{len(cards)} gloses à relire (" + ", ".join(f"{k} {n}" for k, n in kinds.most_common()) + f"), {n_cand} candidats ({sum(not c['candidats'] for c in cards)} gloses sans "
          f"candidat) ; " + ", ".join(f"{k} {n}" for k, n in done.most_common()))
    if missing:
        print(f"⚠️  {missing} candidat(s) sans vidéo lisible.")
    print(f"CSV  : {CSV_PATH.relative_to(SRC.parent)}")
    print(f"Page : {HTML_PATH.relative_to(SRC.parent)}")
    if args.ouvrir:
        opener = {"darwin": ["open"], "win32": ["cmd", "/c", "start", ""]}.get(sys.platform, ["xdg-open"])
        subprocess.run([*opener, str(HTML_PATH)], check=False)


def variants_to_change() -> dict[str, str]:
    """Glose du projet -> signe LSFB actuel, pour les signes dont aucun clip ne convenait."""
    if not CLIP_CHOICES.exists():
        return {}
    with open(CLIP_CHOICES, newline="", encoding="utf-8-sig") as f:
        labels = {r["etiquette_lsfb"] for r in csv.DictReader(f) if r["verdict"] == "variante"}
    with open(SELECTION, newline="", encoding="utf-8") as f:
        return {g: r["etiquette_lsfb"] for r in csv.DictReader(f) if r["etiquette_lsfb"] in labels
                for g in r["gloses_projet"].split()}


def main_blocking(args, by_label, bases, labels, old, freq, sentences) -> None:
    """Relecture ciblée (gloses bloquantes, toutes les gloses sans signe, ou variantes à
    changer) : le CSV n'est pas réécrit, l'export de la page le remplace."""
    current = variants_to_change() if args.variantes else {}
    if args.variantes and not current:
        sys.exit(f"Aucun signe « autre variante » dans {CLIP_CHOICES.name} (page revue_clips.py).")
    context = (("Aucun clip du signe actuel ne convenait (relecture des clips) : quelle autre variante "
                "LSFB peut jouer cette glose ?") if args.variantes else
               ("Déjà relue sans signe retenu : voici d'autres pistes (synonymes, signes joués à la "
                "suite). « À épeler » reste possible pour un nom propre."))
    if args.absentes:
        context = ("Relue sans signe retenu. En langue des signes, le nom, le verbe et l'adjectif d'une "
                   "même famille (PLUIE / PLEUVOIR, FATIGUER / FATIGUÉ) sont souvent un seul signe : "
                   "juge le geste et le sens, pas la forme du mot. « À épeler » reste possible.")
        index = load_dictionary()
        absent = {g for g, p in index["gloses"].items() if p["type"] == "absent"}
        bench = blocking_glosses(absent)
        glosses = {g: bench.get(g, []) for g in sorted(absent)}
    else:
        glosses = blocking_glosses(set(current) if args.variantes else None)
    cards = []
    for gloss, texts in glosses.items():
        prev = old.get(gloss, {})
        extra = [prev["choix"]] if prev.get("verdict") == "autre" and prev.get("choix") else []
        refused = prev.get("candidats", "").split()
        cards.append({"glose": gloss, "frequence": freq.get(gloss, 0), "phrases": texts or sentences.get(gloss, []),
                      "bloque": len(texts),
                      "statut": labels.propose(gloss)["statut"], "contexte": context,
                      "candidats": blocking_candidates(gloss, bases, by_label, refused, extra, current.get(gloss, "")),
                      "ancien": {k: prev.get(k, "") for k in ("candidats", "choix", "verdict", "note", "date")}
                      if prev.get("verdict") else None,
                      "choix": "", "verdict": "", "note": "", "date": ""})
    cards.sort(key=lambda c: (-c["bloque"], -c["frequence"], c["glose"]))
    missing = attach_videos(cards, by_label, download=not args.sans_video)
    others = [r for g, r in old.items() if g not in {c["glose"] for c in cards}]
    mode = "variantes" if args.variantes else "absentes" if args.absentes else "bloquantes"
    path = write_html(cards, sorted(by_label), mode, others)
    verb = "apparaissent dans" if args.variantes else "bloquent"
    print(f"{len(cards)} gloses, dont {sum(bool(c['bloque']) for c in cards)} {verb} "
          f"{sum(c['bloque'] for c in cards)} phrase(s) du banc ; "
          f"{sum(len(c['candidats']) for c in cards)} candidats ({sum(not c['candidats'] for c in cards)} gloses sans candidat)")
    if missing:
        print(f"⚠️  {missing} candidat(s) sans vidéo lisible.")
    print(f"Page : {path.relative_to(SRC.parent)} — son export remplace {CSV_PATH.relative_to(SRC.parent)}")
    if args.ouvrir:
        opener = {"darwin": ["open"], "win32": ["cmd", "/c", "start", ""]}.get(sys.platform, ["xdg-open"])
        subprocess.run([*opener, str(path)], check=False)


if __name__ == "__main__":
    main()
