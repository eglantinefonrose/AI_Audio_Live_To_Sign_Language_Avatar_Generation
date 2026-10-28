"""Téléchargement d'un sous-ensemble de LSFB-ISOL (Université de Namur), glose par glose.

LSFB = langue des signes de Belgique francophone, PAS la LSF de France : voir
`sign_language_3d_avatar/dataset_generation/README.md` avant d'animer un avatar avec ces données.

Licence des données : CC BY-NC-SA 4.0 (pas d'usage commercial, citation obligatoire).
Le `Downloader` officiel (`lsfb-dataset`) ne filtre que par split et par signeur ;
ce script lit l'index `instances.csv` et ne récupère que les instances des gloses
choisies. Bibliothèque standard uniquement.

    uv run python sign_language_3d_avatar/dataset_generation/lsfb_subset.py map            # correspondances -> sign_language_3d_avatar/dataset_generation/lsfb_mapping.csv
    uv run python sign_language_3d_avatar/dataset_generation/lsfb_subset.py download --words MAISON,AIDER-À-TOI --dry-run
    uv run python sign_language_3d_avatar/dataset_generation/lsfb_subset.py download --words-file sign_language_3d_avatar/dataset_generation/lsfb_words.txt
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent   # sign_language_3d_avatar/
ROOT = BASE.parent                               # racine du projet
SOURCE = "https://lsfb.info.unamur.be/static/datasets/lsfb_v2/isol"
DEFAULT_DEST = BASE / "dataset_generation" / "data" / "lsfb"
MAPPING_PATH = BASE / "dataset_generation" / "lsfb_mapping.csv"
LICENCE = ("CC BY-NC-SA 4.0 ; Meurant, Laurence (2015), Corpus LSFB, "
           "LSFB-Lab, FRS-FNRS et Université de Namur ; https://lsfb.info.unamur.be/")
INDEX_FILES = ("instances.csv", "instances_special.csv", "metadata/sign_occurrences.csv")
LANDMARKS = ("pose", "left_hand", "right_hand", "face")

# Pointages du projet -> étiquettes LSFB (classe « special », sans fichiers servis dans ISOL).
POINTS = {"MOI": "PT:PRO1", "TOI": "PT:PRO2", "LUI": "PT:PRO3",
          "NOUS": "PT:PRO4", "VOUS": "PT:PRO5", "EUX": "PT:PRO6"}
DIRECTIONAL_RE = re.compile(r"^(.+)-À-(MOI|TOI|LUI|NOUS|VOUS|EUX)$")
# Nombres : LSFB les note en chiffres.
NUMBERS = {"UN": "1", "DEUX": "2", "TROIS": "3", "QUATRE": "4", "CINQ": "5", "SIX": "6",
           "SEPT": "7", "HUIT": "8", "NEUF": "9", "DIX": "10", "ONZE": "11", "DOUZE": "12",
           "TREIZE": "13", "QUATORZE": "14", "QUINZE": "15", "SEIZE": "16", "VINGT": "20",
           "CENT": "100", "DEUX-CENTS": "200", "QUATRE-CENTS": "400", "MILLE": "1000"}
# Équivalents proposés à la main (sens proche, autre étiquette) : toujours à valider.
SYNONYMS = {"PÈRE": "PAPA", "MÈRE": "MAMAN", "PARENT": "PARENTS", "D'ACCORD": "ACCORD",
            "REMERCIER": "MERCI", "PLEUVOIR": "PLUIE", "NEIGER": "NEIGE", "TÉLÉPHONER": "TELEPHONE",
            "SEUL": "SEULE", "FATIGUÉ": "FATIGUER", "CUISINER": "CUISINE", "SYMPA": "SYMPATHIQUE",
            "SUCCÈS": "REUSSIR", "REVENIR": "RETOUR", "COÛTER": "PRIX", "LSF": "LS",
            "DÎNER": "MANGER", "TOUT-LE-MONDE": "TOUT+MONDE"}
# Gloses dont l'accent distingue deux mots que LSFB (sans accents) confond.
HOMOGRAPHS = {"OÙ": "OU (« ou » ?) / OU.OU : où ≠ ou, à vérifier dans la vidéo"}
# Statuts téléchargés par défaut ; les autres demandent --inclure-incertains.
SAFE_STATUSES = {"exact", "directionnel"}


# --------------------------------------------------------------------------- index

def fetch(url: str, path: Path, timeout: int = 60) -> int:
    """Télécharge `url` dans `path` avec reprise (.part + en-tête Range). Renvoie les octets écrits."""
    if path.exists():
        return 0
    part = path.with_name(path.name + ".part")
    path.parent.mkdir(parents=True, exist_ok=True)
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        if e.code == 416:  # .part déjà complet
            part.rename(path)
            return 0
        raise
    with resp:
        # Le serveur renvoie une page HTML quand le fichier n'existe pas.
        if resp.headers.get_content_type() == "text/html":
            raise FileNotFoundError(url)
        mode = "ab" if have and resp.status == 206 else "wb"
        written = 0
        with open(part, mode) as f:
            while chunk := resp.read(1 << 16):
                f.write(chunk)
                written += len(chunk)
    part.rename(path)
    return written


def head_size(url: str, timeout: int = 30) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.headers.get_content_type() == "text/html":
                return None
            return int(resp.headers.get("Content-Length", 0))
    except urllib.error.HTTPError:
        return None


def load_index(dest: Path) -> tuple[list[dict], Counter]:
    """Instances servies (avec fichiers) et effectifs des étiquettes « special » (pointages…)."""
    for name in INDEX_FILES:
        fetch(f"{SOURCE}/{name}", dest / name)
    with open(dest / "instances.csv", newline="", encoding="utf-8") as f:
        instances = list(csv.DictReader(f))
    with open(dest / "instances_special.csv", newline="", encoding="utf-8") as f:
        special = Counter(r["sign"] for r in csv.DictReader(f))
    return instances, special


# ----------------------------------------------------------------- correspondances

def norm(label: str) -> str:
    """Forme de comparaison : sans accents, apostrophes/espaces -> '-', ponctuation finale ôtée."""
    s = unicodedata.normalize("NFKD", label.replace("\xa0", " "))
    s = "".join(c for c in s if not unicodedata.combining(c)).upper().strip(" !?)")
    s = s.replace("Œ", "OE").replace("Æ", "AE")
    s = re.sub(r"[’' ]+", "-", s)
    return re.sub(r"-{2,}", "-", s)


class LsfbLabels:
    def __init__(self, instances: list[dict], special: Counter):
        self.count = Counter(r["sign"] for r in instances)
        self.special = special
        self.by_norm: dict[str, list[str]] = defaultdict(list)   # forme exacte
        self.by_base: dict[str, list[str]] = defaultdict(list)   # avant le « .VARIANTE »
        for label in self.count:
            self.by_norm[norm(label)].append(label)
            if "." in label:
                self.by_base[norm(label.split(".", 1)[0])].append(label)

    def _best(self, labels: list[str]) -> list[str]:
        return sorted(labels, key=lambda l: -self.count[l])

    def lookup(self, n: str) -> tuple[list[str], list[str]]:
        """(étiquettes exactes, variantes « BASE.X ») pour une forme normalisée."""
        return self._best(self.by_norm.get(n, [])), self._best(self.by_base.get(n, []))

    def resolve(self, gloss: str) -> dict:
        row = {"glose_projet": gloss, "etiquette_lsfb": "", "statut": "absent",
               "instances": 0, "autres_etiquettes": "", "note": ""}
        if gloss in POINTS:
            label = POINTS[gloss]
            row.update(etiquette_lsfb=label, statut="pointage", instances=0,
                       note=f"{self.special[label]} occurrences annotées dans le corpus, "
                            "mais pas de clip servi dans ISOL : à générer (pointage)")
            return row
        base, note = gloss, ""
        if m := DIRECTIONAL_RE.match(gloss):
            base = m.group(1)
            note = f"forme de citation ; orientation vers {m.group(2)} à produire par l'avatar"
        exact, variants = self.lookup(norm(NUMBERS.get(base, base)))
        if base in SYNONYMS:
            targets = SYNONYMS[base].split("+")
            if all(t in self.count for t in targets):
                row.update(etiquette_lsfb=SYNONYMS[base], statut="synonyme",
                           instances=min(self.count[t] for t in targets),
                           note=(note + " ; " if note else "") + "équivalent proposé à la main, à valider")
                return row
        if gloss in HOMOGRAPHS and exact:
            row.update(etiquette_lsfb=exact[0], statut="incertain", instances=self.count[exact[0]],
                       autres_etiquettes=" ".join(exact[1:] + variants), note=HOMOGRAPHS[gloss])
        elif exact:
            row.update(etiquette_lsfb=exact[0], statut="directionnel" if note else "exact",
                       instances=self.count[exact[0]],
                       autres_etiquettes=" ".join(exact[1:] + variants), note=note)
        elif variants:
            row.update(etiquette_lsfb=variants[0], statut="variante", instances=self.count[variants[0]],
                       autres_etiquettes=" ".join(variants[1:]),
                       note=(note + " ; " if note else "") + "plusieurs signes possibles, choisir le bon")
        elif "-" in base:
            parts = [self.lookup(norm(p)) for p in base.split("-")]
            labels = [(e or v or [None])[0] for e, v in parts]
            if all(labels):
                row.update(etiquette_lsfb="+".join(labels), statut="décomposé",
                           instances=min(self.count[l] for l in labels),
                           note="composé absent : signes séparés, enchaînement à valider")
        if row["statut"] == "absent" and self.count.get("NS:" + norm(base)):
            label = "NS:" + norm(base)
            row.update(etiquette_lsfb=label, statut="nom-propre", instances=self.count[label],
                       note="signe-nom belge ; épellation dactylologique possible à la place")
        return row


def project_glosses() -> list[str]:
    glosses: set[str] = set()
    for line in (ROOT / "lsf_translator" / "lexicon.txt").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            glosses.add(line.strip())
    with open(ROOT / "eval" / "dataset.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                glosses.update(json.loads(line)["glosses"])
    return sorted(glosses)


def cmd_map(args) -> None:
    instances, special = load_index(args.dest)
    labels = LsfbLabels(instances, special)
    rows = [labels.resolve(g) for g in project_glosses()]
    with open(MAPPING_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    stats = Counter(r["statut"] for r in rows)
    print(f"{len(rows)} gloses du projet -> {MAPPING_PATH.relative_to(ROOT)}")
    for status, n in stats.most_common():
        print(f"  {status:13} {n}")


# ------------------------------------------------------------------ téléchargement

def read_words(args) -> list[str]:
    words = [w.strip() for w in (args.words or "").split(",") if w.strip()]
    if args.words_file:
        for line in Path(args.words_file).read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                words.append(line)
    return list(dict.fromkeys(words))


def pick(rows: list[dict], n: int | None) -> list[dict]:
    """Jusqu'à `n` instances, en tournant sur les signeurs (variété pour l'avatar)."""
    by_signer: dict[str, list[dict]] = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["id"]):
        by_signer[r["signer"]].append(r)
    queues, out = [by_signer[s] for s in sorted(by_signer)], []
    while any(queues) and (n is None or len(out) < n):
        for q in queues:
            if q and (n is None or len(out) < n):
                out.append(q.pop(0))
    return out


def file_paths(instance_id: str, args) -> list[str]:
    paths = [f"poses/{lm}/{instance_id}.npy" for lm in args.landmarks]
    if args.raw:
        paths += [f"poses_raw/{lm}/{instance_id}.npy" for lm in args.landmarks]
    if args.videos:
        paths.append(f"videos/{instance_id}.mp4")
    return paths


def human(n: float) -> str:
    for unit in ("o", "Ko", "Mo", "Go"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "o" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} To"


def estimate(selected: list[tuple[str, str, dict]], args) -> tuple[int, int, float]:
    """(fichiers à télécharger, déjà présents, octets estimés) : HEAD sur un échantillon, extrapolé à la durée."""
    todo, present = [], 0
    # Plusieurs gloses du projet partagent une étiquette (AIDER-À-MOI, AIDER-À-TOI) : on dédoublonne.
    for inst in {inst["id"]: inst for _, _, inst in selected}.values():
        for p in file_paths(inst["id"], args):
            if (args.dest / p).exists():
                present += 1
            else:
                todo.append((p, inst))
    if not todo:
        return 0, present, 0.0
    step = max(1, len(todo) // args.sample)
    sample = todo[::step][: args.sample]
    rate: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])  # dossier -> [octets, ms]
    with ThreadPoolExecutor(args.jobs) as ex:
        sizes = ex.map(lambda t: head_size(f"{SOURCE}/{urllib.parse.quote(t[0])}"), sample)
        for (p, inst), size in zip(sample, sizes):
            if size is not None:
                kind = p.rsplit("/", 1)[0]
                rate[kind][0] += size
                rate[kind][1] += max(1, int(inst["end"]) - int(inst["start"]))
    total = 0.0
    for p, inst in todo:
        b, ms = rate.get(p.rsplit("/", 1)[0], (0.0, 1.0))
        total += b / ms * max(1, int(inst["end"]) - int(inst["start"]))
    return len(todo), present, total


def write_manifest(dest: Path, entries: list[dict]) -> Path:
    path = dest / "manifest.csv"
    old = {}
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            old = {(r["glose_projet"], r["instance_id"]): r for r in csv.DictReader(f)}
    for e in entries:
        old[(e["glose_projet"], e["instance_id"])] = e
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(entries[0]))
        w.writeheader()
        w.writerows(sorted(old.values(), key=lambda r: (r["glose_projet"], r["instance_id"])))
    return path


def cmd_download(args) -> None:
    words = read_words(args)
    if not words:
        sys.exit("Aucun mot : --words A,B ou --words-file fichier.txt")
    instances, special = load_index(args.dest)
    labels = LsfbLabels(instances, special)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in instances:
        by_label[r["sign"]].append(r)

    selected: list[tuple[str, str, dict]] = []   # (glose projet, étiquette LSFB, instance)
    print(f"{'glose projet':24} {'étiquette LSFB':24} {'statut':12} {'dispo':>6} {'pris':>5}")
    for word in words:
        if word.startswith("lsfb:"):
            row = {"glose_projet": word[5:], "etiquette_lsfb": word[5:], "statut": "exact", "note": ""}
        else:
            row = labels.resolve(word)
        targets = [l for l in row["etiquette_lsfb"].split("+") if l in by_label]
        ok = row["statut"] in SAFE_STATUSES or (args.inclure_incertains and targets)
        taken = 0
        if ok:
            for label in targets:
                chosen = pick(by_label[label], args.max_par_glose)
                selected += [(row["glose_projet"], label, inst) for inst in chosen]
                taken += len(chosen)
        dispo = sum(len(by_label[l]) for l in targets)
        flag = "" if ok else "  (ignoré" + (" : --inclure-incertains)" if targets else ")")
        print(f"{row['glose_projet']:24} {row['etiquette_lsfb'] or '-':24} {row['statut']:12} "
              f"{dispo:>6} {taken:>5}{flag}")

    n_files, present, size = estimate(selected, args)
    signers = len({inst["signer"] for _, _, inst in selected})
    uniq = len({inst["id"] for _, _, inst in selected})
    print(f"\n{len(selected)} instances ({uniq} distinctes), {signers} signeurs ; {n_files} fichiers à télécharger "
          f"(≈ {human(size)}), {present} déjà présents. Fichiers : {', '.join(args.landmarks)}"
          f"{' + bruts' if args.raw else ''}{' + vidéos' if args.videos else ''}")
    if args.dry_run or not selected:
        return

    jobs = [(p, inst) for _, _, inst in selected for p in file_paths(inst["id"], args)]
    jobs = list({p: (p, inst) for p, inst in jobs}.values())
    done, failed, written = 0, [], 0
    with ThreadPoolExecutor(args.jobs) as ex:
        futures = {ex.submit(fetch, f"{SOURCE}/{urllib.parse.quote(p)}", args.dest / p): p for p, _ in jobs}
        for fut in as_completed(futures):
            done += 1
            try:
                written += fut.result()
            except Exception as e:  # noqa: BLE001 — on note et on continue, une relance reprendra
                failed.append(f"{futures[fut]} : {e}")
            if done % 50 == 0 or done == len(jobs):
                print(f"\r  {done}/{len(jobs)} fichiers, {human(written)} reçus", end="", flush=True)
    print()

    entries = []
    for gloss, label, inst in selected:
        files = [p for p in file_paths(inst["id"], args) if (args.dest / p).exists()]
        entries.append({"glose_projet": gloss, "etiquette_lsfb": label, "instance_id": inst["id"],
                        "signeur": inst["signer"], "debut_ms": inst["start"], "fin_ms": inst["end"],
                        "fichiers": " ".join(files), "source": SOURCE, "licence": LICENCE})
    path = write_manifest(args.dest, entries)
    print(f"Manifeste : {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    if failed:
        print(f"{len(failed)} échecs (relancer la même commande pour reprendre) :", *failed[:10], sep="\n  ")
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="dossier des données (défaut sign_language_3d_avatar/dataset_generation/data/lsfb)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("map", help="écrit sign_language_3d_avatar/dataset_generation/lsfb_mapping.csv (gloses du projet -> étiquettes LSFB)")
    d = sub.add_parser("download", help="télécharge les instances des gloses choisies")
    d.add_argument("--words", help="gloses séparées par des virgules (projet, ou lsfb:ÉTIQUETTE)")
    d.add_argument("--words-file", help="une glose par ligne, # pour commenter")
    d.add_argument("--max-par-glose", type=int, default=10, help="instances max par étiquette (0 = toutes)")
    d.add_argument("--landmarks", type=lambda s: s.split(","), default=["pose", "left_hand", "right_hand"],
                   help=f"parmi {','.join(LANDMARKS)} (défaut pose,left_hand,right_hand)")
    d.add_argument("--raw", action="store_true", help="ajoute les poses brutes (sans lissage)")
    d.add_argument("--videos", action="store_true", help="ajoute les vidéos .mp4")
    d.add_argument("--inclure-incertains", action="store_true",
                   help="télécharge aussi les statuts variante / décomposé / incertain")
    d.add_argument("--dry-run", action="store_true", help="affiche le volume sans rien télécharger")
    d.add_argument("--jobs", type=int, default=8, help="connexions parallèles")
    d.add_argument("--sample", type=int, default=40, help="taille de l'échantillon HEAD pour l'estimation")
    args = ap.parse_args()
    args.dest = args.dest.resolve()
    if args.cmd == "download":
        args.max_par_glose = args.max_par_glose or None
        bad = set(args.landmarks) - set(LANDMARKS)
        if bad:
            ap.error(f"landmarks inconnus : {', '.join(bad)}")
    {"map": cmd_map, "download": cmd_download}[args.cmd](args)


if __name__ == "__main__":
    main()
