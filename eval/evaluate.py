"""Évaluation du traducteur LSF : mesure la qualité des gloses sur un jeu de
paires de référence (`eval/dataset.jsonl`).

Appelle Ollama avec **exactement** le prompt de production (`build_messages` +
`GLOSS_JSON_SCHEMA`), puis compare la sortie à la glose attendue selon trois
mesures complémentaires :

- **exact**   : la séquence de gloses est identique (ordre compris) ;
- **F1**      : recouvrement lexical (multiset) — mesure le bon CHOIX des gloses
                indépendamment de l'ordre (détecte mots-outils en trop / gloses
                manquantes) ;
- **ordre**   : longueur de la plus longue sous-séquence commune / |référence|
                — mesure le respect de l'ORDRE.

Affiche aussi le **taux de fuite** de mots-outils (DE, DANS, POUR…) et une
ventilation par **phénomène** (étiquettes du dataset).

Avec `--save`, enregistre un **record immuable** (JSON) dans `eval/results/` et
ajoute une ligne au tableau de bord `eval/RESULTS.md`. Le record pince les trois
variables qui déterminent un résultat : modèle (nom+digest), prompt (hash+commit
git) et dataset (sha256). Voir `eval/README.md`.

Usage :
    uv run python eval/evaluate.py                       # mesure, sans sauvegarde
    uv run python eval/evaluate.py --save --label "..."  # mesure + archive
    uv run python eval/evaluate.py --model qwen2.5:7b
    uv run python eval/evaluate.py --quiet               # agrégats seulement
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from ollama import Client

from config import CONFIG
from lsf_translator.prompts import (
    FEW_SHOT_EXAMPLES,
    GLOSS_JSON_SCHEMA,
    SYSTEM_PROMPT,
    TRAIN_SYSTEM,
    build_messages,
    build_rules_only_messages,
)

# Modes de prompt : "prod" = prompt de production (règles + few-shot) ;
# "rules-only" = prompt d'ENTRAÎNEMENT des modèles fine-tunés (règles seules).
PROMPT_BUILDERS = {"prod": build_messages, "rules-only": build_rules_only_messages}

RESULTS_DIR = PROJECT / "eval" / "results"
RESULTS_MD = PROJECT / "eval" / "RESULTS.md"
DATASET = PROJECT / "eval" / "dataset.jsonl"
# Copies figées des datasets évalués : eval/datasets/<dataset_sha>/<nom>.jsonl
DATASETS_DIR = PROJECT / "eval" / "datasets"
MODELS_DIR = PROJECT / "finetune" / "models"   # provenance des modèles fine-tunés

# Gloses "mots-outils" qui ne devraient JAMAIS apparaître (fuite de français).
FORBIDDEN = {
    "DE", "DANS", "POUR", "LE", "LA", "LES", "UN", "UNE", "DES", "À", "AU",
    "ET", "OU", "QUE", "AVEC", "EN", "DU", "CE", "CETTE", "SE", "NE", "ON", "VA",
}


# -- métriques -----------------------------------------------------------
def multiset_f1(pred: list[str], gold: list[str]) -> tuple[float, float, float]:
    cp, cg = Counter(pred), Counter(gold)
    tp = sum((cp & cg).values())
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def lcs_len(a: list[str], b: list[str]) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] \
                else max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def order_score(pred: list[str], gold: list[str]) -> float:
    if not gold:
        return 1.0 if not pred else 0.0
    return lcs_len(pred, gold) / len(gold)


# -- provenance (pour --save) -------------------------------------------
def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


def prompt_hash(mode: str = "prod") -> str:
    """Hash du prompt assemblé : change dès que le système ou un exemple change."""
    if mode == "rules-only":
        return _sha(TRAIN_SYSTEM.encode("utf-8"))
    return _sha((SYSTEM_PROMPT + repr(FEW_SHOT_EXAMPLES)).encode("utf-8"))


def git_info() -> tuple[str | None, bool]:
    def run(args: list[str]) -> str:
        return subprocess.run(
            ["git", *args], cwd=PROJECT, capture_output=True, text=True
        ).stdout.strip()
    try:
        commit = run(["rev-parse", "--short", "HEAD"]) or None
        dirty = bool(run(["status", "--porcelain"]))
        return commit, dirty
    except Exception:
        return None, False


def model_digest(client: Client, name: str) -> str | None:
    try:
        listing = client.list()
        models = listing.get("models") if isinstance(listing, dict) \
            else getattr(listing, "models", [])
        for m in models or []:
            mn = m.get("model") if isinstance(m, dict) else getattr(m, "model", None)
            dg = m.get("digest") if isinstance(m, dict) else getattr(m, "digest", None)
            if mn == name or (mn and mn.split(":")[0] == name.split(":")[0]):
                return (dg or "")[:19] or None
    except Exception:
        return None
    return None


def training_info(model: str, digest: str | None) -> dict | None:
    """Provenance d'entraînement d'un modèle fine-tuné (finetune/models/<tag>.json,
    écrit par finetune/register_ollama.py). None pour un modèle de base."""
    path = MODELS_DIR / (model.replace(":", "-").replace("/", "-") + ".json")
    if not path.exists():
        return None
    reg = json.loads(path.read_text(encoding="utf-8"))
    info = dict(reg.get("training", {}))
    # Le tag peut avoir été recréé depuis l'enregistrement : on le signale.
    info["digest_match"] = bool(digest) and digest == reg.get("ollama_digest")
    return info


def format_training(t: dict | None) -> str:
    if not t:
        return ""
    d = t.get("duration_s")
    approx = "~" if t.get("approximate") else ""
    # Début inconnu (run reconstitué) : on date par la fin d'entraînement.
    when = (t.get("started_at") or t.get("ended_at") or "?")[:16].replace("T", " ")
    dur = f"{approx}{d // 60} min" if isinstance(d, int) else "durée inconnue"
    return f"entraîné {approx}{when} ({dur})"


def measure_prompt_tokens(client: Client, model: str, rows: list[dict],
                          mode: str = "prod") -> int | None:
    """Taille réelle du prompt (num_ctx large) pour détecter une troncature."""
    longest = max((r["text"] for r in rows), key=len)
    try:
        r = client.chat(
            model=model, messages=PROMPT_BUILDERS[mode](longest, "fr"),
            format=GLOSS_JSON_SCHEMA, options={"num_ctx": 8192, "temperature": 0.0},
        )
        return r.get("prompt_eval_count")
    except Exception:
        return None


# -- appel LLM -----------------------------------------------------------
def translate(client: Client, model: str, text: str, language: str,
              mode: str = "prod") -> list[str]:
    r = client.chat(
        model=model, messages=PROMPT_BUILDERS[mode](text, language),
        format=GLOSS_JSON_SCHEMA,
        options={"temperature": 0.0, "num_ctx": CONFIG.ollama.num_ctx},
    )
    content = r["message"]["content"]
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return []
    glosses = data.get("glosses", []) if isinstance(data, dict) else []
    return [str(g).strip() for g in glosses if str(g).strip()]


# -- sauvegarde ----------------------------------------------------------
def snapshot_dataset(path: Path, sha: str) -> Path:
    """Fige une copie du dataset évalué sous eval/datasets/<sha>/ (jamais
    réécrite) : un record reste ainsi rejouable même si dataset.jsonl évolue."""
    dest = DATASETS_DIR / sha / path.name
    if dest.exists():
        if _sha(dest.read_bytes()) != sha:
            sys.exit(f"Copie altérée : {dest} ne correspond plus au hash {sha}.")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, dest)
    dest.chmod(0o444)   # lecture seule : une copie figée ne se modifie pas
    return dest


def save_record(record: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_model = record["model"].replace(":", "-").replace("/", "-")
    stamp = record["timestamp"].replace(":", "-")
    fname = f"{stamp}__{safe_model}__{record['git_commit'] or 'nogit'}.json"
    path = RESULTS_DIR / fname
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    _append_results_md(record, fname)
    return path


def _append_results_md(record: dict, fname: str) -> None:
    header = (
        "# Suivi des évaluations\n\n"
        "Tableau de bord des runs d'évaluation LSF (généré par "
        "`eval/evaluate.py --save`). Une ligne par run ; `git*` = modifs non "
        "commitées au moment du run. Détail complet dans `eval/results/`.\n\n"
        "| Date | Heure | Label | Modèle | git | Dataset éval | Dataset train "
        "| Exact | F1 | Ordre | Fuites | Record |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    if not RESULTS_MD.exists():
        RESULTS_MD.write_text(header, encoding="utf-8")
    with RESULTS_MD.open("a", encoding="utf-8") as fh:
        fh.write(results_row(record, fname))


def results_row(record: dict, fname: str) -> str:
    """Une ligne du tableau RESULTS.md pour un record."""
    m = record["metrics"]
    # Dataset d'entraînement : seulement pour un modèle fine-tuné (sinon « — »).
    t = record.get("training") or {}
    train_sha = t.get("train_sha")
    if train_sha and t.get("train_snapshot"):   # lien vers la copie figée
        train_cell = f"[{train_sha}](../{t['train_snapshot']}/)"
    else:
        train_cell = train_sha or "—"
    trained = format_training(record.get("training"))
    model_cell = record["model"] + (f" ({trained})" if trained else "")
    git = (record["git_commit"] or "—") + ("*" if record["git_dirty"] else "")
    return (
        f"| {record['timestamp'][:10]} | {record['timestamp'][11:19]} "
        f"| {record['label']} | {model_cell} "
        f"| {git} | [{record['dataset_sha']}](datasets/{record['dataset_sha']}/) "
        f"| {train_cell} "
        f"| {m['exact']*100:.0f}% | {m['f1']:.3f} | {m['order']:.3f} "
        f"| {m['leaks']} | [{fname}](results/{fname}) |\n"
    )


# -- boucle --------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Évaluation gloses LSF.")
    ap.add_argument("--model", default=CONFIG.ollama.model)
    ap.add_argument("--dataset", default=str(DATASET))
    ap.add_argument("--prompt", choices=sorted(PROMPT_BUILDERS), default="prod",
                    help="prod = règles + few-shot (défaut) ; rules-only = prompt "
                         "d'entraînement, sans exemples (pour les modèles fine-tunés).")
    ap.add_argument("--quiet", action="store_true", help="Agrégats seulement.")
    ap.add_argument("--save", action="store_true",
                    help="Archive le run (JSON + ligne dans RESULTS.md).")
    ap.add_argument("--label", default="(sans label)",
                    help="Description courte du run (ex. 'ajout lexique').")
    args = ap.parse_args()

    rows = [json.loads(l) for l in
            Path(args.dataset).read_text(encoding="utf-8").splitlines() if l.strip()]
    client = Client(host=CONFIG.ollama.host)

    digest = model_digest(client, args.model)
    training = training_info(args.model, digest)
    print(f"Modèle : {args.model}  —  {len(rows)} phrases  —  prompt {args.prompt}")
    if training:
        print(f"         {format_training(training)}")
        if not training["digest_match"]:
            print("         ⚠️  digest Ollama ≠ celui enregistré : le tag a été recréé "
                  "depuis finetune/register_ollama.py, provenance douteuse.")
    print()
    sum_f = sum_o = exact = leaks = 0
    by_phen: dict[str, list[float]] = {}
    per_sentence: list[dict] = []

    for row in rows:
        gold = row["glosses"]
        pred = translate(client, args.model, row["text"], row.get("language", "fr"),
                         args.prompt)
        _, _, f = multiset_f1(pred, gold)
        o = order_score(pred, gold)
        is_exact = pred == gold
        leak = [g for g in pred if g in FORBIDDEN]
        sum_f += f
        sum_o += o
        exact += int(is_exact)
        leaks += int(bool(leak))
        for ph in row.get("phenomena", []):
            by_phen.setdefault(ph, []).append(f)
        per_sentence.append({
            "text": row["text"], "gold": gold, "pred": pred,
            "f1": round(f, 3), "order": round(o, 3),
            "exact": is_exact, "leak": leak,
        })

        if not args.quiet:
            flag = "✅" if is_exact else ("🟡" if f >= 0.8 else "❌")
            extra = f"  | fuite: {leak}" if leak else ""
            print(f"{flag} {row['text']}")
            print(f"    attendu : {gold}")
            print(f"    obtenu  : {pred}")
            print(f"    F1={f:.2f}  ordre={o:.2f}{extra}\n")

    n = len(rows)
    metrics = {
        "exact": exact / n, "exact_count": exact, "n": n,
        "f1": round(sum_f / n, 4), "order": round(sum_o / n, 4), "leaks": leaks,
    }
    by_phen_avg = {ph: round(sum(v) / len(v), 3) for ph, v in by_phen.items()}

    print("── Agrégats ──────────────────────────────")
    print(f"  Exact      : {exact}/{n} ({100*exact/n:.0f}%)")
    print(f"  F1 moyen   : {metrics['f1']:.3f}")
    print(f"  Ordre moyen: {metrics['order']:.3f}")
    print(f"  Fuites     : {leaks}/{n} phrase(s) avec mot-outil")
    print("\n── F1 moyen par phénomène ────────────────")
    for ph in sorted(by_phen_avg, key=by_phen_avg.get):
        print(f"  {ph:24s} {by_phen_avg[ph]:.2f}  (n={len(by_phen[ph])})")

    if args.save:
        commit, dirty = git_info()
        tokens = measure_prompt_tokens(client, args.model, rows, args.prompt)
        ds_sha = _sha(Path(args.dataset).read_bytes())
        snapshot = snapshot_dataset(Path(args.dataset), ds_sha)
        now = datetime.now()
        record = {
            "label": args.label,
            "timestamp": now.strftime("%Y-%m-%dT%H:%M:%S"),
            "time": now.strftime("%H:%M:%S"),
            "model": args.model,
            "model_digest": digest,
            "training": training,
            "git_commit": commit, "git_dirty": dirty,
            "prompt_mode": args.prompt,
            "prompt_hash": prompt_hash(args.prompt),
            "num_ctx": CONFIG.ollama.num_ctx,
            "few_shot": len(FEW_SHOT_EXAMPLES) if args.prompt == "prod" else 0,
            "prompt_tokens": tokens,
            "dataset": str(Path(args.dataset).relative_to(PROJECT)),
            "dataset_sha": ds_sha,
            "dataset_snapshot": str(snapshot.relative_to(PROJECT)),
            "dataset_n": n,
            "metrics": metrics,
            "by_phenomenon": by_phen_avg,
            "per_sentence": per_sentence,
        }
        path = save_record(record)
        warn = ""
        if tokens and tokens > CONFIG.ollama.num_ctx:
            warn = (f"\n  ⚠️  prompt {tokens} tokens > num_ctx {CONFIG.ollama.num_ctx} "
                    f": TRONCATURE probable.")
        print(f"\n💾  Record : {path.relative_to(PROJECT)}")
        print(f"    Tableau : {RESULTS_MD.relative_to(PROJECT)}{warn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
