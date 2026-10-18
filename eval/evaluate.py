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
ventilation par **phénomène** (étiquettes du dataset) pour cibler les régressions.

Usage :
    uv run python eval/evaluate.py                 # modèle de config.py
    uv run python eval/evaluate.py --model qwen2.5:7b
    uv run python eval/evaluate.py --quiet          # agrégats seulement
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from ollama import Client

from config import CONFIG
from lsf_translator.prompts import GLOSS_JSON_SCHEMA, build_messages

DATASET = PROJECT / "eval" / "dataset.jsonl"

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


# -- appel LLM -----------------------------------------------------------
def translate(client: Client, model: str, text: str, language: str) -> list[str]:
    r = client.chat(
        model=model, messages=build_messages(text, language),
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


# -- boucle --------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Évaluation gloses LSF.")
    ap.add_argument("--model", default=CONFIG.ollama.model)
    ap.add_argument("--dataset", default=str(DATASET))
    ap.add_argument("--quiet", action="store_true", help="Agrégats seulement.")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.dataset).read_text(encoding="utf-8").splitlines() if l.strip()]
    client = Client(host=CONFIG.ollama.host)

    print(f"Modèle : {args.model}  —  {len(rows)} phrases\n")
    sum_f = sum_o = exact = leaks = 0
    by_phen: dict[str, list[float]] = {}

    for row in rows:
        gold, pred = row["glosses"], translate(
            client, args.model, row["text"], row.get("language", "fr")
        )
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

        if not args.quiet:
            flag = "✅" if is_exact else ("🟡" if f >= 0.8 else "❌")
            print(f"{flag} {row['text']}")
            print(f"    attendu : {gold}")
            print(f"    obtenu  : {pred}")
            extra = f"  | fuite: {leak}" if leak else ""
            print(f"    F1={f:.2f}  ordre={o:.2f}{extra}\n")

    n = len(rows)
    print("── Agrégats ──────────────────────────────")
    print(f"  Exact      : {exact}/{n} ({100*exact/n:.0f}%)")
    print(f"  F1 moyen   : {sum_f/n:.3f}")
    print(f"  Ordre moyen: {sum_o/n:.3f}")
    print(f"  Fuites     : {leaks}/{n} phrase(s) avec mot-outil")
    print("\n── F1 moyen par phénomène ────────────────")
    for ph in sorted(by_phen, key=lambda k: sum(by_phen[k]) / len(by_phen[k])):
        vals = by_phen[ph]
        print(f"  {ph:24s} {sum(vals)/len(vals):.2f}  (n={len(vals)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
