"""Compare deux runs d'évaluation archivés par `evaluate.py --save`.

Affiche les deltas de métriques agrégées, la ventilation par phénomène, et
surtout les phrases dont la sortie a changé (corrigées ✅ / régressées ❌ /
améliorées ↑ / dégradées ↓) — c'est ce qui rend un changement de prompt ou de
modèle *actionnable*.

Usage :
    uv run python eval/compare.py <runA.json> <runB.json>
    uv run python eval/compare.py A.json B.json     # noms résolus sous eval/results/

A = référence (avant), B = nouveau (après). Les deltas sont B - A.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT / "eval" / "results"


def load(arg: str) -> dict:
    p = Path(arg)
    if not p.exists():
        p = RESULTS_DIR / arg          # tolère un simple nom de fichier
    if not p.exists():
        sys.exit(f"Introuvable : {arg}")
    return json.loads(p.read_text(encoding="utf-8"))


def _delta(a: float, b: float, pct: bool = False) -> str:
    d = b - a
    arrow = "▲" if d > 0 else ("▼" if d < 0 else "=")
    if pct:
        return f"{a*100:.0f}% → {b*100:.0f}%  {arrow}{abs(d)*100:+.0f}pt".replace("+", "")
    return f"{a:.3f} → {b:.3f}  {arrow}{d:+.3f}"


def header(tag: str, r: dict) -> None:
    git = (r.get("git_commit") or "—") + ("*" if r.get("git_dirty") else "")
    print(f"  {tag}: {r.get('label','?')}")
    print(f"      modèle={r.get('model')} ({r.get('model_digest') or '?'})  "
          f"git={git}  prompt={r.get('prompt_hash','?')}  "
          f"few_shot={r.get('few_shot','?')}  mode={r.get('prompt_mode', 'prod')}")
    t = r.get("training")
    if t:
        d = t.get("duration_s")
        approx = "~" if t.get("approximate") else ""
        print(f"      entraîné {approx}{str(t.get('started_at','?'))[:16].replace('T', ' ')}"
              f" ({approx}{d // 60 if isinstance(d, int) else '?'} min, "
              f"val loss {t.get('final_val_loss', '?')})")


def main() -> int:
    if len(sys.argv) != 3:
        sys.exit("Usage: compare.py <runA.json> <runB.json>")
    A, B = load(sys.argv[1]), load(sys.argv[2])

    print("══ Runs comparés (A = avant, B = après) ══════════════════")
    header("A", A)
    header("B", B)

    # Garde-fous : on ne compare que ce qui est comparable.
    if A.get("dataset_sha") != B.get("dataset_sha"):
        print("\n  ⚠️  DATASETS DIFFÉRENTS (sha distinct) : comparaison peu fiable.")
    if A.get("prompt_hash") == B.get("prompt_hash") and A.get("model") == B.get("model"):
        print("\n  ℹ️  Prompt ET modèle identiques : écarts dus au seul non-déterminisme ?")

    ma, mb = A["metrics"], B["metrics"]
    print("\n══ Métriques agrégées (Δ = B − A) ════════════════════════")
    print(f"  Exact  : {_delta(ma['exact'], mb['exact'], pct=True)}")
    print(f"  F1     : {_delta(ma['f1'], mb['f1'])}")
    print(f"  Ordre  : {_delta(ma['order'], mb['order'])}")
    print(f"  Fuites : {ma['leaks']} → {mb['leaks']}")

    # Ventilation par phénomène (sur l'union des clés).
    pa, pb = A.get("by_phenomenon", {}), B.get("by_phenomenon", {})
    changed_ph = sorted(k for k in set(pa) | set(pb) if pa.get(k) != pb.get(k))
    if changed_ph:
        print("\n══ Phénomènes qui ont bougé ══════════════════════════════")
        for k in changed_ph:
            va, vb = pa.get(k), pb.get(k)
            if va is None or vb is None:
                print(f"  {k:24s} {'(absent)' if va is None else f'{va:.2f}'} → "
                      f"{'(absent)' if vb is None else f'{vb:.2f}'}")
            else:
                print(f"  {k:24s} {_delta(va, vb)}")

    # Changements par phrase (appariés par texte).
    a_by = {s["text"]: s for s in A.get("per_sentence", [])}
    b_by = {s["text"]: s for s in B.get("per_sentence", [])}
    buckets: dict[str, list] = {"✅ corrigée": [], "❌ régressée": [],
                                "↑ améliorée": [], "↓ dégradée": [], "~ modifiée": []}
    for text in a_by.keys() & b_by.keys():
        a, b = a_by[text], b_by[text]
        if a["pred"] == b["pred"]:
            continue
        if not a["exact"] and b["exact"]:
            k = "✅ corrigée"
        elif a["exact"] and not b["exact"]:
            k = "❌ régressée"
        elif b["f1"] > a["f1"]:
            k = "↑ améliorée"
        elif b["f1"] < a["f1"]:
            k = "↓ dégradée"
        else:
            k = "~ modifiée"
        buckets[k].append((text, a, b))

    print("\n══ Changements par phrase ════════════════════════════════")
    total = sum(len(v) for v in buckets.values())
    if not total:
        print("  (aucune phrase n'a changé)")
    for k, items in buckets.items():
        for text, a, b in items:
            print(f"  {k} : {text}")
            print(f"      gold : {a['gold']}")
            print(f"      A    : {a['pred']}  (F1 {a['f1']:.2f})")
            print(f"      B    : {b['pred']}  (F1 {b['f1']:.2f})")

    only_a = a_by.keys() - b_by.keys()
    only_b = b_by.keys() - a_by.keys()
    if only_a or only_b:
        print("\n  ⚠️  Phrases non appariées (dataset modifié) : "
              f"{len(only_a)} seulement dans A, {len(only_b)} seulement dans B.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
