# Suivi des évaluations

Tableau de bord des runs d'évaluation LSF (généré par `eval/evaluate.py --save`). Une ligne par run ; `git*` = modifs non commitées au moment du run. Détail complet dans `eval/results/`.

| Date | Label | Modèle | git | Exact | F1 | Ordre | Fuites | Record |
|---|---|---|---|---|---|---|---|---|
| 2026-10-19 | gemma2 — prompt lexique+questions | gemma2:9b | 5ed97b6* | 31% | 0.957 | 0.821 | 0 | [2026-10-19T00-40-44__gemma2-9b__5ed97b6.json](results/2026-10-19T00-40-44__gemma2-9b__5ed97b6.json) |
| 2026-10-19 | qwen2.5:7b — même prompt | qwen2.5:7b | 5ed97b6* | 12% | 0.792 | 0.635 | 0 | [2026-10-19T00-41-33__qwen2.5-7b__5ed97b6.json](results/2026-10-19T00-41-33__qwen2.5-7b__5ed97b6.json) |
