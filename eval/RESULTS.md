# Suivi des évaluations

Tableau de bord des runs d'évaluation LSF (généré par `eval/evaluate.py --save`). Une ligne par run ; `git*` = modifs non commitées au moment du run. Détail complet dans `eval/results/`.

| Date | Heure | Label | Modèle | git | Dataset éval | Dataset train | Exact | F1 | Ordre | Fuites | Record |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-19 | 00:40:44 | gemma2 — prompt lexique+questions | gemma2:9b | 5ed97b6* | [3db947280fd3](datasets/3db947280fd3/) | — | 31% | 0.957 | 0.821 | 0 | [2026-10-19T00-40-44__gemma2-9b__5ed97b6.json](results/2026-10-19T00-40-44__gemma2-9b__5ed97b6.json) |
| 2026-10-19 | 00:41:33 | qwen2.5:7b — même prompt | qwen2.5:7b | 5ed97b6* | [3db947280fd3](datasets/3db947280fd3/) | — | 12% | 0.792 | 0.635 | 0 | [2026-10-19T00-41-33__qwen2.5-7b__5ed97b6.json](results/2026-10-19T00-41-33__qwen2.5-7b__5ed97b6.json) |
| 2026-10-19 | 10:42:29 | finetune v1 (qwen3b) | lsf-qwen3b:v1 (entraîné ~2026-10-19 01:54 (durée inconnue)) | 885ed73* | [3db947280fd3](datasets/3db947280fd3/) | [fa6131f5e0ee](../finetune/data/fa6131f5e0ee/) | 0% | 0.592 | 0.497 | 1 | [2026-10-19T10-42-29__lsf-qwen3b-v1__885ed73.json](results/2026-10-19T10-42-29__lsf-qwen3b-v1__885ed73.json) |
| 2026-10-19 | 14:04:51 | fine-tune v2 (dataset Claude Opus) | lsf-qwen3b:v2 (entraîné ~2026-10-19 12:07 (~20 min)) | 9f40c38* | [3db947280fd3](datasets/3db947280fd3/) | [e604d1cd6448](../finetune/data/e604d1cd6448/) | 6% | 0.686 | 0.613 | 0 | [2026-10-19T14-04-51__lsf-qwen3b-v2__9f40c38.json](results/2026-10-19T14-04-51__lsf-qwen3b-v2__9f40c38.json) |
| 2026-10-19 | 14:22:37 | base qwen2.5:3b — prompt prod | qwen2.5:3b | 9f40c38* | [3db947280fd3](datasets/3db947280fd3/) | — | 6% | 0.680 | 0.568 | 5 | [2026-10-19T14-22-37__qwen2.5-3b__9f40c38.json](results/2026-10-19T14-22-37__qwen2.5-3b__9f40c38.json) |
| 2026-10-19 | 14:22:51 | base qwen2.5:3b — règles seules | qwen2.5:3b | 9f40c38* | [3db947280fd3](datasets/3db947280fd3/) | — | 0% | 0.423 | 0.277 | 3 | [2026-10-19T14-22-51__qwen2.5-3b__9f40c38.json](results/2026-10-19T14-22-51__qwen2.5-3b__9f40c38.json) |
| 2026-10-19 | 14:23:20 | fine-tune v2 — règles seules (prompt d'entraînement) | lsf-qwen3b:v2 (entraîné ~2026-10-19 12:07 (~20 min)) | 9f40c38* | [3db947280fd3](datasets/3db947280fd3/) | [e604d1cd6448](../finetune/data/e604d1cd6448/) | 6% | 0.653 | 0.567 | 0 | [2026-10-19T14-23-20__lsf-qwen3b-v2__9f40c38.json](results/2026-10-19T14-23-20__lsf-qwen3b-v2__9f40c38.json) |
