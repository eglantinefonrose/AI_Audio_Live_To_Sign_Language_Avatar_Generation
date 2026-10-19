"""Publie un entraînement sur HuggingFace : poids (adaptateur) + éval + provenance.

Idée directrice : UN repo MODÈLE versionné par TAGS (v1, v2, …). À chaque
entraînement on pousse l'adaptateur LoRA + le record d'éval + le manifeste de
provenance, puis on pose un tag immuable. `main` = dernière version, chaque tag
= un entraînement figé (même logique que les tags Ollama / `eval/results/`).

Le dataset synthétique va dans un repo DATASET séparé (optionnel, --dataset-repo).

Pourquoi l'ADAPTATEUR et pas le modèle fusionné :
  - l'adaptateur (~8 Mo) est le vrai produit de l'entraînement ;
  - le fused (~6 Go) se régénère depuis `base + adaptateur` via `mlx_lm.fuse`.
  Ajoute --fused pour publier aussi le modèle prêt à servir (coûteux en espace).

Prérequis :
  uv pip install huggingface_hub
  uv run huggingface-cli login          # token avec accès WRITE

Exemple :
  uv run python finetune/publish_hf.py \
      --model-repo mon-user/lsf-qwen3b \
      --version v1 \
      --eval eval/results/2026-10-19T10-12-40__lsf-qwen3b-v1__885ed73.json \
      --dataset-repo mon-user/lsf-fr-gloss \
      --private

⚠️  Le modèle de base (Qwen2.5-3B-Instruct) est Apache-2.0 : les dérivés sont
    autorisés, mentionne la base et la licence dans la model card (fait ci-dessous).
    Données 100 % synthétiques non validées → marque le run « expérimental ».
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT / "finetune" / "data"
ADAPTERS_DIR = PROJECT / "finetune" / "adapters"
FUSED_DIR = PROJECT / "finetune" / "fused"
BASE_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"


def _require_hub():
    try:
        from huggingface_hub import HfApi  # noqa: F401
    except ImportError:
        sys.exit("huggingface_hub manquant. Installe-le :\n"
                 "  uv pip install huggingface_hub\n"
                 "  uv run huggingface-cli login")
    from huggingface_hub import HfApi
    return HfApi()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_model_card(version: str, manifest: dict | None, record: dict | None,
                     dataset_repo: str | None) -> str:
    """Construit un README.md (model card) à partir de la provenance disponible."""
    m = record.get("metrics", {}) if record else {}
    lines = [
        "---",
        "license: apache-2.0",
        f"base_model: {BASE_MODEL}",
        "tags:",
        "  - lsf",
        "  - gloss",
        "  - lora",
        "  - experimental",
        "---",
        "",
        f"# LSF gloss translator — `{version}`",
        "",
        "Adaptateur LoRA pour la traduction **français → glose LSF**, entraîné sur",
        f"un dataset **synthétique** (distillation) au-dessus de `{BASE_MODEL}`.",
        "",
        "> ⚠️ **Expérimental.** Données générées par un modèle professeur, **non",
        "> validées par un locuteur LSF**. À ne pas utiliser comme vérité terrain.",
        "",
        "## Provenance",
        "",
    ]
    if manifest:
        lines += [
            f"- **Professeur** : `{manifest.get('teacher', '?')}`",
            f"- **prompt_hash** : `{manifest.get('prompt_hash', '?')}`",
            f"- **Exemples** : {manifest.get('train', '?')} train / "
            f"{manifest.get('valid', '?')} valid (accepté {manifest.get('accepted', '?')})",
            f"- **Dataset synthétique** : {manifest.get('synthetic', True)}, "
            f"validé humain : {manifest.get('human_validated', False)}",
        ]
    if dataset_repo:
        lines.append(f"- **Dataset** : [`{dataset_repo}`](https://huggingface.co/datasets/{dataset_repo})")
    if m:
        lines += [
            "",
            "## Évaluation",
            "",
            f"Mesuré avec `eval/evaluate.py` sur `{record.get('dataset', '?')}` "
            f"({record.get('dataset_n', '?')} phrases).",
            "",
            "| Exact | F1 | Ordre | Fuites |",
            "|---|---|---|---|",
            f"| {m.get('exact', 0)*100:.0f}% | {m.get('f1', 0):.3f} | "
            f"{m.get('order', 0):.3f} | {m.get('leaks', '?')} |",
            "",
            "> ⚠️ Vérifie que l'éval a utilisé le **même prompt** qu'à l'entraînement",
            "> (règles seules, sans few-shot), sinon le score est sous-évalué.",
        ]
    lines += [
        "",
        "## Usage (MLX)",
        "",
        "```bash",
        f"uv run mlx_lm.generate --model {BASE_MODEL} \\",
        "  --adapter-path adapters --prompt \"...\"",
        "```",
        "",
        f"_Publié le {datetime.now(timezone.utc).strftime('%Y-%m-%d')}._",
        "",
    ]
    return "\n".join(lines)


def build_dataset_card(manifest: dict | None) -> str:
    """Dataset card HF : le bloc `configs` est ce qui fait marcher le Viewer.

    Sans cette config, HF tente d'auto-détecter les splits et échoue ici car
    train/valid (format chat) et pairs (format lisible) ont des schémas mixtes.
    """
    m = manifest or {}
    front = [
        "---",
        "license: apache-2.0",
        "language:",
        "- fr",
        "tags:",
        "- lsf",
        "- gloss",
        "- synthetic",
        "pretty_name: FR to LSF gloss (synthetic)",
        "configs:",
        "- config_name: default",
        "  data_files:",
        "  - split: train",
        "    path: train.jsonl",
        "  - split: validation",
        "    path: valid.jsonl",
        "- config_name: readable",
        "  data_files:",
        "  - split: full",
        "    path: pairs.jsonl",
        "---",
        "",
        "# Français -> glose LSF (synthétique)",
        "",
        "Dataset **synthétique** (distillation) pour fine-tuner un traducteur",
        "français -> glose LSF. Généré par un modèle professeur, **non validé par",
        "un locuteur LSF** — expérimental, à ne pas prendre comme vérité terrain.",
        "",
        "## Contenu",
        "",
        "- **default** (`train.jsonl` / `valid.jsonl`) : format chat prêt pour",
        "  l'entraînement (system = règles LSF, user = phrase, assistant = gloses).",
        "- **readable** (`pairs.jsonl`) : mêmes paires en lisible `{text, glosses, phenomenon}`.",
    ]
    if m:
        front += [
            "",
            "## Provenance",
            "",
            f"- Professeur : `{m.get('teacher', '?')}`",
            f"- prompt_hash : `{m.get('prompt_hash', '?')}`",
            f"- Exemples : {m.get('train', '?')} train / {m.get('valid', '?')} valid "
            f"(accepté {m.get('accepted', '?')} sur {m.get('generated', '?')} générés)",
            f"- Synthétique : {m.get('synthetic', True)} · "
            f"validé humain : {m.get('human_validated', False)}",
        ]
    return "\n".join(front) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Publie poids + éval sur HuggingFace.")
    ap.add_argument("--model-repo", required=True,
                    help="Repo modèle, ex. mon-user/lsf-qwen3b")
    ap.add_argument("--version", required=True, help="Tag de version, ex. v1")
    ap.add_argument("--adapters", default=str(ADAPTERS_DIR),
                    help="Dossier de l'adaptateur LoRA à publier.")
    ap.add_argument("--eval", dest="eval_path", default=None,
                    help="Record d'éval JSON de cette version (eval/results/...).")
    ap.add_argument("--manifest", default=str(DATA_DIR / "manifest.json"),
                    help="Manifeste du dataset (provenance).")
    ap.add_argument("--fused", nargs="?", const=str(FUSED_DIR), default=None,
                    help="Publier AUSSI le modèle fusionné (~6 Go). Lourd.")
    ap.add_argument("--dataset-repo", default=None,
                    help="Repo dataset, ex. mon-user/lsf-fr-gloss (optionnel).")
    ap.add_argument("--data", default=str(DATA_DIR),
                    help="Dossier dataset à publier dans --dataset-repo.")
    ap.add_argument("--private", action="store_true", help="Repos privés.")
    ap.add_argument("--dry-run", action="store_true",
                    help="N'upload rien : montre ce qui serait poussé.")
    args = ap.parse_args()

    adapters = Path(args.adapters)
    if not adapters.is_dir():
        sys.exit(f"Adaptateur introuvable : {adapters} (lance d'abord mlx_lm.lora)")

    manifest = _load_json(Path(args.manifest)) if Path(args.manifest).exists() else None
    record = _load_json(Path(args.eval_path)) if args.eval_path else None
    card = build_model_card(args.version, manifest, record, args.dataset_repo)

    # Récapitulatif de ce qui part.
    print(f"── Publication {args.model_repo} @ {args.version} "
          f"({'privé' if args.private else 'public'}) ──")
    print(f"  adapters/         ← {adapters}")
    if record:
        print(f"  eval/{Path(args.eval_path).name}  ← {args.eval_path}")
    if manifest:
        print(f"  manifest.json     ← {args.manifest}")
    if args.fused:
        print(f"  fused/            ← {args.fused}  (⚠️ ~6 Go)")
    if args.dataset_repo:
        print(f"  dataset → {args.dataset_repo}  ← {args.data}")
    print(f"  README.md         ← model card auto ({len(card)} car.)")

    if args.dry_run:
        print("\n(dry-run : rien n'a été poussé)\n--- model card ---\n" + card)
        return 0

    api = _require_hub()

    # 1) Repo modèle + uploads (commits sur main).
    api.create_repo(args.model_repo, repo_type="model",
                    private=args.private, exist_ok=True)
    api.upload_folder(repo_id=args.model_repo, folder_path=str(adapters),
                      path_in_repo="adapters",
                      commit_message=f"{args.version}: adaptateur LoRA")
    api.upload_file(repo_id=args.model_repo, path_in_repo="README.md",
                    path_or_fileobj=card.encode("utf-8"),
                    commit_message=f"{args.version}: model card")
    if record:
        api.upload_file(repo_id=args.model_repo,
                        path_in_repo=f"eval/{Path(args.eval_path).name}",
                        path_or_fileobj=args.eval_path,
                        commit_message=f"{args.version}: éval")
    if manifest:
        api.upload_file(repo_id=args.model_repo, path_in_repo="manifest.json",
                        path_or_fileobj=args.manifest,
                        commit_message=f"{args.version}: provenance")
    if args.fused:
        api.upload_folder(repo_id=args.model_repo, folder_path=args.fused,
                          path_in_repo="fused",
                          commit_message=f"{args.version}: modèle fusionné")

    # 2) Tag immuable de la version (pointe sur le dernier commit de main).
    try:
        api.create_tag(args.model_repo, tag=args.version, repo_type="model")
        print(f"✅ tag {args.version} posé")
    except Exception as exc:  # tag déjà présent = on ne réécrit jamais un tag
        print(f"⚠️  tag non posé ({exc}). N'écrase jamais un tag : "
              f"utilise une nouvelle version.")

    # 3) Dataset (repo séparé, optionnel).
    if args.dataset_repo:
        api.create_repo(args.dataset_repo, repo_type="dataset",
                        private=args.private, exist_ok=True)
        api.upload_folder(repo_id=args.dataset_repo, repo_type="dataset",
                          folder_path=args.data,
                          commit_message=f"dataset ({args.version})")
        # Dataset card : déclare les splits, sinon le Viewer HF reste vide
        # (train/valid au format chat + pairs au format lisible = schémas mixtes).
        api.upload_file(repo_id=args.dataset_repo, repo_type="dataset",
                        path_in_repo="README.md",
                        path_or_fileobj=build_dataset_card(manifest).encode("utf-8"),
                        commit_message="dataset card + config splits")
        print(f"✅ dataset + card poussés → {args.dataset_repo}")

    print(f"\n🔗 https://huggingface.co/{args.model_repo}/tree/{args.version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
