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

Gemma 2 (repo séparé, licence Gemma, prompt sans rôle system) :
  uv run python finetune/publish_hf.py \
      --model-repo mon-user/lsf-gemma9b --version v1 \
      --adapters finetune/adapters-gemma9b \
      --eval eval/results/<…>__lsf-gemma9b-v1__<…>.json --eval <record règles seules> \
      --modelfile finetune/Modelfile.gemma9b \
      --gguf finetune/fused-gemma9b-gguf/lsf-gemma9b-v1-Q4_K_M.gguf \
      --dataset-repo mon-user/lsf-fr-gloss

⚠️  Licence : déduite du modèle de base (FAMILIES). Qwen2.5 = Apache-2.0 ; Gemma =
    Gemma Terms of Use (dérivés autorisés, mais la notice et les restrictions
    d'usage doivent accompagner les poids : la model card les reprend).
    Données 100 % synthétiques non validées → marque le run « expérimental ».

Dataset : chaque version poussée sur le repo dataset reçoit un tag = son train_sha,
et la model card pointe sur ce tag (données exactes de l'entraînement). Un modèle
entraîné sur une variante (ex. gemma_variant.py) publie et cite la SOURCE
(`derived_from`) : la variante s'en déduit de façon déterministe.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from snapshot_data import latest

PROJECT = Path(__file__).resolve().parent.parent
ADAPTERS_DIR = PROJECT / "finetune" / "adapters"
FUSED_DIR = PROJECT / "finetune" / "fused"
DATA_DIR = PROJECT / "finetune" / "data"
# Repli si training_meta.json est absent ; sinon la base vient de la provenance.
BASE_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"

# Licence et particularités de prompt, par famille du modèle de base.
FAMILIES = {
    "qwen": {
        "license": ["license: apache-2.0"],
        "notice": [],
        "prompt": ["Prompt d'entraînement : `system` = règles LSF (`TRAIN_SYSTEM`),",
                   "`user` = phrase à transcrire, réponse = `{\"glosses\": [...]}`."],
    },
    "gemma": {
        "license": ["license: gemma",
                    "license_link: https://ai.google.dev/gemma/terms"],
        "notice": [
            "## Licence",
            "",
            "Gemma is provided under and subject to the Gemma Terms of Use found at",
            "[ai.google.dev/gemma/terms](https://ai.google.dev/gemma/terms).",
            "",
            "Ce modèle est une **modification** de Gemma 2 (adaptateur LoRA entraîné",
            "sur des gloses LSF). Son usage est soumis aux mêmes conditions, dont la",
            "[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy),",
            "à transmettre à toute personne à qui tu redistribues ces poids.",
            "",
        ],
        "prompt": ["Gemma 2 n'a **pas de rôle `system`** : à l'entraînement, les règles LSF",
                   "(`TRAIN_SYSTEM`) sont placées en tête du **premier message user**, suivies",
                   "d'une ligne vide puis de la phrase. Le `Modelfile` fourni reproduit ce",
                   "rendu pour Ollama : envoie `system` + `user` normalement, il les fusionne."],
    },
}


def family(base: str) -> str:
    return "gemma" if "gemma" in base.lower() else "qwen"


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


def build_model_card(version: str, manifest: dict | None, records: list[dict],
                     dataset_repo: str | None, training: dict | None = None,
                     dataset_rev: str | None = None, extras: dict | None = None) -> str:
    """Construit un README.md (model card) à partir de la provenance disponible."""
    extras = extras or {}
    base = (training or {}).get("base_model") or BASE_MODEL
    fam = FAMILIES[family(base)]
    lines = [
        "---",
        *fam["license"],
        f"base_model: {base}",
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
        f"un dataset **synthétique** (distillation) au-dessus de `{base}`.",
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
        if manifest.get("variant"):
            lines.append(f"- **Variante** : `{manifest['variant']}` dérivée de "
                         f"`{manifest.get('derived_from', '?')}` "
                         f"({manifest.get('variant_note', 'finetune/gemma_variant.py')})")
    if training:
        d = training.get("duration_s")
        approx = " (approximatif, reconstitué)" if training.get("approximate") else ""
        hp = training.get("hyperparams") or {}
        lines += [
            f"- **Entraîné le** : {training.get('started_at') or '?'} → "
            f"{training.get('ended_at') or '?'}{approx}",
            f"- **Dataset d'entraînement (sha)** : `{training.get('train_sha') or '?'}`",
            f"- **Durée d'entraînement** : "
            f"{f'{d // 3600}h{(d % 3600) // 60:02d}m{d % 60:02d}s' if isinstance(d, int) else '?'}",
            f"- **Hyperparamètres** : " + (", ".join(f"{k}={v}" for k, v in hp.items()) or "?"),
            f"- **Val loss finale** : {training.get('final_val_loss', '?')}",
        ]
    if dataset_repo:
        url = f"https://huggingface.co/datasets/{dataset_repo}"
        if dataset_rev:
            url += f"/tree/{dataset_rev}"
        lines.append(f"- **Dataset** : [`{dataset_repo}`"
                     f"{f' @ {dataset_rev}' if dataset_rev else ''}]({url})")
    scored = [r for r in records if r.get("metrics")]
    if scored:
        modes = {"prod": "prod (règles + few-shot)", "rules-only": "règles seules (prompt d'entraînement)"}
        lines += [
            "",
            "## Évaluation",
            "",
            f"Mesuré avec `eval/evaluate.py` sur `{scored[0].get('dataset', '?')}` "
            f"(sha `{scored[0].get('dataset_sha', '?')}`, {scored[0].get('dataset_n', '?')} phrases).",
            "",
            "| Prompt | Exact | F1 | Ordre | Fuites |",
            "|---|---|---|---|---|",
        ]
        for r in scored:
            m = r["metrics"]
            lines.append(f"| {modes.get(r.get('prompt_mode'), r.get('prompt_mode') or '?')} | "
                         f"{m.get('exact', 0)*100:.0f}% | {m.get('f1', 0):.3f} | "
                         f"{m.get('order', 0):.3f} | {m.get('leaks', '?')} |")
        if not any(r.get("prompt_mode") == "rules-only" for r in scored):
            lines += ["",
                      "> ⚠️ Aucune éval avec le **prompt d'entraînement** (règles seules) :",
                      "> le score peut être sous- ou sur-évalué."]
    lines += ["", "## Format du prompt", "", *fam["prompt"], ""]
    if extras.get("gguf"):
        lines += [
            "## Usage (Ollama)",
            "",
            f"`{extras['gguf']}` (GGUF, llama.cpp) + `Modelfile` (template et règles LSF) :",
            "",
            "```bash",
            f"huggingface-cli download {extras.get('repo', '<ce-repo>')} --local-dir lsf-model",
            "cd lsf-model && ollama create lsf -f Modelfile",
            "```",
            "",
        ]
    lines += [
        "## Usage (MLX)",
        "",
        "```bash",
        f"uv run mlx_lm.generate --model {base} \\",
        "  --adapter-path adapters --prompt \"...\"",
        "```",
        "",
        *fam["notice"],
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
    ap = argparse.ArgumentParser(description="Publie poids + éval + dataset sur HuggingFace.")
    ap.add_argument("--model-repo", default=None,
                    help="Repo modèle, ex. mon-user/lsf-qwen3b (optionnel : omets pour publier le dataset seul).")
    ap.add_argument("--version", default=None,
                    help="Tag de version, ex. v1 (requis avec --model-repo).")
    ap.add_argument("--adapters", default=str(ADAPTERS_DIR),
                    help="Dossier de l'adaptateur LoRA à publier.")
    ap.add_argument("--eval", dest="eval_paths", action="append", default=[],
                    help="Record d'éval JSON de cette version (eval/results/...). "
                         "Répétable (ex. prompt prod + règles seules).")
    ap.add_argument("--manifest", default=None,
                    help="Manifeste du dataset (défaut : <dataset>/manifest.json).")
    ap.add_argument("--fused", nargs="?", const=str(FUSED_DIR), default=None,
                    help="Publier AUSSI le modèle fusionné (~6 Go). Lourd.")
    ap.add_argument("--gguf", default=None,
                    help="Publier AUSSI un GGUF prêt à servir (ex. Q4_K_M), dans gguf/.")
    ap.add_argument("--modelfile", default=None,
                    help="Modelfile Ollama à publier (FROM réécrit vers le GGUF publié).")
    ap.add_argument("--dataset-repo", default=None,
                    help="Repo dataset, ex. mon-user/lsf-fr-gloss (optionnel).")
    ap.add_argument("--data", default=None,
                    help="Dataset à publier (défaut : celui de l'entraînement, "
                         "train_snapshot, sinon le plus récent de finetune/data/).")
    ap.add_argument("--private", action="store_true", help="Repos privés.")
    ap.add_argument("--dry-run", action="store_true",
                    help="N'upload rien : montre ce qui serait poussé.")
    args = ap.parse_args()

    if not args.model_repo and not args.dataset_repo:
        sys.exit("Rien à publier : donne --model-repo et/ou --dataset-repo.")
    if args.model_repo and not args.version:
        sys.exit("--version est requis avec --model-repo (ex. --version v1).")

    records = [_load_json(Path(p)) for p in args.eval_paths]
    record = records[0] if records else None
    adapters = Path(args.adapters)

    # Provenance d'entraînement : celle du record d'éval (rattachée au tag
    # Ollama évalué) prime ; sinon celle écrite par train.sh à côté de l'adaptateur.
    training = (record or {}).get("training")
    meta_path = adapters / "training_meta.json"
    if not training and meta_path.exists():
        training = _load_json(meta_path)

    # Dataset : celui qui a réellement servi à l'entraînement, sinon le plus récent.
    snap = (training or {}).get("train_snapshot")
    if args.data:
        data_dir = Path(args.data)
    elif snap and (PROJECT / snap).is_dir():
        data_dir = PROJECT / snap
    else:
        data_dir = latest()
    manifest_path = Path(args.manifest) if args.manifest else data_dir / "manifest.json"
    args.manifest = str(manifest_path)
    manifest = _load_json(manifest_path) if manifest_path.exists() else None
    # Variante (ex. gemma_variant.py) : on publie et cite la SOURCE, dont la
    # variante se déduit de façon déterministe ; le manifeste de la variante reste
    # celui poussé avec le modèle.
    source = (manifest or {}).get("derived_from")
    if source and not args.data and (DATA_DIR / source).is_dir():
        data_dir = DATA_DIR / source
    args.data = str(data_dir)
    dataset_rev = data_dir.name if args.dataset_repo else None   # tag = train_sha

    card = None
    if args.model_repo:
        if not adapters.is_dir():
            sys.exit(f"Adaptateur introuvable : {adapters} (lance d'abord mlx_lm.lora)")
        if not training:
            print("⚠️  Aucune date d'entraînement (training_meta.json absent) : "
                  "entraîne via ./finetune/train.sh.")
        if args.modelfile and not args.gguf:
            sys.exit("--modelfile sans --gguf : le FROM pointerait sur des poids non publiés.")
        extras = {"repo": args.model_repo}
        if args.gguf:
            extras["gguf"] = f"gguf/{Path(args.gguf).name}"
        card = build_model_card(args.version, manifest, records, args.dataset_repo, training,
                                dataset_rev, extras)

    # Récapitulatif de ce qui part.
    print(f"── Publication ({'privé' if args.private else 'public'}) ──")
    if args.model_repo:
        print(f"  modèle → {args.model_repo} @ {args.version}")
        print(f"    adapters/       ← {adapters}")
        for path in args.eval_paths:
            print(f"    eval/{Path(path).name}  ← {path}")
        if manifest:
            print(f"    manifest.json   ← {args.manifest}")
        if args.fused:
            print(f"    fused/          ← {args.fused}  (⚠️ ~6 Go)")
        if args.gguf:
            size = Path(args.gguf).stat().st_size / 1e9
            print(f"    gguf/{Path(args.gguf).name}  ← {args.gguf}  ({size:.1f} Go)")
        if args.modelfile:
            print(f"    Modelfile       ← {args.modelfile}  (FROM → ./gguf/{Path(args.gguf).name})")
        print(f"    README.md       ← model card auto ({len(card)} car.)")
    if args.dataset_repo:
        print(f"  dataset → {args.dataset_repo} @ tag {dataset_rev}  ← {args.data}  (+ card auto)")

    if args.dry_run:
        tail = ("\n--- model card ---\n" + card) if card else ""
        print("\n(dry-run : rien n'a été poussé)" + tail)
        return 0

    api = _require_hub()

    # 1) Repo modèle + uploads (commits sur main).
    if args.model_repo:
        api.create_repo(args.model_repo, repo_type="model",
                        private=args.private, exist_ok=True)
        api.upload_folder(repo_id=args.model_repo, folder_path=str(adapters),
                          path_in_repo="adapters",
                          commit_message=f"{args.version}: adaptateur LoRA")
        api.upload_file(repo_id=args.model_repo, path_in_repo="README.md",
                        path_or_fileobj=card.encode("utf-8"),
                        commit_message=f"{args.version}: model card")
        for path in args.eval_paths:
            api.upload_file(repo_id=args.model_repo,
                            path_in_repo=f"eval/{Path(path).name}",
                            path_or_fileobj=path,
                            commit_message=f"{args.version}: éval")
        if manifest:
            api.upload_file(repo_id=args.model_repo, path_in_repo="manifest.json",
                            path_or_fileobj=args.manifest,
                            commit_message=f"{args.version}: provenance")
        if args.fused:
            api.upload_folder(repo_id=args.model_repo, folder_path=args.fused,
                              path_in_repo="fused",
                              commit_message=f"{args.version}: modèle fusionné")
        if args.gguf:
            api.upload_file(repo_id=args.model_repo, path_in_repo=extras["gguf"],
                            path_or_fileobj=args.gguf,
                            commit_message=f"{args.version}: GGUF")
        if args.modelfile:
            # FROM local (./fused-…) → GGUF publié, chemin relatif à la racine du repo.
            mf = Path(args.modelfile).read_text(encoding="utf-8").splitlines()
            mf = [f"FROM ./{extras['gguf']}" if l.startswith("FROM ") else l for l in mf]
            api.upload_file(repo_id=args.model_repo, path_in_repo="Modelfile",
                            path_or_fileobj="\n".join(mf).encode("utf-8") + b"\n",
                            commit_message=f"{args.version}: Modelfile Ollama")

        # 2) Tag immuable de la version (pointe sur le dernier commit de main).
        try:
            api.create_tag(args.model_repo, tag=args.version, repo_type="model")
            print(f"✅ tag {args.version} posé")
        except Exception as exc:  # tag déjà présent = on ne réécrit jamais un tag
            print(f"⚠️  tag non posé ({exc}). N'écrase jamais un tag : "
                  f"utilise une nouvelle version.")

    # 3) Dataset (repo séparé, optionnel).
    if args.dataset_repo:
        # Manifeste du dataset POUSSÉ (la source, si le modèle a appris sur une variante).
        dm = Path(args.data) / "manifest.json"
        data_manifest = _load_json(dm) if dm.exists() else manifest
        api.create_repo(args.dataset_repo, repo_type="dataset",
                        private=args.private, exist_ok=True)
        api.upload_folder(repo_id=args.dataset_repo, repo_type="dataset",
                          folder_path=args.data,
                          commit_message=f"dataset ({args.version or 'update'})")
        # Dataset card : déclare les splits, sinon le Viewer HF reste vide
        # (train/valid au format chat + pairs au format lisible = schémas mixtes).
        api.upload_file(repo_id=args.dataset_repo, repo_type="dataset",
                        path_in_repo="README.md",
                        path_or_fileobj=build_dataset_card(data_manifest).encode("utf-8"),
                        commit_message="dataset card + config splits")
        try:   # tag = train_sha : la model card pointe sur ces données exactes
            api.create_tag(args.dataset_repo, tag=dataset_rev, repo_type="dataset")
        except Exception as exc:   # déjà taggé (même contenu) : on ne réécrit pas
            print(f"⚠️  tag dataset {dataset_rev} non posé ({exc}).")
        print(f"✅ dataset + card poussés → {args.dataset_repo} @ {dataset_rev}")

    print("")
    if args.model_repo:
        print(f"🔗 modèle  : https://huggingface.co/{args.model_repo}/tree/{args.version}")
    if args.dataset_repo:
        print(f"🔗 dataset : https://huggingface.co/datasets/{args.dataset_repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
