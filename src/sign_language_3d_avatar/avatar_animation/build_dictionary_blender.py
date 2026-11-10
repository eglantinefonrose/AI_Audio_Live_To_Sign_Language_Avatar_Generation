"""Assemble le dictionnaire de signes sur un avatar : une action Blender par signe LSFB.

Entrées : les trajectoires nettoyées (`clean_signs.py`) et l'index (`sequence.py index`).
Sorties dans `data/dictionnaire/` (ignoré par git, dérivé de LSFB CC BY-NC-SA 4.0) :

- `dictionnaire_<avatar>.blend` : l'avatar et toutes les actions `SIGNE:<ÉTIQUETTE>` ;
- `dictionnaire_<avatar>.glb` : le même contenu en glTF (une animation par signe),
  réutilisable hors de Blender (three.js, Unity…) ;
- l'index (`index.json`) est complété avec ces fichiers et le nombre d'images par signe.

Avatar : mannequin intégré par défaut, ou un modèle VRoid Studio exporté en `.vrm` :

    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \\
        --python src/sign_language_3d_avatar/avatar_animation/build_dictionary_blender.py -- \\
        [--avatar chemin/vers/avatar.vrm] [--limite 10] [--sans-glb]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import avatar_rig  # noqa: E402
from sequence import DICT, INDEX, load_index, load_sign  # noqa: E402


def load_avatar(path: str | None):
    if path:
        p = Path(path).expanduser().resolve()
        arm, mapping, version = avatar_rig.load_vrm(p)
        return arm, mapping, version, p.stem
    arm, mapping, version = avatar_rig.load_mannequin()
    return arm, mapping, version, "mannequin"


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(prog="build_dictionary_blender.py")
    ap.add_argument("--avatar", help="fichier .vrm exporté de VRoid Studio (défaut : mannequin)")
    ap.add_argument("--limite", type=int, default=0, help="nombre de signes (0 = tous), pour tester")
    ap.add_argument("--sans-glb", action="store_true", help="ne pas exporter le .glb")
    args = ap.parse_args(argv)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    index = load_index()
    arm, mapping, version, name = load_avatar(args.avatar)
    missing = [h for h in ("leftUpperArm", "rightUpperArm", "leftHand", "rightHand", "rightIndexProximal")
               if h not in mapping]
    if missing:
        print(f"⚠️  os humanoïdes absents de l'avatar : {', '.join(missing)}")
    rt = avatar_rig.Retargeter(arm, mapping, version)
    bpy.context.scene.render.fps = index["fps"]

    labels = sorted(index["signes"])[: args.limite or None]
    t0, frames = time.time(), {}
    for i, label in enumerate(labels, 1):
        act = rt.action(load_sign(label, index), f"SIGNE:{label}")
        act["etiquette_lsfb"] = label
        frames[label] = int(index["signes"][label]["images"])
        if i % 50 == 0:
            print(f"  {i}/{len(labels)} signes ({time.time() - t0:.0f} s)")
    arm.animation_data.action = None

    DICT.mkdir(parents=True, exist_ok=True)
    blend = DICT / f"dictionnaire_{name}.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    out = {"blend": blend.name, "signes": len(labels), "version_vrm": version if args.avatar else None}
    if not args.sans_glb:
        glb = DICT / f"dictionnaire_{name}.glb"
        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.export_scene.gltf(filepath=str(glb), export_format="GLB", export_animations=True,
                                  export_animation_mode="ACTIONS", export_anim_slide_to_zero=True)
        out["glb"] = glb.name
    full = json.loads(INDEX.read_text(encoding="utf-8"))
    full.setdefault("avatars", {})[name] = out
    INDEX.write_text(json.dumps(full, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(labels)} signes sur l'avatar « {name} » en {time.time() - t0:.0f} s → "
          f"{', '.join(v for k, v in out.items() if k in ('blend', 'glb'))} (dans {DICT})")


if __name__ == "__main__":
    main()
