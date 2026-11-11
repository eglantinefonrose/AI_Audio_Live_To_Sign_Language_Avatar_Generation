"""Relève les articulations de l'avatar, image par image, pour l'évaluation (Blender).

Pour chaque élément d'un fichier de travail JSON (signe isolé ou phrase en gloses), le
transfert sur l'avatar est fait exactement comme pour la vidéo (`avatar_rig.Retargeter`),
puis on lit la position **réelle** des os posés : épaules, coudes, poignets et les 21
points de chaque main, sous les mêmes noms que `pose3d.joints`. On mesure donc ce que
l'avatar fait, après le transfert, et non la trajectoire qu'on lui demande.

    Blender -b --factory-startup --python export_avatar_blender.py -- \\
        --travail travail.json --sortie avatar.npz [--avatar mon_avatar.vrm]

Fichier de travail : {"signes": ["MAISON", …], "phrases": [["MOI", "MAISON"], …]}.
Sortie .npz : `noms` (articulations), `signe::<ÉTIQUETTE>` et `phrase::<i>` (T, N, 3), et
`meta` (JSON : chronologie de chaque phrase, avatar). Lancé par `evaluate_avatar.py`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import bpy
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "avatar_animation"))
import avatar_rig  # noqa: E402
from avatar_rig import FINGER_JOINTS, FINGER_SEGMENTS, SIDES, THUMB_SEGMENTS  # noqa: E402
from sequence import assemble, load_index, load_sign  # noqa: E402


def joint_bones(version: int) -> dict[str, tuple[str, str]]:
    """Articulation (nom pose3d) -> (os humanoïde, "head" ou "tip")."""
    out = {}
    for side, s in SIDES.items():
        out[f"shoulder.{s}"] = (f"{side}UpperArm", "head")
        out[f"elbow.{s}"] = (f"{side}LowerArm", "head")
        out[f"wrist.{s}"] = (f"{side}Hand", "head")
        out[f"hand.{s}.0"] = (f"{side}Hand", "head")
        for finger, chain in FINGER_JOINTS.items():
            segs = THUMB_SEGMENTS[version] if finger == "Thumb" else FINGER_SEGMENTS
            for seg, k in zip(segs, chain):
                out[f"hand.{s}.{k}"] = (f"{side}{finger}{seg}", "head")
            out[f"hand.{s}.{chain[-1]}"] = (f"{side}{finger}{segs[-1]}", "tip")
    return out


def capture(arm, rt: avatar_rig.Retargeter, mapping: dict[str, str], wanted: dict,
            n: int) -> np.ndarray:
    """Positions monde (n, N, 3) des articulations `wanted`, sur les images 1..n."""
    scene, mw = bpy.context.scene, arm.matrix_world
    bones = arm.data.bones
    out = np.full((n, len(wanted), 3), np.nan)
    for f in range(n):
        scene.frame_set(f + 1)
        for i, (bone_h, where) in enumerate(wanted.values()):
            name = mapping.get(bone_h)
            if name is None:
                continue
            pb = arm.pose.bones[name]
            p = pb.head
            if where == "tip":
                # Pointe du doigt : direction de repos de l'os, tournée comme l'os posé.
                rot = pb.matrix.to_3x3() @ bones[name].matrix_local.to_3x3().inverted()
                p = pb.head + rot @ (rt.rest_dir[name] * bones[name].length)
            out[f, i] = tuple(mw @ p)
    return out


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(prog="export_avatar_blender.py")
    ap.add_argument("--travail", required=True)
    ap.add_argument("--sortie", required=True)
    ap.add_argument("--avatar", help="fichier .vrm (VRoid Studio) ; défaut : mannequin")
    args = ap.parse_args(argv)

    job = json.loads(Path(args.travail).read_text(encoding="utf-8"))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    index = load_index()
    if args.avatar:
        arm, mapping, version = avatar_rig.load_vrm(Path(args.avatar).expanduser().resolve())
    else:
        arm, mapping, version = avatar_rig.load_mannequin()
    rt = avatar_rig.Retargeter(arm, mapping, version)
    wanted = joint_bones(version)
    arrays, meta = {}, {"avatar": Path(args.avatar).name if args.avatar else "mannequin",
                        "version_vrm": version, "phrases": []}
    t0 = time.time()

    def run(joints: dict[str, np.ndarray], key: str) -> None:
        n = len(joints["hip"])
        act = rt.action(joints, key)
        arrays[key] = capture(arm, rt, mapping, wanted, n).astype(np.float32)
        bpy.data.actions.remove(act)

    for k, label in enumerate(job.get("signes", [])):
        run(load_sign(label, index), f"signe::{label}")
        if k % 50 == 0:
            print(f"[éval] signes {k}/{len(job['signes'])} ({time.time() - t0:.0f} s)", flush=True)
    for k, glosses in enumerate(job.get("phrases", [])):
        joints, timeline = assemble(glosses, index)
        run(joints, f"phrase::{k}")
        meta["phrases"].append({"gloses": glosses, "chronologie": [list(t) for t in timeline]})
    print(f"[éval] {len(arrays)} éléments relevés en {time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(args.sortie, noms=np.array(list(wanted)), meta=json.dumps(meta, ensure_ascii=False),
                        **arrays)


if __name__ == "__main__":
    main()
