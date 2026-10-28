"""Retargeting des poses LSFB (MediaPipe) vers un squelette humanoïde Blender.

Pour chaque instance : un squelette (torse, bras, mains à 5 doigts, tête) est
construit, ses os suivent les points MediaPipe (contraintes « Damped Track »,
longueurs d'os fixes), puis l'animation est cuite en rotations et exportée :

    src/sign_language_3d_avatar/avatar_animation/data/retarget/<GLOSE>/<instance>.glb   squelette + maillage simple + animation
    src/sign_language_3d_avatar/avatar_animation/data/retarget/<GLOSE>/<instance>.bvh   animation seule (réutilisable dans Unity, etc.)
    src/sign_language_3d_avatar/avatar_animation/data/retarget/<GLOSE>/<instance>.mp4   aperçu vidéo de face (.profil.mp4 avec --vues face,profil)

Lancement (Blender sans interface ; les arguments viennent après `--`) :

    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \\
        --python src/sign_language_3d_avatar/avatar_animation/retarget_blender.py -- --words MAISON,MERCI --par-glose 1

Les sorties dérivent de données CC BY-NC-SA 4.0 : même licence, jamais commitées.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

BASE = Path(__file__).resolve().parent.parent   # src/sign_language_3d_avatar/
ROOT = BASE.parent                               # src/ (lsf_translator/, eval/)
DATA = BASE / "dataset_generation" / "data" / "lsfb"
OUT = BASE / "avatar_animation" / "data" / "retarget"
FPS = 50                      # LSFB : 50 images/s (frames ≈ 49,5 × durée)
SHOULDER_WIDTH = 0.38         # m : échelle du squelette généré

# MediaPipe Pose : indices utiles. « L »/« R » = gauche/droite du signeur.
NOSE, EAR_L, EAR_R = 0, 7, 8
SHOULDER = {"L": 11, "R": 12}
ELBOW = {"L": 13, "R": 14}
WRIST = {"L": 15, "R": 16}
HIP_L, HIP_R = 23, 24
# Proportions anthropométriques (÷ largeur d'épaules) pour retrouver la profondeur.
UPPER_ARM, FOREARM = 0.80, 0.68
# MediaPipe Hands : chaînes de doigts (base -> pointe).
FINGERS = {"thumb": (1, 2, 3, 4), "index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
           "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}


# ---------------------------------------------------------------- points 3D

def to_blender(xyz: np.ndarray, aspect: float) -> np.ndarray:
    """Image MediaPipe (x à droite, y en bas, z vers la caméra < 0) -> Blender (Z en haut, caméra en -Y)."""
    out = np.empty_like(xyz)
    out[..., 0] = xyz[..., 0] * aspect
    out[..., 1] = xyz[..., 2] * aspect
    out[..., 2] = -xyz[..., 1]
    return out


def lift(parent: np.ndarray, child: np.ndarray, length: float) -> np.ndarray:
    """Recalcule la profondeur de `child` pour que |child - parent| = length (signe pris de MediaPipe)."""
    d = child - parent
    planar = d[:, 0] ** 2 + d[:, 2] ** 2
    depth = np.sqrt(np.clip(length ** 2 - planar, 0.0, None))
    sign = np.where(d[:, 1] < 0, -1.0, 1.0)
    out = child.copy()
    out[:, 1] = parent[:, 1] + sign * depth
    return out


def joints(pose: np.ndarray, hands: dict[str, np.ndarray], aspect: float) -> dict[str, np.ndarray]:
    """Positions 3D (T, 3) de chaque articulation, en mètres, bassin à l'origine."""
    p = to_blender(pose.astype(np.float64), aspect)
    hip = (p[:, HIP_L] + p[:, HIP_R]) / 2
    sw = np.median(np.linalg.norm(p[:, SHOULDER["L"]] - p[:, SHOULDER["R"]], axis=1))
    scale = SHOULDER_WIDTH / sw
    p = (p - hip[:, None]) * scale
    # Profondeur MediaPipe du torse et du visage peu fiable (la tête part en avant) :
    # torse droit (on garde seulement sa rotation), tête au-dessus du torse.
    sh_depth = (p[:, SHOULDER["L"], 1] + p[:, SHOULDER["R"], 1]) / 2
    p[:, :, 1] -= sh_depth[:, None]
    j = {"hip": np.zeros((len(p), 3)), "head": (p[:, EAR_L] + p[:, EAR_R]) / 2}
    j["head"][:, 1] = 0.0
    j["nose"] = lift(j["head"], p[:, NOSE], 0.10)
    for s in "LR":
        j[f"shoulder.{s}"] = p[:, SHOULDER[s]]
        # La profondeur MediaPipe des bras est très bruitée : on la reconstruit
        # à partir de longueurs d'os plausibles (au moins la longueur 2D vue).
        for name, a, b, ratio in (("elbow", SHOULDER[s], ELBOW[s], UPPER_ARM),
                                  ("wrist", ELBOW[s], WRIST[s], FOREARM)):
            parent = j[f"shoulder.{s}"] if name == "elbow" else j[f"elbow.{s}"]
            child = p[:, b] - p[:, a] + parent
            planar = np.linalg.norm((child - parent)[:, [0, 2]], axis=1)
            length = max(ratio * SHOULDER_WIDTH, float(np.percentile(planar, 98)))
            j[f"{name}.{s}"] = lift(parent, child, length)
        # Main : forme 3D de MediaPipe Hands, collée au poignet du bras.
        h = to_blender(hands[s].astype(np.float64), aspect) * scale
        h = h - h[:, :1] + j[f"wrist.{s}"][:, None]
        for k in range(21):
            j[f"hand.{s}.{k}"] = h[:, k]
    j["chest"] = (j["shoulder.L"] + j["shoulder.R"]) / 2
    return j


# --------------------------------------------------------------- squelette

def bone_specs() -> list[tuple[str, str | None, str, str]]:
    """(os, parent, articulation de tête, articulation visée)."""
    specs = [("spine", None, "hip", "chest"), ("neck", "spine", "chest", "head"),
             ("head", "neck", "head", "nose")]
    for s in "LR":
        h = f"hand.{s}"
        specs += [(f"clavicle.{s}", "spine", "chest", f"shoulder.{s}"),
                  (f"upper_arm.{s}", f"clavicle.{s}", f"shoulder.{s}", f"elbow.{s}"),
                  (f"forearm.{s}", f"upper_arm.{s}", f"elbow.{s}", f"wrist.{s}"),
                  (f"hand.{s}", f"forearm.{s}", f"wrist.{s}", f"{h}.9")]
        for finger, chain in FINGERS.items():
            parent = f"hand.{s}"
            for a, b in zip(chain, chain[1:]):
                name = f"{finger}.{chain.index(a) + 1}.{s}"
                specs.append((name, parent, f"{h}.{a}", f"{h}.{b}"))
                parent = name
    return specs


def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def build_rig(j: dict[str, np.ndarray], name: str):
    """Armature au repos = première image ; empties animés = cibles des contraintes."""
    targets = bpy.data.objects.new("targets", None)
    bpy.context.scene.collection.objects.link(targets)
    empties = {}
    for key, xyz in j.items():
        e = bpy.data.objects.new(f"T_{key}", None)
        e.parent = targets
        bpy.context.scene.collection.objects.link(e)
        for f, v in enumerate(xyz):
            e.location = v
            e.keyframe_insert("location", frame=f + 1)
        empties[key] = e

    arm_data = bpy.data.armatures.new(name)
    arm = bpy.data.objects.new(name, arm_data)
    bpy.context.scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    specs = bone_specs()
    for bone, parent, head, tail in specs:
        eb = arm_data.edit_bones.new(bone)
        eb.head, eb.tail = Vector(j[head][0]), Vector(j[tail][0])
        if eb.length < 1e-4:
            eb.tail = eb.head + Vector((0, 0, 0.01))
        if bone.startswith("hand."):
            s = bone[-1]
            # Axe Z de la main vers l'index : sert de référence de roulis.
            eb.align_roll(Vector(j[f"hand.{s}.5"][0] - j[f"hand.{s}.17"][0]))
        if parent:
            eb.parent = arm_data.edit_bones[parent]
    bpy.ops.object.mode_set(mode="POSE")
    for bone, _, _, tail in specs:
        pb = arm.pose.bones[bone]
        c = pb.constraints.new("DAMPED_TRACK")
        c.target = empties[tail]
        if bone.startswith("hand."):
            lt = pb.constraints.new("LOCKED_TRACK")
            lt.target, lt.track_axis, lt.lock_axis = empties[f"hand.{bone[-1]}.5"], "TRACK_Z", "LOCK_Y"
        if bone == "spine":
            pb.constraints.new("COPY_LOCATION").target = empties["hip"]
    return arm, targets


def bake(arm, n_frames: int) -> None:
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.select_all(action="SELECT")
    bpy.ops.nla.bake(frame_start=1, frame_end=n_frames, only_selected=True, visual_keying=True,
                     clear_constraints=True, use_current_action=True, bake_types={"POSE"})
    bpy.ops.object.mode_set(mode="OBJECT")


def add_body(arm) -> None:
    """Maillage minimal : un cylindre par os, une sphère pour la tête, attachés aux os."""
    skin = bpy.data.materials.new("skin")
    skin.diffuse_color = (0.85, 0.66, 0.53, 1)
    for bone in arm.data.bones:
        length = bone.length
        if bone.name == "head":
            bpy.ops.mesh.primitive_uv_sphere_add(radius=0.11, segments=24, ring_count=12)
        else:
            radius = 0.007 if "." in bone.name and bone.name.split(".")[0] in FINGERS else 0.03
            radius = 0.06 if bone.name == "spine" else radius
            radius = 0.02 if bone.name.startswith("hand.") else radius
            bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=length, vertices=12)
        obj = bpy.context.active_object
        obj.name = f"M_{bone.name}"
        obj.data.materials.append(skin)
        rest = arm.matrix_world @ bone.matrix_local
        offset = Vector((0, 0.11 * 0.3, 0)) if bone.name == "head" else Vector((0, length / 2, 0))
        # Cylindre aligné sur l'axe Y de l'os, centré au milieu de l'os.
        world = rest @ Matrix.Translation(offset) @ Matrix.Rotation(math.radians(90), 4, "X")
        obj.parent, obj.parent_type, obj.parent_bone = arm, "BONE", bone.name
        bpy.context.view_layer.update()
        obj.matrix_world = world


def setup_render(n_frames: int, path: Path, view: str) -> None:
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, n_frames
    scene.render.fps = FPS
    try:
        scene.render.engine = "BLENDER_WORKBENCH"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = scene.render.resolution_y = 480
    world = bpy.data.worlds.new("w")
    world.color = (0.93, 0.93, 0.93)
    scene.world = world
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    cam.data.type, cam.data.ortho_scale = "ORTHO", 1.3
    if view == "profil":  # vu depuis la gauche du signeur
        cam.location, cam.rotation_euler = (3, 0, 0.35), (math.radians(90), 0, math.radians(90))
    else:
        cam.location, cam.rotation_euler = (0, -3, 0.35), (math.radians(90), 0, 0)
    scene.collection.objects.link(cam)
    scene.camera = cam
    s = scene.render.image_settings
    if hasattr(s, "media_type"):
        s.media_type = "VIDEO"
    s.file_format = "FFMPEG"
    scene.render.ffmpeg.format, scene.render.ffmpeg.codec = "MPEG4", "H264"
    scene.render.filepath = str(path)


def retarget(instance: dict, gloss: str, aspect: float, views: list[str]) -> Path:
    files = {p.split("/")[1]: DATA / p for p in instance["fichiers"].split()}
    pose = np.load(files["pose"], allow_pickle=False)
    hands = {"L": np.load(files["left_hand"], allow_pickle=False),
             "R": np.load(files["right_hand"], allow_pickle=False)}
    j = joints(pose, hands, aspect)
    n = len(pose)

    reset_scene()
    bpy.context.scene.render.fps = FPS
    arm, targets = build_rig(j, "lsfb_signer")
    bpy.context.scene.frame_start, bpy.context.scene.frame_end = 1, n
    bake(arm, n)
    for obj in [targets, *targets.children]:
        bpy.data.objects.remove(obj)
    add_body(arm)

    out = OUT / gloss / instance["instance_id"]
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.export_anim.bvh(filepath=str(out.with_suffix(".bvh")), frame_start=1, frame_end=n)
    bpy.ops.export_scene.gltf(filepath=str(out.with_suffix(".glb")), export_format="GLB",
                              export_animations=True)
    for view in views:
        suffix = ".mp4" if view == "face" else f".{view}.mp4"
        setup_render(n, out.with_suffix(suffix), view)
        bpy.ops.render.render(animation=True)
    return out


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(prog="retarget_blender.py")
    ap.add_argument("--words", required=True, help="gloses du projet (colonne glose_projet du manifeste)")
    ap.add_argument("--par-glose", type=int, default=1, help="instances par glose")
    ap.add_argument("--aspect", type=float, default=4 / 3, help="largeur/hauteur des vidéos source")
    ap.add_argument("--vues", default="face", help="aperçus .mp4 : face, profil, face,profil ou aucune")
    args = ap.parse_args(argv)

    with open(DATA / "manifest.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for word in args.words.split(","):
        chosen = [r for r in rows if r["glose_projet"] == word][: args.par_glose]
        if not chosen:
            print(f"[retarget] {word} : absent du manifeste, ignoré")
        for r in chosen:
            views = [v for v in args.vues.split(",") if v in ("face", "profil")]
            out = retarget(r, word, args.aspect, views)
            print(f"[retarget] {word} {r['instance_id']} -> {out.relative_to(ROOT)}.{{glb,bvh,mp4}}")


if __name__ == "__main__":
    main()
