"""Avatar Blender (mannequin ou VRM VRoid) et transfert d'animation par directions d'os.

Le transfert ne dépend ni de la taille ni des proportions de l'avatar : pour chaque os
humanoïde, on calcule la **direction** voulue à partir des articulations 3D (`pose3d`),
puis la rotation qui y amène l'os depuis sa pose de repos. Les mains, le torse et la
tête sont orientés par deux vecteurs (direction + axe transverse), le reste par la
rotation minimale, ce qui garde le roulis hérité du parent.

Deux avatars :
- `load_mannequin()` : squelette humanoïde aux proportions standard (pose en A) et
  maillage simple, sans fichier externe ;
- `load_vrm(chemin)` : modèle VRoid Studio (`.vrm`, VRM 0.x ou 1.0), importé par
  l'importeur glTF de Blender (un .vrm est un glTF). La table des os humanoïdes est lue
  dans l'extension VRM du fichier ; aucune extension Blender n'est nécessaire (les
  matériaux MToon sont simplifiés, suffisant pour des démos).

Noms d'os humanoïdes = ceux de la norme VRM (`leftUpperArm`, `rightIndexProximal`…).
Côté signeur : « L » = gauche du signeur = `left` de l'avatar.
"""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

UP, FORWARD = Vector((0, 0, 1)), Vector((0, -1, 0))   # l'avatar regarde vers −Y
SIDES = {"left": "L", "right": "R"}
FINGER_JOINTS = {"Thumb": (1, 2, 3, 4), "Index": (5, 6, 7, 8), "Middle": (9, 10, 11, 12),
                 "Ring": (13, 14, 15, 16), "Little": (17, 18, 19, 20)}
FINGER_SEGMENTS = ("Proximal", "Intermediate", "Distal")
THUMB_SEGMENTS = {0: ("Proximal", "Intermediate", "Distal"),        # VRM 0.x
                  1: ("Metacarpal", "Proximal", "Distal")}          # VRM 1.0 (et mannequin)


# ------------------------------------------------------------- correspondances
def drive_specs(thumb_names=THUMB_SEGMENTS[1]) -> dict[str, tuple[str, str, str | None]]:
    """Os humanoïde -> (articulation de départ, articulation visée, vecteur secondaire)."""
    specs = {"spine": ("hip", "chest", "across"), "chest": ("hip", "chest", "across"),
             "upperChest": ("hip", "chest", "across"), "neck": ("chest", "head", "across"),
             "head": ("chest", "head", "forward")}
    for side, s in SIDES.items():
        specs[f"{side}Shoulder"] = ("chest", f"shoulder.{s}", None)
        specs[f"{side}UpperArm"] = (f"shoulder.{s}", f"elbow.{s}", None)
        specs[f"{side}LowerArm"] = (f"elbow.{s}", f"wrist.{s}", None)
        specs[f"{side}Hand"] = (f"wrist.{s}", f"hand.{s}.9", f"palm.{s}")
        for finger, chain in FINGER_JOINTS.items():
            names = thumb_names if finger == "Thumb" else FINGER_SEGMENTS
            for seg, a, b in zip(names, chain, chain[1:]):
                specs[f"{side}{finger}{seg}"] = (f"hand.{s}.{a}", f"hand.{s}.{b}", None)
    return specs


def secondary(kind: str, j: dict[str, np.ndarray], f: int) -> Vector:
    if kind == "across":
        return Vector(j["shoulder.L"][f] - j["shoulder.R"][f])
    if kind == "forward":
        return Vector(j["nose"][f] - j["head"][f])
    s = kind[-1]                                   # palm.L / palm.R : index -> auriculaire
    return Vector(j[f"hand.{s}.5"][f] - j[f"hand.{s}.17"][f])


# -------------------------------------------------------------------- mannequin
def _mannequin_rest() -> dict[str, tuple[Vector, Vector, str | None]]:
    """Os -> (tête, queue, parent) en mètres, pose en A, regard vers −Y, gauche en +X."""
    b = {"hips": ((0, 0, 0), (0, 0, 0.10), None), "spine": ((0, 0, 0.10), (0, 0, 0.30), "hips"),
         "chest": ((0, 0, 0.30), (0, 0, 0.47), "spine"), "neck": ((0, 0, 0.47), (0, 0, 0.57), "chest"),
         "head": ((0, 0, 0.57), (0, 0, 0.80), "neck")}
    for side, sx in (("left", 1), ("right", -1)):
        sh, a = Vector((0.19 * sx, 0, 0.45)), math.radians(30)
        down = Vector((math.sin(a) * sx, 0.04, -math.cos(a))).normalized()
        el, wr = sh + down * 0.29, sh + down * 0.55
        b[f"{side}Shoulder"] = ((0.03 * sx, 0, 0.44), tuple(sh), "chest")
        b[f"{side}UpperArm"] = (tuple(sh), tuple(el), f"{side}Shoulder")
        b[f"{side}LowerArm"] = (tuple(el), tuple(wr), f"{side}UpperArm")
        b[f"{side}Hand"] = (tuple(wr), tuple(wr + down * 0.08), f"{side}LowerArm")
        # Paume tournée vers le corps ; index devant (−Y), auriculaire derrière (+Y).
        for finger, dy, lengths in (("Index", -0.028, (0.045, 0.027, 0.022)),
                                    ("Middle", -0.009, (0.050, 0.030, 0.024)),
                                    ("Ring", 0.010, (0.046, 0.028, 0.022)),
                                    ("Little", 0.027, (0.036, 0.021, 0.019))):
            p, parent = wr + down * 0.085 + Vector((0, dy, 0)), f"{side}Hand"
            for seg, ln in zip(FINGER_SEGMENTS, lengths):
                q = p + down * ln
                b[f"{side}{finger}{seg}"] = (tuple(p), tuple(q), parent)
                p, parent = q, f"{side}{finger}{seg}"
        tdir = (down * 0.6 + Vector((0, -0.75, 0)) + Vector((-0.25 * sx, 0, 0))).normalized()
        p, parent = wr + down * 0.02 + Vector((-0.012 * sx, -0.02, 0)), f"{side}Hand"
        for seg, ln in zip(THUMB_SEGMENTS[1], (0.040, 0.032, 0.025)):
            q = p + tdir * ln
            b[f"{side}Thumb{seg}"] = (tuple(p), tuple(q), parent)
            p, parent = q, f"{side}Thumb{seg}"
    return {k: (Vector(h), Vector(t), par) for k, (h, t, par) in b.items()}


def load_mannequin() -> tuple[bpy.types.Object, dict[str, str], int]:
    """Crée le mannequin ; renvoie (armature, os humanoïde -> nom d'os, version VRM des pouces)."""
    rest = _mannequin_rest()
    arm = bpy.data.objects.new("mannequin", bpy.data.armatures.new("mannequin"))
    bpy.context.scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    for name, (head, tail, _) in rest.items():
        eb = arm.data.edit_bones.new(name)
        eb.head, eb.tail = head, tail
    for name, (_, _, parent) in rest.items():
        if parent:
            arm.data.edit_bones[name].parent = arm.data.edit_bones[parent]
    bpy.ops.object.mode_set(mode="OBJECT")
    _mannequin_body(arm)
    return arm, {n: n for n in rest}, 1


def _mannequin_body(arm) -> None:
    """Maillage minimal : un cylindre par os, une sphère pour la tête, attachés aux os."""
    skin = bpy.data.materials.new("peau")
    skin.diffuse_color = (0.85, 0.66, 0.53, 1)
    for bone in arm.data.bones:
        n = bone.name
        if n == "head":
            bpy.ops.mesh.primitive_uv_sphere_add(radius=0.11, segments=24, ring_count=12)
        else:
            r = 0.008 if any(f in n for f in FINGER_JOINTS) else 0.03
            r = {"hips": 0.07, "spine": 0.075, "chest": 0.09, "neck": 0.035}.get(n, r)
            r = 0.022 if n.endswith("Hand") else r
            bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=bone.length, vertices=12)
        obj = bpy.context.active_object
        obj.name = f"M_{n}"
        obj.data.materials.append(skin)
        rest = arm.matrix_world @ bone.matrix_local
        offset = Vector((0, 0.11 * 0.3, 0)) if n == "head" else Vector((0, bone.length / 2, 0))
        world = rest @ Matrix.Translation(offset) @ Matrix.Rotation(math.radians(90), 4, "X")
        obj.parent, obj.parent_type, obj.parent_bone = arm, "BONE", n
        bpy.context.view_layer.update()
        obj.matrix_world = world


# ------------------------------------------------------------------------- VRM
def _vrm_json(path: Path) -> dict:
    """Chunk JSON d'un fichier .vrm (glTF binaire)."""
    data = path.read_bytes()
    if data[:4] != b"glTF":
        raise ValueError(f"{path} n'est pas un fichier VRM/glTF binaire")
    length, kind = struct.unpack_from("<I4s", data, 12)
    if kind != b"JSON":
        raise ValueError("chunk JSON introuvable")
    return json.loads(data[20:20 + length])


def vrm_humanoid(path: Path) -> tuple[dict[str, str], int]:
    """(os humanoïde -> nom du nœud, version VRM 0 ou 1)."""
    g = _vrm_json(path)
    nodes, ext = g["nodes"], g.get("extensions", {})
    if "VRMC_vrm" in ext:
        bones = ext["VRMC_vrm"]["humanoid"]["humanBones"]
        return {k: nodes[v["node"]]["name"] for k, v in bones.items()}, 1
    if "VRM" in ext:
        bones = ext["VRM"]["humanoid"]["humanBones"]
        return {b["bone"]: nodes[b["node"]]["name"] for b in bones if b.get("node", -1) >= 0}, 0
    raise ValueError(f"{path} : pas d'extension VRM (VRMC_vrm ou VRM)")


def load_vrm(path: Path) -> tuple[bpy.types.Object, dict[str, str], int]:
    mapping, version = vrm_humanoid(path)
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    new = [o for o in bpy.data.objects if o not in before]
    arms = [o for o in new if o.type == "ARMATURE"]
    if not arms:
        raise RuntimeError("aucune armature importée depuis le VRM")
    arm = max(arms, key=lambda o: len(o.data.bones))
    # Les noms de nœuds deviennent des noms d'os ; on ne garde que ceux trouvés.
    mapping = {k: v for k, v in mapping.items() if v in arm.data.bones}
    if version == 0:   # VRM 0.x regarde vers +Z glTF (= +Y Blender) : demi-tour
        roots = [o for o in new if o.parent is None]
        for o in roots:
            o.rotation_mode = "XYZ"
            o.rotation_euler.z += math.pi
        bpy.context.view_layer.update()
    return arm, mapping, version


# ------------------------------------------------------------------- transfert
class Retargeter:
    """Calcule, image par image, les rotations locales des os humanoïdes."""

    def __init__(self, arm, mapping: dict[str, str], version: int = 1):
        self.arm, self.map = arm, mapping
        specs = drive_specs(THUMB_SEGMENTS[version])
        self.specs = {mapping[h]: v for h, v in specs.items() if h in mapping}
        self.to_arm = arm.matrix_world.to_3x3().inverted()    # monde -> espace armature
        bones = arm.data.bones
        self.order = []
        stack = [b for b in bones if b.parent is None]
        while stack:                                          # parents avant enfants
            b = stack.pop(0)
            self.order.append(b.name)
            stack.extend(b.children)
        self.rest = {b.name: b.matrix_local.to_3x3().normalized() for b in bones}
        self.parent = {b.name: (b.parent.name if b.parent else None) for b in bones}
        heads = {b.name: b.head_local.copy() for b in bones}
        # Direction de repos de chaque os piloté : vers la tête de l'os humanoïde suivant.
        human = {v: k for k, v in mapping.items()}
        nxt = self._next_bone(human)
        self.rest_dir = {}
        for name in self.order:          # parents avant enfants (repli sur le parent)
            if name not in self.specs:
                continue
            target, par = nxt.get(name), self.parent[name]
            if target:
                d = heads[target] - heads[name]
            elif human.get(name) == "head":
                d = self.to_arm @ UP      # l'orientation des os terminaux importés n'est pas fiable
            elif par in self.rest_dir:   # dernière phalange : dans le prolongement de la précédente
                d = self.rest_dir[par].copy()
            else:
                d = bones[name].tail_local - heads[name]
            self.rest_dir[name] = d.normalized()
        # Vecteurs secondaires de repos (espace armature).
        g = lambda h: heads[mapping[h]] if h in mapping else None
        across = (g("leftUpperArm") - g("rightUpperArm")) if g("leftUpperArm") and g("rightUpperArm") else Vector((1, 0, 0))
        self.rest_sec = {"across": across.normalized(), "forward": (self.to_arm @ FORWARD).normalized()}
        for side, s in SIDES.items():
            a, b = g(f"{side}IndexProximal"), g(f"{side}LittleProximal")
            self.rest_sec[f"palm.{s}"] = (a - b).normalized() if a and b else Vector((0, -1, 0))

    def _next_bone(self, human: dict[str, str]) -> dict[str, str]:
        """Os piloté -> os humanoïde dont la tête sert de cible au repos."""
        m = self.map
        pairs = {"spine": "chest", "chest": "upperChest" if "upperChest" in m else "neck",
                 "upperChest": "neck", "neck": "head"}
        for side in SIDES:
            pairs |= {f"{side}Shoulder": f"{side}UpperArm", f"{side}UpperArm": f"{side}LowerArm",
                      f"{side}LowerArm": f"{side}Hand", f"{side}Hand": f"{side}MiddleProximal"}
            for finger in FINGER_JOINTS:
                segs = [s for s in ("Metacarpal",) + FINGER_SEGMENTS if f"{side}{finger}{s}" in m]
                for a, b in zip(segs, segs[1:]):
                    pairs[f"{side}{finger}{a}"] = f"{side}{finger}{b}"
        return {m[a]: m[b] for a, b in pairs.items() if a in m and b in m}

    @staticmethod
    def _frame(primary: Vector, sec: Vector) -> Matrix:
        e1 = primary.normalized()
        e2 = (sec - sec.dot(e1) * e1)
        e2 = e2.normalized() if e2.length > 1e-6 else e1.orthogonal().normalized()
        return Matrix((e1, e2, e1.cross(e2))).transposed()

    def frame(self, j: dict[str, np.ndarray], f: int) -> dict[str, Matrix]:
        """Rotations locales (matrix_basis 3×3) de tous les os pour l'image f."""
        pose_world, basis = {}, {}
        for name in self.order:
            par = self.parent[name]
            carried = (pose_world[par] @ self.rest[par].transposed() @ self.rest[name]) if par else self.rest[name]
            target = carried
            spec = self.specs.get(name)
            if spec:
                a, b, sec = spec
                d = self.to_arm @ Vector(j[b][f] - j[a][f])
                if d.length > 1e-6:
                    if sec:
                        s = self.to_arm @ secondary(sec, j, f)
                        q = self._frame(d, s) @ self._frame(self.rest_dir[name], self.rest_sec[sec]).transposed()
                        target = q @ self.rest[name]
                    else:
                        cur = carried @ self.rest[name].transposed() @ self.rest_dir[name]
                        target = cur.rotation_difference(d).to_matrix() @ carried
            pose_world[name] = target
            basis[name] = carried.transposed() @ target
        return basis

    def action(self, j: dict[str, np.ndarray], name: str, frame_start: int = 1):
        """Crée une action (quaternions, une clé par image) pour les articulations j."""
        n = len(next(iter(j.values())))
        frames = [self.frame(j, f) for f in range(n)]
        act = bpy.data.actions.new(name)
        act.use_fake_user = True
        ad = self.arm.animation_data or self.arm.animation_data_create()
        ad.action = act
        for bone in self.order:
            if bone not in self.specs:        # cheveux, vêtements… : laissés au repos
                continue
            pb = self.arm.pose.bones[bone]
            pb.rotation_mode = "QUATERNION"
            quats = [frames[f][bone].to_quaternion() for f in range(n)]
            for k in range(1, n):                             # continuité des quaternions
                if quats[k].dot(quats[k - 1]) < 0:
                    quats[k] = -quats[k]
            path = f'pose.bones["{bone}"].rotation_quaternion'
            for i in range(4):
                fc = act.fcurve_ensure_for_datablock(self.arm, path, index=i, group_name=bone)
                fc.keyframe_points.add(n)
                co = []
                for f, q in enumerate(quats):
                    co += [frame_start + f, q[i]]
                fc.keyframe_points.foreach_set("co", co)
                fc.update()
        return act


def frame_camera(arm, cam, margin: float = 1.25, azimuth_deg: float = 0.0) -> None:
    """Cadre le haut du corps de l'avatar (de la taille au-dessus de la tête). `azimuth_deg`
    tourne la caméra autour de l'avatar (20 à 30° : vue de trois-quarts, profondeur lisible)."""
    mw = arm.matrix_world
    pts = [mw @ b.head_local for b in arm.data.bones] + [mw @ b.tail_local for b in arm.data.bones]
    head = max(p.z for p in pts)
    hips = [mw @ b.head_local for b in arm.data.bones if "hip" in b.name.lower()]
    low = min(p.z for p in hips) if hips else min(p.z for p in pts)
    cx = sum(p.x for p in pts) / len(pts)
    height = (head - low) * margin
    az = math.radians(azimuth_deg)
    cam.data.type, cam.data.ortho_scale = "ORTHO", height
    cam.location = (cx + 5 * math.sin(az), -5 * math.cos(az), (head + low) / 2)
    cam.rotation_euler = (math.radians(90), 0, az)
