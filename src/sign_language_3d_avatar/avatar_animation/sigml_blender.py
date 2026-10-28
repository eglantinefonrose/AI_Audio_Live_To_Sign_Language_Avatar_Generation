"""Animation Blender à partir de SiGML (HamNoSys 4 en XML) — prototype.

Interprète un SOUS-ENSEMBLE des balises `<hns_sign>` (cf. sigmlhns.dtd de l'UEA) :
configuration de la main, direction des doigts, orientation de la paume,
emplacement, contact, mouvements droits / en arc / circulaires, répétition,
changement de configuration (`hamreplace`), symétrie à deux mains. Le non-manuel
(visage) n'est pas animé. Ce n'est PAS JASigning : les conventions d'orientation
et les distances sont celles de ce script (voir sign_language_3d_avatar/avatar_animation/README.md).

Sorties (sign_language_3d_avatar/avatar_animation/data/sigml/) : un .mp4 par signe, `_enchainement.mp4` (tous les
signes à la suite) et `_enchainement.glb`.

    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \\
        --python sign_language_3d_avatar/avatar_animation/sigml_blender.py -- sign_language_3d_avatar/dataset_generation/sigml/lexique_brouillon.sigml [--gloses MERCI,TOI]
"""
from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, replace
from pathlib import Path

import bpy
from mathutils import Matrix, Quaternion, Vector

BASE = Path(__file__).resolve().parent.parent   # sign_language_3d_avatar/
ROOT = BASE.parent                               # racine du projet
OUT = BASE / "avatar_animation" / "data" / "sigml"
FPS = 25

# Repère Blender : l'avatar regarde vers -Y ; sa gauche = +X ; haut = +Z.
AXES = {"u": Vector((0, 0, 1)), "d": Vector((0, 0, -1)), "l": Vector((1, 0, 0)),
        "r": Vector((-1, 0, 0)), "o": Vector((0, -1, 0)), "i": Vector((0, 1, 0))}
SIDE = {"R": -1.0, "L": 1.0}   # côté X de chaque main

# Emplacements (point de contact, en m, bassin à l'origine). `x` = décalé côté main.
SITES = {"forehead": (0, -0.10, 0.72), "nose": (0, -0.115, 0.66), "lips": (0, -0.10, 0.615),
         "chin": (0, -0.085, 0.585), "underchin": (0, -0.05, 0.56), "head": (0, 0, 0.78),
         "headtop": (0, 0, 0.78), "cheek": ("x", 0.07, -0.08, 0.64), "ear": ("x", 0.105, 0, 0.67),
         "neck": (0, -0.05, 0.52), "shoulders": ("x", 0.18, -0.04, 0.44),
         "chest": ("x", 0.05, -0.11, 0.36), "stomach": ("x", 0.05, -0.12, 0.18),
         "belowstomach": ("x", 0.05, -0.12, 0.08), "neutralspace": ("x", 0.10, -0.30, 0.33)}

# Configurations : flexion (MCP, PIP, DIP) en degrés par doigt, écartement, pouce.
EXT, FOLD = (0, 0, 0), (85, 100, 60)
HANDSHAPES = {
    "hamflathand": dict(fingers=[EXT] * 4, thumb="along"),
    "hamfist": dict(fingers=[FOLD] * 4, thumb="across"),
    "hamfinger2": dict(fingers=[EXT, FOLD, FOLD, FOLD], thumb="across"),
    "hamfinger23": dict(fingers=[EXT, EXT, FOLD, FOLD], thumb="across"),
    "hamfinger23spread": dict(fingers=[EXT, EXT, FOLD, FOLD], thumb="across", spread=12),
    "hamfinger2345": dict(fingers=[EXT] * 4, thumb="across"),
    "hampinch12": dict(fingers=[(40, 45, 20), EXT, EXT, EXT], thumb="oppose"),
    "hampinchall": dict(fingers=[(30, 25, 10)] * 4, thumb="oppose"),
    "hamcee12": dict(fingers=[(25, 35, 20), FOLD, FOLD, FOLD], thumb="open"),
    "hamceeall": dict(fingers=[(25, 35, 20)] * 4, thumb="open"),
}
REACH = {"hamfist": 0.07, "hampinchall": 0.15, "hamcee12": 0.13, "hamceeall": 0.13}  # poignet -> point de contact


@dataclass
class HandState:
    loc: Vector                     # position du poignet
    fingers_dir: Vector             # direction des doigts tendus
    palm: Vector                    # normale de la paume
    shape: str = "hamflathand"
    thumb_out: bool = False
    bend: str = ""                  # "", "bent", "hooked"


@dataclass
class Key:
    frame: int
    hands: dict[str, HandState] = field(default_factory=dict)


# ------------------------------------------------------------------ SiGML

def direction(token: str, prefix: str) -> Vector:
    v = Vector()
    for c in token[len(prefix):]:
        v += AXES[c]
    return v.normalized()


def palm_vector(token: str, fingers: Vector) -> Vector:
    """Paume relative aux doigts : doigts verticaux -> u = vers l'avant, d = vers soi."""
    letters = token[len("hampalm"):]
    if abs(fingers.z) > 0.8:
        letters = letters.replace("u", "o").replace("d", "i")
    p = Vector()
    for c in letters:
        p += AXES[c]
    p -= p.dot(fingers) * fingers
    if p.length < 1e-3:   # paume parallèle aux doigts : on prend une perpendiculaire
        p = fingers.cross(Vector((1, 0, 0))) if abs(fingers.x) < 0.9 else fingers.cross(Vector((0, 0, 1)))
    return p.normalized()


def site(name: str, hand: str) -> Vector:
    s = SITES[name]
    if s[0] == "x":
        return Vector((SIDE[hand] * s[1], s[2], s[3]))
    return Vector(s)


def hand_matrix(fingers: Vector, palm: Vector) -> Matrix:
    y, z = fingers.normalized(), palm.normalized()
    x = y.cross(z).normalized()
    return Matrix((x, y, z)).transposed()


def wrist_for(contact: Vector, st: HandState, neutral: bool = False) -> Vector:
    """Poignet tel que la partie de la main la plus proche de l'emplacement y soit.

    Doigts vers l'avant (loin du corps) : c'est la base de la main qui est à
    l'emplacement ; dans l'espace neutre, le milieu de la main ; sinon le bout des doigts.
    """
    reach = REACH.get(st.shape, 0.16)
    if neutral:
        reach *= 0.5
    elif st.fingers_dir.y < -0.5:
        reach = 0.02
    return contact - hand_matrix(st.fingers_dir, st.palm) @ Vector((0, reach, 0))


def mirror(st: HandState, parallel: bool) -> HandState:
    m = lambda v: Vector((-v.x, v.y, v.z))
    return replace(st, loc=m(st.loc), fingers_dir=st.fingers_dir if parallel else m(st.fingers_dir),
                   palm=st.palm if parallel else m(st.palm))


def parse_sign(manual: ET.Element, start: int, rest: dict[str, HandState]) -> tuple[list[Key], int]:
    """Clés d'animation d'un signe (main droite dominante)."""
    tokens = [c.tag for c in manual]
    symm = next((t for t in tokens if t in ("hamsymmlr", "hamsymmpar")), None)
    st = HandState(loc=Vector(), fingers_dir=Vector((0, 0, 1)), palm=Vector((0, -1, 0)))
    loc_name, contact = "neutralspace", None
    moves: list[dict] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if t in HANDSHAPES:
            target = moves[-1].setdefault("end", {}) if moves and moves[-1].get("replace") else None
            (target.__setitem__("shape", t) if target is not None else setattr(st, "shape", t))
        elif t in ("hamthumboutmod", "hamthumbopenmod"):
            st.thumb_out = True
        elif t in ("hamfingerbendmod", "hamdoublebent"):
            st.bend = "bent"
        elif t in ("hamfingerhookmod", "hamdoublehooked"):
            st.bend = "hooked"
        elif t.startswith("hamextfinger"):
            d = direction(t, "hamextfinger")
            if moves and moves[-1].get("replace"):
                moves[-1].setdefault("end", {})["fingers"] = d
            else:
                st.fingers_dir = d
        elif t.startswith("hampalm") and t != "hampalm":
            if moves and moves[-1].get("replace"):
                moves[-1].setdefault("end", {})["palm"] = t
            else:
                st.palm = palm_vector(t, st.fingers_dir)
        elif t[3:] in SITES:
            if moves:
                moves[-1]["to_site"] = t[3:]
            else:
                loc_name = t[3:]
        elif t in ("hamtouch", "hamclose") and not moves:
            contact = t
        elif t.startswith("hammove") and t[7:] and set(t[7:]) <= set(AXES):
            moves.append({"kind": "line", "dir": direction(t, "hammove"), "size": 0.15, "frames": 10})
        elif t.startswith("hamcircle") and t[9:] and set(t[9:]) <= set(AXES):
            moves.append({"kind": "circle", "axis": direction(t, "hamcircle"), "size": 0.05, "frames": 18})
        elif t.startswith("hamarc") and moves:
            moves[-1]["arc"] = AXES[t[-1]]
        elif t == "hamsmallmod" and moves:
            moves[-1]["size"] *= 0.45
        elif t == "hamlargemod" and moves:
            moves[-1]["size"] *= 1.8
        elif t == "hamfast" and moves:
            moves[-1]["frames"] = int(moves[-1]["frames"] * 0.6)
        elif t == "hamslow" and moves:
            moves[-1]["frames"] = int(moves[-1]["frames"] * 1.6)
        elif t == "hamreplace":
            if not moves:  # changement sur place
                moves.append({"kind": "line", "dir": Vector(), "size": 0.0, "frames": 8})
            moves[-1]["replace"] = True
        elif t.startswith("hamrepeat") and moves:
            moves[-1]["repeat"] = 2 if "several" not in t else 3
        # hamparbegin/hamparend/hamseqbegin/... : la structure est aplatie.
        i += 1

    point = site(loc_name, "R")
    neutral = loc_name == "neutralspace"
    if neutral and symm and contact == "hamtouch":
        point.x, neutral = 0.0, False      # deux mains qui se touchent au centre
    elif contact == "hamclose" or (contact is None and not neutral):
        point += Vector((0, -0.03 if contact else -0.08, 0))
    st.loc = wrist_for(point, st, neutral)

    def both(s: HandState) -> dict[str, HandState]:
        hands = {"R": s, "L": rest["L"]}
        if symm:
            hands["L"] = mirror(s, symm == "hamsymmpar")
        return hands

    frame = start + 10
    keys = [Key(frame, both(st)), Key(frame + 3, both(st))]
    frame += 3
    cur = st
    for mv in moves:
        for rep in range(mv.get("repeat", 1)):
            if rep:  # répétition depuis le début : retour rapide au départ
                frame += 5
                keys.append(Key(frame, both(origin)))
            origin = cur
            end = replace(cur)
            if mv.get("replace") and "end" in mv:
                e = mv["end"]
                end.shape = e.get("shape", end.shape)
                end.fingers_dir = e.get("fingers", end.fingers_dir)
                end.palm = palm_vector(e["palm"], end.fingers_dir) if "palm" in e else \
                    (end.palm - end.palm.dot(end.fingers_dir) * end.fingers_dir).normalized()
            if mv["kind"] == "line":
                delta = mv["dir"] * mv["size"]
                if "to_site" in mv:
                    delta = wrist_for(site(mv["to_site"], "R"), end) - cur.loc
                end.loc = cur.loc + delta
                if "arc" in mv:
                    mid = replace(cur, loc=cur.loc + delta / 2 + mv["arc"] * mv["size"] * 0.3)
                    keys.append(Key(frame + mv["frames"] // 2, both(mid)))
                frame += mv["frames"]
                keys.append(Key(frame, both(end)))
            else:  # cercle : 8 points dans le plan perpendiculaire à l'axe
                axis = mv["axis"]
                u = axis.cross(Vector((0, 0, 1)) if abs(axis.z) < 0.9 else Vector((1, 0, 0))).normalized()
                v = axis.cross(u)
                center = cur.loc - u * mv["size"]
                for k in range(1, 9):
                    a = 2 * math.pi * k / 8
                    p = center + (u * math.cos(a) + v * math.sin(a)) * mv["size"]
                    keys.append(Key(frame + mv["frames"] * k // 8, both(replace(end, loc=p))))
                frame += mv["frames"]
            cur = end
    frame += 5
    keys.append(Key(frame, both(cur)))
    return keys, frame


# --------------------------------------------------------------- avatar

def reset() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.fps = FPS


FINGER_NAMES = ("index", "middle", "ring", "pinky")
FINGER_LEN = {"index": (0.040, 0.025, 0.020), "middle": (0.045, 0.028, 0.022),
              "ring": (0.042, 0.026, 0.020), "pinky": (0.032, 0.020, 0.018), "thumb": (0.040, 0.032, 0.025)}
FINGER_X = {"index": 0.03, "middle": 0.01, "ring": -0.01, "pinky": -0.03}


def build_avatar():
    arm_data = bpy.data.armatures.new("signer")
    arm = bpy.data.objects.new("signer", arm_data)
    bpy.context.scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm_data.edit_bones

    def bone(name, head, tail, parent=None, roll_to=None):
        b = eb.new(name)
        b.head, b.tail = Vector(head), Vector(tail)
        if roll_to is not None:
            b.align_roll(Vector(roll_to))
        if parent:
            b.parent = eb[parent]
        return b

    bone("spine", (0, 0, 0), (0, 0, 0.45))
    bone("neck", (0, 0, 0.45), (0, 0, 0.56), "spine")
    bone("head", (0, 0, 0.56), (0, 0, 0.78), "neck")
    for h, sx in SIDE.items():
        sh, el, wr = (sx * 0.18, 0, 0.44), (sx * 0.19, 0.02, 0.15), (sx * 0.19, -0.03, -0.11)
        bone(f"clavicle.{h}", (0, 0, 0.43), sh, "spine")
        bone(f"upper_arm.{h}", sh, el, f"clavicle.{h}")
        bone(f"forearm.{h}", el, wr, f"upper_arm.{h}")
        palm_in = (-sx, 0, 0)   # au repos, paume tournée vers le corps
        hand = bone(f"hand.{h}", wr, (wr[0], wr[1], wr[2] - 0.09), f"forearm.{h}", palm_in)
        m = hand.matrix.copy()
        thumb_side = 1.0 if h == "R" else -1.0     # pouce du côté +X local (main droite)
        rot = m.to_3x3()
        for f in FINGER_NAMES:
            head = m @ Vector((thumb_side * FINGER_X[f], 0.09, 0))
            parent = f"hand.{h}"
            for k, length in enumerate(FINGER_LEN[f]):
                tail = head + rot @ Vector((0, length, 0))
                bone(f"{f}.{k + 1}.{h}", head, tail, parent, rot @ Vector((0, 0, 1)))
                parent, head = f"{f}.{k + 1}.{h}", tail
        head = m @ Vector((thumb_side * 0.025, 0.015, 0.0))
        tdir = rot @ Vector((thumb_side * 0.6, 0.8, 0)).normalized()
        parent = f"hand.{h}"
        for k, length in enumerate(FINGER_LEN["thumb"]):
            tail = head + tdir * length
            bone(f"thumb.{k + 1}.{h}", head, tail, parent, rot @ Vector((0, 0, 1)))
            parent, head = f"thumb.{k + 1}.{h}", tail
    bpy.ops.object.mode_set(mode="POSE")

    targets = {}
    for h, sx in SIDE.items():
        t = bpy.data.objects.new(f"wrist.{h}", None)
        t.rotation_mode = "QUATERNION"
        pole = bpy.data.objects.new(f"pole.{h}", None)
        pole.location = (sx * 0.45, 0.35, 0.0)
        for o in (t, pole):
            bpy.context.scene.collection.objects.link(o)
        ik = arm.pose.bones[f"forearm.{h}"].constraints.new("IK")
        ik.target, ik.pole_target, ik.chain_count = t, pole, 2
        ik.pole_angle = math.radians(90)
        cr = arm.pose.bones[f"hand.{h}"].constraints.new("COPY_ROTATION")
        cr.target = t
        targets[h] = t
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
    bpy.ops.object.mode_set(mode="OBJECT")
    return arm, targets


def add_body(arm) -> None:
    skin = bpy.data.materials.new("skin")
    skin.diffuse_color = (0.86, 0.68, 0.55, 1)
    shirt = bpy.data.materials.new("shirt")
    shirt.diffuse_color = (0.18, 0.32, 0.55, 1)
    dark = bpy.data.materials.new("dark")
    dark.diffuse_color = (0.08, 0.08, 0.1, 1)

    # Attacher les maillages en position de repos (sinon l'IK décale tout).
    arm.data.pose_position = "REST"
    bpy.context.view_layer.update()

    def attach(obj, bone_name, world, mat):
        obj.data.materials.append(mat)
        obj.parent, obj.parent_type, obj.parent_bone = arm, "BONE", bone_name
        bpy.context.view_layer.update()
        obj.matrix_world = world

    for b in arm.data.bones:
        rest = arm.matrix_world @ b.matrix_local
        mid = rest @ Matrix.Translation((0, b.length / 2, 0)) @ Matrix.Rotation(math.radians(90), 4, "X")
        if b.name == "head":
            bpy.ops.mesh.primitive_uv_sphere_add(radius=0.105, segments=32, ring_count=16)
            attach(bpy.context.active_object, "head", Matrix.Translation((0, 0, 0.665)), skin)
            for x in (-0.035, 0.035):
                bpy.ops.mesh.primitive_uv_sphere_add(radius=0.012)
                attach(bpy.context.active_object, "head", Matrix.Translation((x, -0.095, 0.69)), dark)
            bpy.ops.mesh.primitive_cone_add(radius1=0.015, depth=0.03)
            attach(bpy.context.active_object, "head",
                   Matrix.Translation((0, -0.115, 0.655)) @ Matrix.Rotation(math.radians(90), 4, "X"), skin)
            continue
        if b.name == "spine":
            bpy.ops.mesh.primitive_cylinder_add(radius=0.13, depth=b.length + 0.05, vertices=24)
            obj = bpy.context.active_object
            obj.scale = (1, 0.6, 1)
            attach(obj, b.name, Matrix.Translation((0, 0, 0.22)) @ Matrix.Diagonal((1, 0.6, 1, 1)), shirt)
            continue
        finger = b.name.split(".")[0] in (*FINGER_NAMES, "thumb")
        radius = 0.0085 if finger else 0.035 if "arm" in b.name or "clavicle" in b.name else 0.03
        if b.name.startswith("hand."):
            bpy.ops.mesh.primitive_cube_add(size=1)
            obj = bpy.context.active_object
            world = rest @ Matrix.Translation((0, 0.05, 0)) @ Matrix.Diagonal((0.085, 0.10, 0.025, 1))
            attach(obj, b.name, world, skin)
            continue
        bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=b.length + (0.012 if finger else 0.04),
                                            vertices=12)
        attach(bpy.context.active_object, b.name, mid, skin if finger or "forearm" in b.name else shirt)
    arm.data.pose_position = "POSE"


def rest_state(h: str) -> HandState:
    sx = SIDE[h]
    return HandState(loc=Vector((sx * 0.19, -0.03, -0.11)), fingers_dir=Vector((0, 0, -1)),
                     palm=Vector((-sx, 0, 0)))


def finger_rotations(st: HandState, h: str) -> dict[str, Quaternion]:
    spec = HANDSHAPES.get(st.shape, HANDSHAPES["hamflathand"])
    out = {}
    spread = math.radians(spec.get("spread", 0))
    for f, curl in zip(FINGER_NAMES, spec["fingers"]):
        curl = list(curl)
        if curl == list(EXT) and st.bend == "bent":
            curl = [80, 0, 0]
        elif curl == list(EXT) and st.bend == "hooked":
            curl = [0, 80, 70]
        side = {"index": 1, "middle": 0.3, "ring": -0.3, "pinky": -1}[f] * (1 if h == "R" else -1)
        for k, a in enumerate(curl):
            q = Quaternion((1, 0, 0), math.radians(a))
            if k == 0 and spread:
                q = Quaternion((0, 0, 1), side * spread) @ q
            out[f"{f}.{k + 1}.{h}"] = q
    ts = 1 if h == "R" else -1
    thumb = "open" if st.thumb_out else spec["thumb"]
    # (abduction autour de la normale de paume, flexion des 3 phalanges)
    abd, flex = {"along": (-25, (5, 5, 5)), "across": (-55, (35, 30, 25)),
                 "oppose": (-40, (40, 25, 15)), "open": (25, (10, 10, 5))}[thumb]
    for k, a in enumerate(flex):
        q = Quaternion((1, 0, 0), math.radians(a))
        if k == 0:
            q = Quaternion((0, 0, 1), math.radians(-ts * abd)) @ q
        out[f"thumb.{k + 1}.{h}"] = q
    return out


def apply_keys(arm, targets, keys: list[Key]) -> None:
    prev = {h: None for h in SIDE}
    for key in keys:
        for h, st in key.hands.items():
            t = targets[h]
            t.location = st.loc
            q = hand_matrix(st.fingers_dir, st.palm).to_quaternion()
            if prev[h] is not None:
                q.make_compatible(prev[h])
            prev[h] = q
            t.rotation_quaternion = q
            t.keyframe_insert("location", frame=key.frame)
            t.keyframe_insert("rotation_quaternion", frame=key.frame)
            for name, rot in finger_rotations(st, h).items():
                pb = arm.pose.bones[name]
                pb.rotation_quaternion = rot
                pb.keyframe_insert("rotation_quaternion", frame=key.frame)


def add_label(text: str, start: int, end: int) -> None:
    curve = bpy.data.curves.new(f"label_{text}", "FONT")
    curve.body, curve.size, curve.align_x = text, 0.07, "CENTER"
    obj = bpy.data.objects.new(f"label_{text}", curve)
    obj.location, obj.rotation_euler = (0, -0.3, -0.24), (math.radians(90), 0, 0)
    mat = bpy.data.materials.new("label")
    mat.diffuse_color = (0.1, 0.1, 0.1, 1)
    curve.materials.append(mat)
    bpy.context.scene.collection.objects.link(obj)
    for f, hide in ((1, True), (start, False), (end, True)):
        if f < 1:
            continue
        obj.hide_render = obj.hide_viewport = hide
        obj.keyframe_insert("hide_render", frame=f)
        obj.keyframe_insert("hide_viewport", frame=f)


def setup_render(end: int, path: Path) -> None:
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, end
    try:
        scene.render.engine = "BLENDER_WORKBENCH"
        scene.display.shading.color_type = "MATERIAL"
        scene.display.shading.light = "STUDIO"
    except (TypeError, AttributeError):
        scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x, scene.render.resolution_y = 540, 600
    world = bpy.data.worlds.new("w")
    world.color = (0.94, 0.94, 0.94)
    scene.world = world
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    cam.data.lens = 50
    # Légèrement de trois quarts pour voir la profondeur.
    cam.location = (-0.75, -2.3, 0.45)
    cam.rotation_euler = (math.radians(88), 0, math.radians(-18))
    scene.collection.objects.link(cam)
    scene.camera = cam
    s = scene.render.image_settings
    if hasattr(s, "media_type"):
        s.media_type = "VIDEO"
    s.file_format = "FFMPEG"
    scene.render.ffmpeg.format, scene.render.ffmpeg.codec = "MPEG4", "H264"
    scene.render.filepath = str(path)


def animate(signs: list[tuple[str, ET.Element]], path: Path, export_glb: bool = False) -> None:
    reset()
    arm, targets = build_avatar()
    add_body(arm)
    rest = {h: rest_state(h) for h in SIDE}
    keys = [Key(1, dict(rest))]
    frame = 1
    for gloss, manual in signs:
        sign_keys, end = parse_sign(manual, frame, rest)
        keys += sign_keys
        add_label(gloss, sign_keys[0].frame - 6, end)
        frame = end
    keys.append(Key(frame + 12, dict(rest)))
    apply_keys(arm, targets, keys)
    end = frame + 18
    path.parent.mkdir(parents=True, exist_ok=True)
    if export_glb:
        bpy.context.scene.frame_start, bpy.context.scene.frame_end = 1, end
        bpy.ops.export_scene.gltf(filepath=str(path.with_suffix(".glb")), export_format="GLB",
                                  export_animations=True, export_force_sampling=True)
    setup_render(end, path)
    bpy.ops.render.render(animation=True)


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(prog="sigml_blender.py")
    ap.add_argument("sigml", type=Path)
    ap.add_argument("--gloses", help="sous-ensemble, séparé par des virgules (défaut : toutes)")
    ap.add_argument("--sans-enchainement", action="store_true")
    args = ap.parse_args(argv)

    root = ET.parse(args.sigml).getroot()
    signs = [(s.get("gloss"), s.find("hamnosys_manual")) for s in root.iter("hns_sign")]
    if args.gloses:
        wanted = args.gloses.split(",")
        signs = [s for s in signs if s[0] in wanted]
    for gloss, manual in signs:
        animate([(gloss, manual)], OUT / f"{gloss}.mp4")
        print(f"[sigml] {gloss} -> {(OUT / gloss).relative_to(ROOT)}.mp4")
    if len(signs) > 1 and not args.sans_enchainement:
        animate(signs, OUT / "_enchainement.mp4", export_glb=True)
        print(f"[sigml] enchaînement -> {(OUT / '_enchainement').relative_to(ROOT)}.{{mp4,glb}}")


if __name__ == "__main__":
    main()
