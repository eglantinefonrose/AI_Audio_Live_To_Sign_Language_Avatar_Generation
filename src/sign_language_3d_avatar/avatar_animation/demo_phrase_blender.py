"""Démo : l'avatar signe une phrase (suite de gloses) et la rend en vidéo dans Blender.

La phrase est assemblée par `sequence.assemble` (repos, transitions, signes LSFB nettoyés,
composés et mots épelés joués signe par signe ou lettre par lettre, pointages générés,
mots sans signe affichés), transférée sur l'avatar, puis rendue en
.mp4 (Workbench, vue de trois-quarts, 25 images/s) avec la glose en cours en sous-titre
(`--sans-sous-titres` : seuls les mots sans signe restent écrits).

    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \\
        --python src/sign_language_3d_avatar/avatar_animation/demo_phrase_blender.py -- \\
        --gloses "MOI MAISON AIDER-À-TOI MERCI" [--avatar avatar.vrm [--sans-manches]] [--sortie demo.mp4] [--gros-plan] [--sans-sous-titres]

Le plus simple est de passer par `src/sign_language_3d_avatar/tester_phrase_lsfb.py`.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import avatar_rig  # noqa: E402
from sequence import DICT, assemble, load_index  # noqa: E402

def add_subtitles(timeline, missing_only: bool = False) -> None:
    """Glose en cours incrustée en haut à gauche de chaque image (tampon de rendu Blender,
    toujours au premier plan), mise à jour à chaque image par un gestionnaire d'événement.
    `missing_only` : seulement les mots sans signe (sinon, rien ne les montrerait)."""
    text = {}
    for start, end, gloss, kind, _, sign in timeline:
        if missing_only and kind != "absent":
            continue
        label = {"absent": f"« {gloss.lower()} » — pas de signe",
                 "lettre": f"{gloss} — {sign[3:] if sign else ''}"}.get(kind, gloss)
        for f in range(start + 1, end + 1):     # image i des articulations = image i + 1
            text[f] = label
    r = bpy.context.scene.render
    r.use_stamp = True
    for field in ("date", "time", "render_time", "frame", "frame_range", "memory", "hostname",
                  "camera", "lens", "scene", "marker", "filename", "sequencer_strip"):
        setattr(r, f"use_stamp_{field}", False)
    r.use_stamp_note, r.stamp_font_size = True, 44
    r.stamp_foreground, r.stamp_background = (1, 1, 1, 1), (0.1, 0.1, 0.12, 0.85)

    def update(scene, *_):                      # rien à écrire : pas de bandeau vide
        scene.render.stamp_note_text = text.get(scene.frame_current, " ")
        scene.render.use_stamp = scene.frame_current in text

    bpy.app.handlers.frame_change_pre.append(update)


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(prog="demo_phrase_blender.py")
    ap.add_argument("--gloses", required=True, help="gloses séparées par des espaces")
    ap.add_argument("--avatar", help="fichier .vrm (VRoid Studio) ; défaut : mannequin")
    ap.add_argument("--sortie", default=str(DICT / "demo.mp4"))
    ap.add_argument("--angle", type=float, default=25.0, help="angle de vue (0 = de face)")
    ap.add_argument("--gros-plan", action="store_true", help="caméra qui suit la main droite")
    ap.add_argument("--sans-manches", action="store_true",
                    help="retirer les manches du haut (avatar dont les bras traversent les manches)")
    ap.add_argument("--sans-sous-titres", action="store_true",
                    help="ne pas écrire le mot signé (les mots sans signe restent écrits)")
    args = ap.parse_args(argv)
    render(args.gloses.split(), Path(args.sortie), args.avatar, args.angle, gros_plan=args.gros_plan,
           sleeveless=args.sans_manches, subtitles=not args.sans_sous_titres)


def render(glosses: list[str], sortie: Path, avatar: str | None = None, angle: float = 25.0,
           index: dict | None = None, gros_plan: bool = False, sleeveless: bool = False,
           subtitles: bool = True) -> None:
    """Rend la phrase `glosses` en .mp4 (`sortie`) ; scène remise à zéro à chaque appel.
    `gros_plan` : caméra qui suit la main droite (forme de la main lisible)."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.app.handlers.frame_change_pre.clear()          # sous-titres d'un rendu précédent
    index = index or load_index()
    joints, timeline = assemble(glosses, index)
    for start, end, gloss, kind, detail, _ in timeline:
        print(f"[démo] {start / index['fps']:5.2f} s  {gloss:16s} {detail}")

    if avatar:
        arm, mapping, version = avatar_rig.load_vrm(Path(avatar).expanduser().resolve(), sleeveless)
    else:
        arm, mapping, version = avatar_rig.load_mannequin()
    rt = avatar_rig.Retargeter(arm, mapping, version)
    rt.action(joints, "PHRASE")
    n = len(joints["hip"])

    scene, step = bpy.context.scene, 2                # 50 im/s -> 25 im/s rendues
    scene.frame_start, scene.frame_end, scene.frame_step = 1, n, step
    scene.render.fps = index["fps"] // step
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "TEXTURE" if avatar else "MATERIAL"
    scene.render.resolution_x, scene.render.resolution_y = 720, 720
    world = bpy.data.worlds.new("fond")
    world.color = (0.93, 0.93, 0.93)
    scene.world = world
    cam = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    scene.collection.objects.link(cam)
    scene.camera = cam
    avatar_rig.frame_camera(arm, cam, azimuth_deg=angle)
    if gros_plan:
        hand = arm.data.bones[mapping["rightHand"]]
        az = math.radians(angle)
        cam.data.ortho_scale = hand.length * 4
        cam.location = (5 * math.sin(az), -5 * math.cos(az), 0)
        follow = cam.constraints.new("COPY_LOCATION")     # x, z : bout de la paume ; y : recul fixe
        follow.target, follow.subtarget, follow.head_tail = arm, hand.name, 1.0
        follow.use_y, follow.use_offset = False, True
    add_subtitles(timeline, missing_only=not subtitles)

    out = Path(sortie).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    s = scene.render.image_settings
    if hasattr(s, "media_type"):
        s.media_type = "VIDEO"
    s.file_format = "FFMPEG"
    scene.render.ffmpeg.format, scene.render.ffmpeg.codec = "MPEG4", "H264"
    scene.render.filepath = str(out)
    bpy.ops.render.render(animation=True)
    print(f"[démo] vidéo : {out} ({n / index['fps']:.1f} s)")


if __name__ == "__main__":
    main()
