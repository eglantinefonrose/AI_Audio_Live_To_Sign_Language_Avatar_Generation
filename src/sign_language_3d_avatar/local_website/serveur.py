"""Outil local « Avatar LSFB » : une page web où l'on tape une phrase en gloses LSFB, qui dit
pour chaque glose ce que l'avatar en fera, puis fabrique la vidéo de l'avatar qui signe la
phrase (aperçu et téléchargement). Mode d'emploi pour l'utilisateur : `NOTICE.md`.

Le serveur n'écoute que sur cet ordinateur (127.0.0.1) :

- `GET  /`                      la page (`index.html`) ;
- `GET  /api/verifier?gloses=…` pour chaque glose : signe, composé (signes joués à la suite),
                                épelé, pointage ou absent, avec une suggestion pour un mot absent
                                (`src/lsf_translator/synonymes_lsfb.csv`, puis orthographe proche) ;
- `GET  /api/vocabulaire`       les gloses jouables (saisie assistée) ;
- `POST /api/generer`           `{"gloses": "…", "sous_titres": true}` -> `{"id": …}` : rendu lancé
                                en arrière-plan ; sans sous-titres, seuls les mots sans signe sont écrits ;
- `GET  /api/travail/<id>`      état du rendu (en cours, fini, erreur) et avancement ;
- `GET  /videos/<nom>.mp4`      la vidéo (`?telecharger=1` : en pièce jointe, nom lisible).

Le rendu est celui de `avatar_animation/demo_phrase_blender.py` (Blender en arrière-plan,
vue de trois-quarts, glose en cours en sous-titre), avec le personnage VRoid du projet
(`avatar_animation/data/avatar_sign_4.vrm`), un autre (`--avatar`, `--sans-manches` pour
retirer des manches que les bras traversent) ou le mannequin (`--mannequin`). Une phrase déjà
rendue avec le même dictionnaire et le même code de rendu n'est pas refaite. Les vidéos vont dans `videos/` (ignoré
par git : dérivées de LSFB-ISOL, CC BY-NC-SA 4.0).

Seul numpy est nécessaire (le projet complet tire torch, moshi…) :

    uv run --no-project --with numpy python src/sign_language_3d_avatar/local_website/serveur.py [--ouvrir]
        [--port 8000] [--blender chemin] [--avatar fichier.vrm [--sans-manches] | --mannequin]

Blender : `--blender`, sinon la variable BLENDER, sinon l'emplacement habituel (macOS,
Windows) ou `blender` dans le PATH.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import unicodedata
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE / "avatar_animation"))
from sequence import INDEX, assemble, load_index, resolve  # noqa: E402

DEMO = BASE / "avatar_animation" / "demo_phrase_blender.py"
AVATAR = BASE / "avatar_animation" / "data" / "avatar_sign_4.vrm"
SYNONYMS = BASE.parent / "lsf_translator" / "synonymes_lsfb.csv"
VIDEOS = HERE / "videos"
# Une vidéo déjà rendue est refaite si le dictionnaire ou le code du rendu change.
RENDER_CODE = [INDEX, DEMO] + [BASE / "avatar_animation" / f for f in ("avatar_rig.py", "sequence.py")]
PAGE = HERE / "index.html"
MAX_GLOSSES = 40                                  # phrase plus longue : refusée (rendu trop long)
FRAME = re.compile(r"append frame (\d+)", re.IGNORECASE)


# ------------------------------------------------------------------- gloses
def split(text: str) -> list[str]:
    return [g for g in text.upper().split() if g]


def playable(info: dict) -> bool:
    return info["type"] != "absent"


class Glosses:
    """Dictionnaire de l'avatar : ce qu'il fait de chaque glose, et des suggestions."""

    def __init__(self) -> None:
        self.index = load_index()
        self.vocabulary = sorted(g for g, p in self.index["gloses"].items() if playable(p))
        self.synonyms: dict[str, str] = {}
        if SYNONYMS.exists():
            with open(SYNONYMS, newline="", encoding="utf-8") as f:
                self.synonyms = {r["glose"]: r["remplacement"] for r in csv.DictReader(f) if r.get("remplacement")}
        plain = lambda g: unicodedata.normalize("NFKD", g).encode("ascii", "ignore").decode()
        self.by_plain = {plain(g): g for g in self.vocabulary}
        self.plain = plain

    def check(self, gloss: str) -> dict:
        info = resolve(gloss, self.index)
        kind = info["type"]
        out = {"glose": gloss, "type": kind}
        if kind == "signe":
            out["detail"] = "signe LSFB" + (" (forme de base)" if info.get("forme") == "base" else "")
        elif kind == "composé":
            out["detail"] = "signes joués à la suite : " + " + ".join(info["signes"])
        elif kind == "épelé":
            out["detail"] = "épelé : " + "-".join(l.split(":", 1)[1] for l in info["signes"])
        elif kind == "pointage":
            out["detail"] = "pointage"
        else:
            out["detail"] = "pas de signe : le mot sera affiché"
            out["suggestion"] = self.suggest(gloss)
        return out

    def suggest(self, gloss: str) -> str:
        alt = self.synonyms.get(gloss)
        if alt and playable(resolve(alt, self.index)):
            return alt
        if self.plain(gloss) in self.by_plain:                        # accent oublié : ETE -> ÉTÉ
            return self.by_plain[self.plain(gloss)]
        close = difflib.get_close_matches(gloss, self.vocabulary, n=1, cutoff=0.8)
        return close[0] if close else ""


# ------------------------------------------------------------------- rendu
def find_blender(given: str | None) -> str | None:
    for cand in (given, os.environ.get("BLENDER"), "/Applications/Blender.app/Contents/MacOS/Blender",
                 *sorted(glob.glob(r"C:\Program Files\Blender Foundation\Blender*\blender.exe"), reverse=True),
                 shutil.which("blender")):
        if cand and Path(cand).exists():
            return cand
    return None


class Renderer:
    """Un rendu Blender à la fois ; les travaux sont gardés en mémoire le temps de la session."""

    def __init__(self, blender: str | None, avatar: Path | None, glosses: Glosses, sleeveless: bool = False) -> None:
        self.blender, self.avatar, self.glosses, self.sleeveless = blender, avatar, glosses, sleeveless
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()                  # Blender : un seul rendu à la fois

    def video_name(self, glosses: list[str], subtitles: bool) -> str:
        key = hashlib.sha256(json.dumps([glosses, str(self.avatar or "mannequin"), self.sleeveless, subtitles]).encode()
                             + b"".join(p.read_bytes() for p in RENDER_CODE)).hexdigest()[:16]
        return f"{key}.mp4"

    def start(self, glosses: list[str], subtitles: bool = True) -> str:
        job = {"etat": "en attente", "avancement": 0.0, "gloses": glosses, "sous_titres": subtitles}
        jid = uuid.uuid4().hex[:12]
        self.jobs[jid] = job
        out = VIDEOS / self.video_name(glosses, subtitles)
        if out.exists():
            job.update(etat="fini", avancement=1.0, video=out.name)
        else:
            threading.Thread(target=self.run, args=(job, out), daemon=True).start()
        return jid

    def run(self, job: dict, out: Path) -> None:
        with self.lock:
            job["etat"] = "en cours"
            try:
                total = len(assemble(job["gloses"], self.glosses.index)[0]["hip"])   # images à 50 im/s
                VIDEOS.mkdir(exist_ok=True)
                tmp = out.with_name(out.stem + ".tmp.mp4")
                cmd = [self.blender, "-b", "--factory-startup", "--python", str(DEMO), "--",
                       "--gloses", " ".join(job["gloses"]), "--angle", "15", "--sortie", str(tmp)]
                if self.avatar:
                    cmd += ["--avatar", str(self.avatar)] + (["--sans-manches"] if self.sleeveless else [])
                if not job["sous_titres"]:
                    cmd.append("--sans-sous-titres")
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                        encoding="utf-8", errors="replace")
                tail: list[str] = []
                for line in proc.stdout:
                    tail = (tail + [line.rstrip()])[-15:]
                    if m := FRAME.search(line):
                        job["avancement"] = min(0.99, int(m.group(1)) / max(1, total))
                proc.wait()
                if proc.returncode != 0 or not tmp.exists():
                    raise RuntimeError("Blender n'a pas produit la vidéo :\n" + "\n".join(tail))
                tmp.replace(out)
                job.update(etat="fini", avancement=1.0, video=out.name)
            except Exception as e:                                       # montré sur la page
                job.update(etat="erreur", erreur=str(e))


# ------------------------------------------------------------------- serveur
def download_name(glosses: list[str]) -> str:
    words = "-".join(glosses[:8]).lower()
    plain = unicodedata.normalize("NFKD", words).encode("ascii", "ignore").decode()
    return "avatar-lsfb-" + (re.sub(r"[^a-z0-9-]+", "", plain)[:60] or "phrase") + ".mp4"


def handler(glosses: Glosses, renderer: Renderer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):                    # la fenêtre reste lisible
            pass

        def send_json(self, data, status: int = 200) -> None:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            url = urlparse(self.path)
            query = parse_qs(url.query)
            if url.path in ("/", "/index.html"):
                body = PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif url.path == "/api/verifier":
                words = split(query.get("gloses", [""])[0])
                self.send_json({"gloses": [glosses.check(g) for g in words],
                                "blender": bool(renderer.blender), "max": MAX_GLOSSES})
            elif url.path == "/api/vocabulaire":
                self.send_json(glosses.vocabulary)
            elif url.path.startswith("/api/travail/"):
                job = renderer.jobs.get(url.path.rsplit("/", 1)[-1])
                if job is None:
                    return self.send_json({"erreur": "travail inconnu"}, 404)
                self.send_json({k: v for k, v in job.items() if k not in ("gloses", "sous_titres")}
                               | ({"nom": download_name(job["gloses"])} if job.get("video") else {}))
            elif url.path.startswith("/videos/"):
                self.send_video(url.path.rsplit("/", 1)[-1], query)
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/generer":
                return self.send_error(404)
            try:
                data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            except json.JSONDecodeError:
                return self.send_json({"erreur": "requête illisible"}, 400)
            words = split(str(data.get("gloses", "")))
            if not words:
                return self.send_json({"erreur": "Écrivez au moins une glose."}, 400)
            if len(words) > MAX_GLOSSES:
                return self.send_json({"erreur": f"Phrase trop longue : {MAX_GLOSSES} gloses au plus."}, 400)
            if not renderer.blender:
                return self.send_json({"erreur": "Blender introuvable : installez-le (voir la notice)."}, 500)
            self.send_json({"id": renderer.start(words, bool(data.get("sous_titres", True)))})

        def send_video(self, name: str, query: dict) -> None:
            path = VIDEOS / name
            if not re.fullmatch(r"[0-9a-f]{16}\.mp4", name) or not path.exists():
                return self.send_error(404)
            size = path.stat().st_size
            start, end = 0, size - 1
            ranged = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
            if ranged and (ranged.group(1) or ranged.group(2)):        # Safari lit les vidéos par morceaux
                if ranged.group(1):
                    start = int(ranged.group(1))
                    end = min(int(ranged.group(2)), size - 1) if ranged.group(2) else size - 1
                else:
                    start = max(0, size - int(ranged.group(2)))
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if query.get("telecharger"):
                nom = query.get("nom", ["avatar-lsfb.mp4"])[0]
                self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{quote(nom)}")
            self.end_headers()
            with open(path, "rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0 and (chunk := f.read(min(1 << 16, left))):
                    self.wfile.write(chunk)
                    left -= len(chunk)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ouvrir", action="store_true", help="ouvrir la page dans le navigateur")
    ap.add_argument("--blender", help="chemin de Blender")
    ap.add_argument("--avatar", type=Path, default=AVATAR, help="personnage .vrm (VRoid Studio)")
    ap.add_argument("--sans-manches", action="store_true", help="retirer les manches du haut du personnage")
    ap.add_argument("--mannequin", action="store_true", help="avatar mannequin au lieu du personnage")
    args = ap.parse_args()
    glosses = Glosses()
    renderer = Renderer(find_blender(args.blender), None if args.mannequin else args.avatar.resolve(), glosses,
                        args.sans_manches)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(glosses, renderer))
    url = f"http://localhost:{args.port}"
    print(f"Avatar LSFB : la page est sur {url}")
    print(f"  {len(glosses.vocabulary)} gloses jouables ; avatar : {renderer.avatar.name if renderer.avatar else 'mannequin'}")
    print(f"  Blender : {renderer.blender or 'INTROUVABLE (installez-le, voir NOTICE.md)'}")
    print("  Laissez cette fenêtre ouverte ; fermez-la pour arrêter l'outil.")
    if args.ouvrir:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
