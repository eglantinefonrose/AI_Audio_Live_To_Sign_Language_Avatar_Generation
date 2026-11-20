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
- `GET  /api/travail/<id>`      état du rendu (en cours, fini, erreur), avancement, et les signes de
                                la phrase avec leur début et leur fin dans la vidéo (`segments`) ;
- `GET  /videos/<nom>.mp4`      la vidéo (`?telecharger=1` : en pièce jointe, nom lisible) ;
- `POST /api/avis`              l'avis de l'utilisateur sur chaque signe (correct, à revoir,
                                mauvais ; ce qui ne va pas ; commentaire), ajouté à `retours/avis.csv` ;
- `GET  /api/avis`              nombre d'avis enregistrés ; `GET /api/avis.csv` : le fichier.

Envoi à l'équipe : si `config.json` donne l'adresse d'un tableau Google Sheets (`url_avis`,
application Web Apps Script de `google_sheets/avis.gs`, mise en place : `GOOGLE_SHEETS.md`),
chaque avis enregistré y est aussi envoyé (une ligne par signe jugé, sans nom ni adresse).
`POST /api/avis` renvoie l'avis en texte lisible : si l'envoi échoue ou n'est pas configuré,
la page le met dans un e-mail prêt à envoyer (`FEEDBACK_EMAIL` de `index.html`).

Les avis restent sur cet ordinateur (`retours/`, ignoré par git) : une ligne par signe jugé,
avec l'étiquette LSFB et le clip copié (`instance`), pour retrouver le signe à corriger.

Le rendu est celui de `avatar_animation/demo_phrase_blender.py` (Blender en arrière-plan,
vue de trois-quarts, glose en cours en sous-titre), avec le personnage VRoid du projet
(`avatar_animation/data/avatar_sign_4.vrm`), un autre (`--avatar`, `--sans-manches` pour
retirer des manches que les bras traversent) ou le mannequin (`--mannequin`). Une phrase déjà
rendue avec le même dictionnaire et le même code de rendu n'est pas refaite. Les vidéos vont
dans `videos/` (ignoré par git : dérivées de LSFB-ISOL, CC BY-NC-SA 4.0).

Seul numpy est nécessaire (le projet complet tire torch, moshi…) :

    uv run --no-project --with numpy python src/sign_language_3d_avatar/local_website/serveur.py [--ouvrir]
        [--port 8000] [--blender chemin] [--avatar fichier.vrm [--sans-manches] | --mannequin]
        [--url-avis https://script.google.com/macros/s/…/exec]

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
import urllib.request
import uuid
import webbrowser
from datetime import datetime
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
FEEDBACK = HERE / "retours" / "avis.csv"
CONFIG = HERE / "config.json"                     # {"url_avis": "…"} : tableau des avis de l'équipe
FEEDBACK_FIELDS = ["envoi", "date", "phrase", "glose", "signe_lsfb", "instance", "type", "verdict",
                   "problemes", "commentaire_signe", "commentaire_phrase", "avatar", "sous_titres"]
VERDICTS = ("correct", "à revoir", "mauvais")


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


def segments(glosses: list[str], index: dict) -> tuple[list[dict], int]:
    """(signes joués, dans l'ordre, avec début et fin dans la vidéo en secondes ; nombre
    d'images à 50 im/s). Les lettres d'un mot épelé forment un seul signe à juger, les mots
    sans signe (rien de joué) sont omis."""
    joints, timeline = assemble(glosses, index)
    fps, out = index["fps"], []
    for start, end, gloss, kind, detail, label in timeline:
        if kind == "absent":
            continue
        if kind == "lettre" and out and out[-1]["type"] == "épelé" and out[-1]["glose"] == gloss:
            out[-1]["fin"] = end / fps
            out[-1]["etiquettes"].append(label)
            continue
        out.append({"glose": gloss, "type": "épelé" if kind == "lettre" else kind,
                    "detail": "épelé" if kind == "lettre" else detail,
                    "etiquettes": [label] if label else [], "debut": start / fps, "fin": end / fps})
    return out, len(joints["hip"])


def summary(base: dict, rows: list[dict]) -> str:
    """Un envoi d'avis en texte, pour un e-mail."""
    lines = [f"Phrase : {base['phrase']}", f"Avatar : {base['avatar']} — mot signé affiché : {base['sous_titres']}",
             f"Date : {base['date']}", ""]
    for r in rows:
        if not r.get("verdict"):
            continue
        lines.append(f"{r['glose']} ({r['signe_lsfb'] or r['type']}"
                     + (f", clip {r['instance']}" if r["instance"] else "") + f") : {r['verdict']}")
        if r["problemes"]:
            lines.append(f"  Problèmes : {r['problemes']}")
        if r["commentaire_signe"]:
            lines.append(f"  Précision : {r['commentaire_signe']}")
    if base["commentaire_phrase"]:
        lines += ["", f"Commentaire sur la phrase : {base['commentaire_phrase']}"]
    return "\n".join(lines)


class Feedback:
    """Avis des utilisateurs sur les signes, ajoutés à un CSV local (lisible dans un tableur)."""

    def __init__(self, glosses: Glosses, avatar: str, url: str = "") -> None:
        self.glosses, self.avatar, self.url = glosses, avatar, url
        self.lock = threading.Lock()

    def forward(self, rows: list[dict]) -> bool:
        """Envoie les lignes au tableau de l'équipe (Apps Script) ; True si le tableau confirme."""
        req = urllib.request.Request(self.url, data=json.dumps({"lignes": rows}, ensure_ascii=False).encode("utf-8"),
                                     headers={"Content-Type": "text/plain; charset=utf-8"})
        try:                                 # Apps Script répond par une redirection, suivie en GET
            with urllib.request.urlopen(req, timeout=15) as r:
                return bool(json.loads(r.read() or b"{}").get("ok"))
        except Exception:
            return False

    def count(self) -> int:
        if not FEEDBACK.exists():
            return 0
        with open(FEEDBACK, newline="", encoding="utf-8-sig") as f:
            return len({r["envoi"] for r in csv.DictReader(f)})

    def save(self, data: dict) -> tuple[int, str, bool | None]:
        """Enregistre un envoi et l'envoie au tableau de l'équipe ; renvoie (nombre de signes
        jugés, l'avis en texte lisible, envoyé : oui, non, ou None sans tableau configuré). Les
        signes sont recalculés à partir de la phrase : l'étiquette LSFB et le clip ne
        dépendent pas de la page."""
        words = split(str(data.get("gloses", "")))[:MAX_GLOSSES]
        signs, _ = segments(words, self.glosses.index)
        clip = lambda l: self.glosses.index["signes"].get(l, {}).get("instance", "")
        def short(t, n=500) -> str:                  # tronqué, jamais lu comme une formule de tableur
            t = str(t or "").strip()[:n]
            return "'" + t if t[:1] in ("=", "+", "-", "@") else t
        base = {"envoi": uuid.uuid4().hex[:8], "date": datetime.now().isoformat(timespec="seconds"),
                "phrase": " ".join(words), "commentaire_phrase": short(data.get("commentaire"), 2000),
                "avatar": self.avatar, "sous_titres": "oui" if data.get("sous_titres", True) else "non"}
        rows = []
        for a in data.get("avis", []):
            i = a.get("i")
            if not isinstance(i, int) or not 0 <= i < len(signs) or a.get("verdict") not in VERDICTS:
                continue
            s = signs[i]
            rows.append(base | {"glose": s["glose"], "signe_lsfb": " ".join(s["etiquettes"]),
                                "instance": " ".join(clip(l) for l in s["etiquettes"]), "type": s["type"],
                                "verdict": a["verdict"],
                                "problemes": " ; ".join(short(x, 80) for x in a.get("problemes", [])[:12]),
                                "commentaire_signe": short(a.get("commentaire"))})
        if not rows and base["commentaire_phrase"]:
            rows.append(base)
        if rows:
            with self.lock:
                FEEDBACK.parent.mkdir(exist_ok=True)
                new = not FEEDBACK.exists()
                with open(FEEDBACK, "a", newline="", encoding="utf-8-sig" if new else "utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=FEEDBACK_FIELDS, restval="")
                    if new:
                        w.writeheader()
                    w.writerows(rows)
        sent = (self.forward(rows) if rows else False) if self.url else None
        return sum(1 for r in rows if r.get("verdict")), summary(base, rows), sent


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
        signs, total = segments(glosses, self.glosses.index)
        job = {"etat": "en attente", "avancement": 0.0, "gloses": glosses, "sous_titres": subtitles,
               "segments": signs, "images": total}
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
                total = job["images"]                                          # à 50 im/s
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


def handler(glosses: Glosses, renderer: Renderer, feedback: Feedback):
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
                self.send_json({k: v for k, v in job.items() if k not in ("gloses", "sous_titres", "images")}
                               | ({"nom": download_name(job["gloses"])} if job.get("video") else {}))
            elif url.path.startswith("/videos/"):
                self.send_video(url.path.rsplit("/", 1)[-1], query)
            elif url.path == "/api/avis":
                self.send_json({"n": feedback.count(), "envoi": bool(feedback.url)})
            elif url.path == "/api/avis.csv":
                if not FEEDBACK.exists():
                    return self.send_error(404)
                body = FEEDBACK.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", 'attachment; filename="avis-avatar-lsfb.csv"')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path not in ("/api/generer", "/api/avis"):
                return self.send_error(404)
            try:
                data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            except json.JSONDecodeError:
                return self.send_json({"erreur": "requête illisible"}, 400)
            if path == "/api/avis":
                judged, text, sent = feedback.save(data)
                return self.send_json({"enregistres": judged, "n": feedback.count(), "texte": text, "envoye": sent})
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


def running_tool(port: int) -> bool:
    """Un outil Avatar LSFB répond-il déjà sur ce port (lanceur double-cliqué deux fois) ?"""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/avis", timeout=2) as r:
            return "envoi" in json.loads(r.read())
    except Exception:
        return False


def bind(port: int, make_handler, tries: int = 10) -> tuple[ThreadingHTTPServer | None, int]:
    """Serveur sur le premier port libre à partir de `port` (un autre programme peut l'occuper)."""
    for p in range(port, port + tries):
        try:
            return ThreadingHTTPServer(("127.0.0.1", p), make_handler), p
        except OSError:
            continue
    return None, port


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ouvrir", action="store_true", help="ouvrir la page dans le navigateur")
    ap.add_argument("--blender", help="chemin de Blender")
    ap.add_argument("--avatar", type=Path, default=AVATAR, help="personnage .vrm (VRoid Studio)")
    ap.add_argument("--sans-manches", action="store_true", help="retirer les manches du haut du personnage")
    ap.add_argument("--mannequin", action="store_true", help="avatar mannequin au lieu du personnage")
    ap.add_argument("--url-avis", help="tableau Google Sheets des avis (sinon : url_avis de config.json)")
    args = ap.parse_args()
    if running_tool(args.port):
        page = f"http://localhost:{args.port}"
        print(f"Avatar LSFB est déjà ouvert : la page est sur {page}")
        print("  (une autre fenêtre de l'outil est déjà lancée). Vous pouvez fermer celle-ci.")
        if args.ouvrir:
            webbrowser.open(page)
        return
    config = json.loads(CONFIG.read_text(encoding="utf-8")) if CONFIG.exists() else {}
    url = (args.url_avis or config.get("url_avis") or "").strip()
    glosses = Glosses()
    renderer = Renderer(find_blender(args.blender), None if args.mannequin else args.avatar.resolve(), glosses,
                        args.sans_manches)
    feedback = Feedback(glosses, renderer.avatar.name if renderer.avatar else "mannequin", url)
    server, port = bind(args.port, handler(glosses, renderer, feedback))
    if server is None:
        sys.exit(f"Impossible de démarrer : les ports {args.port} à {args.port + 9} sont tous occupés. "
                 "Redémarrez l'ordinateur, puis relancez l'outil.")
    page = f"http://localhost:{port}"
    print(f"Avatar LSFB : la page est sur {page}")
    if port != args.port:
        print(f"  (le port {args.port} était occupé par un autre programme)")
    print(f"  {len(glosses.vocabulary)} gloses jouables ; avatar : {renderer.avatar.name if renderer.avatar else 'mannequin'}")
    print(f"  Blender : {renderer.blender or 'INTROUVABLE (installez-le, voir NOTICE.md)'}")
    print(f"  Avis : {'envoyés au tableau de l’équipe' if url else 'enregistrés sur cet ordinateur seulement'}")
    print("  Laissez cette fenêtre ouverte ; fermez-la pour arrêter l'outil.")
    if args.ouvrir:
        threading.Timer(0.5, lambda: webbrowser.open(page)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
