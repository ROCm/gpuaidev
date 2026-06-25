# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Helper plumbing for the ComfyUI-on-AMD-Instinct tutorial notebook.

This module keeps the boilerplate out of the notebook so the cells stay short
and readable. It provides:

- ``ComfyUIClient``        - a tiny HTTP client for a running ComfyUI server.
- ``start_comfyui_server`` - launch the server as a background process.
- ``stage_files``          - copy-or-download the tutorial inputs/workflows.
- ``require_files``        - report which expected files are missing.
- ``download_models``      - opt-in download of model checkpoints from Hugging Face.
- ``render_mesh_turntable``- deterministic CPU turntable preview of a .glb mesh.
- ``result_grid_html``     - build the inline photo / mesh / video result grid.
- ``benchmark_workflow``   - warm-up + timed iterations for Part 3.

Only the Python standard library is imported at module load time; the heavier
mesh/render dependencies are imported lazily inside ``render_mesh_turntable``.
"""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# Server lifecycle
# --------------------------------------------------------------------------- #
def server_is_up(server_url: str, timeout: float = 1.0) -> bool:
    """Return True if the ComfyUI ``/system_stats`` endpoint answers with 200."""
    try:
        with urllib.request.urlopen(f"{server_url}/system_stats", timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return False


def start_comfyui_server(
    comfyui_path: Path,
    server_url: str,
    port: int,
    log_path: Path,
    timeout_s: float = 180.0,
) -> int | None:
    """Start ``main.py --listen`` in the background and wait until it is ready.

    Returns the new process PID, or ``None`` if a server was already running at
    ``server_url`` (in which case it is reused).
    """
    if server_is_up(server_url):
        print(f"ComfyUI already responding at {server_url}; reusing it.")
        return None

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Starting ComfyUI server -> {server_url} (log: {log_path})")
    proc = subprocess.Popen(
        [sys.executable, str(Path(comfyui_path) / "main.py"), "--listen", "--port", str(port)],
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        cwd=str(comfyui_path),
    )
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"ComfyUI server exited with code {proc.returncode}. See {log_path}."
            )
        if server_is_up(server_url):
            print(f"Server ready (pid={proc.pid}).")
            return proc.pid
        time.sleep(1.0)
    raise TimeoutError(f"ComfyUI did not become ready within {timeout_s:.0f}s; see {log_path}.")


# --------------------------------------------------------------------------- #
# HTTP client
# --------------------------------------------------------------------------- #
class ComfyUIClient:
    """Minimal client for a running ComfyUI server (``/prompt``, ``/history``, ``/view``)."""

    _OUTPUT_KEYS = ("images", "gifs", "videos", "files", "model", "meshes", "3d")

    def __init__(self, server_url: str) -> None:
        self.server_url = server_url.rstrip("/")

    def queue_prompt(self, workflow: dict, client_id: str | None = None) -> str:
        """POST a workflow to ``/prompt`` and return its ``prompt_id``."""
        payload: dict[str, Any] = {"prompt": workflow}
        if client_id is not None:
            payload["client_id"] = client_id
        req = urllib.request.Request(
            f"{self.server_url}/prompt",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            body = json.loads(resp.read())
        if "prompt_id" not in body:
            raise RuntimeError(f"/prompt did not return a prompt_id: {body}")
        return body["prompt_id"]

    def wait_for_completion(self, prompt_id: str, poll_s: float = 1.0, timeout_s: float = 1800.0) -> dict:
        """Poll ``/history/<prompt_id>`` until the run finishes (or times out)."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with urllib.request.urlopen(f"{self.server_url}/history/{prompt_id}") as resp:
                hist = json.loads(resp.read())
            if prompt_id in hist:
                entry = hist[prompt_id]
                status = entry.get("status", {})
                if status.get("completed", False) or status.get("status_str") in ("success", "error"):
                    return entry
            time.sleep(poll_s)
        raise TimeoutError(f"Prompt {prompt_id} did not complete within {timeout_s}s.")

    def _view(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        qs = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": folder_type})
        with urllib.request.urlopen(f"{self.server_url}/view?{qs}") as resp:
            return resp.read()

    def download_outputs(self, history_entry: dict, dest_dir: Path) -> list[Path]:
        """Fetch every generated file referenced by ``history_entry`` into ``dest_dir``."""
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []
        for node_out in history_entry.get("outputs", {}).values():
            for key in self._OUTPUT_KEYS:
                for item in node_out.get(key, []) or []:
                    fname = item.get("filename")
                    if not fname:
                        continue
                    data = self._view(fname, item.get("subfolder", ""), item.get("type", "output"))
                    out_path = dest_dir / fname
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_bytes(data)
                    saved.append(out_path)
        return saved

    def run_workflow(self, workflow: dict, dest_dir: Path) -> tuple[list[Path], dict]:
        """Submit a workflow, wait for it to finish, and download its outputs."""
        wf = deepcopy(workflow)
        prompt_id = self.queue_prompt(wf, client_id=str(uuid.uuid4()))
        entry = self.wait_for_completion(prompt_id)
        return self.download_outputs(entry, dest_dir), entry


# --------------------------------------------------------------------------- #
# File staging
# --------------------------------------------------------------------------- #
def stage_files(names: list[str], dest_dir: Path, local_srcs: list[Path], base_url: str) -> None:
    """Ensure each file in ``names`` exists in ``dest_dir``.

    Tries ``local_srcs`` first (a local checkout), then downloads from
    ``{base_url}/{name}``.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        dest = dest_dir / name
        if dest.is_file():
            print(f"  present:  {name}")
            continue
        for src in local_srcs:
            candidate = Path(src) / name
            if candidate.is_file() and candidate.resolve() != dest.resolve():
                shutil.copy2(candidate, dest)
                print(f"  copied:   {name}  <-  {src}")
                break
        else:
            url = f"{base_url}/{name}"
            print(f"  download: {name}  <-  {url}")
            urllib.request.urlretrieve(url, dest)


def require_files(paths: list[Path]) -> list[Path]:
    """Return the subset of ``paths`` that do not exist on disk."""
    return [Path(p) for p in paths if not Path(p).is_file()]


def _download_with_progress(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` (atomic via a .part file) with a progress line."""
    dest = Path(dest)
    tmp = dest.with_suffix(dest.suffix + ".part")

    def _hook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        done = min(block_num * block_size, total_size)
        pct = 100.0 * done / total_size
        print(f"\r    {pct:5.1f}%  ({done / 1e9:.2f} / {total_size / 1e9:.2f} GB)", end="")

    urllib.request.urlretrieve(url, tmp, reporthook=_hook)
    print()  # newline after the progress line
    tmp.replace(dest)


def download_models(models: dict[Path, str], enabled: bool) -> list[Path]:
    """Fetch each ``{destination: url}`` model file that is not already present.

    Downloads are gated on ``enabled``: when it is ``False`` nothing is fetched
    and the function only reports which files are missing, so the caller can
    decide to skip the workflow. Existing files are always skipped, so re-running
    is cheap. Returns the list of destinations still missing after the attempt.
    """
    for dest, url in models.items():
        dest = Path(dest)
        if dest.is_file():
            print(f"  present:  {dest.name}")
            continue
        if not enabled:
            print(f"  missing:  {dest.name}  (set DOWNLOAD_MODELS=True to fetch it automatically)")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"  download: {dest.name}  <-  {url}")
        _download_with_progress(url, dest)
    return require_files(list(models.keys()))


# --------------------------------------------------------------------------- #
# Mesh preview (deterministic, CPU-only - not an AI step)
# --------------------------------------------------------------------------- #
def render_mesh_turntable(
    glb_path: Path,
    out_mp4: Path,
    n_frames: int = 36,
    fps: int = 24,
    target_faces: int = 40_000,
    img_size: int = 480,
    base_color: tuple[float, float, float] = (0.82, 0.55, 0.42),
    elev_deg: float = 5.0,
) -> Path:
    """Render a short MP4 turntable of a .glb mesh with painter's-algorithm shading.

    Pure CPU, no GL context. Heavy dependencies are imported here so that simply
    importing this module does not require them.
    """
    import numpy as np
    import trimesh
    import fast_simplification
    from PIL import Image, ImageDraw
    import imageio.v2 as imageio

    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.load(glb_path, force="mesh", process=True)
    if len(mesh.faces) > target_faces:
        verts, faces = fast_simplification.simplify(
            mesh.vertices.astype(np.float32),
            mesh.faces.astype(np.uint32),
            target_count=target_faces,
        )
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    centre = (mesh.bounds[0] + mesh.bounds[1]) * 0.5
    mesh.vertices = mesh.vertices - centre
    mesh.vertices = mesh.vertices / np.linalg.norm(mesh.vertices, axis=1).max()

    V = mesh.vertices.astype(np.float32)
    F = mesh.faces.astype(np.int32)

    light = np.array([0.4, 0.6, 0.7], dtype=np.float32)
    light /= np.linalg.norm(light)
    base = np.array(base_color, dtype=np.float32)

    elev = np.radians(elev_deg)
    Rx = np.array(
        [[1, 0, 0],
         [0, np.cos(elev), -np.sin(elev)],
         [0, np.sin(elev), np.cos(elev)]],
        dtype=np.float32,
    )

    H = W = img_size
    half = (img_size - 4) * 0.5
    cx = cy = img_size * 0.5

    frames = []
    for i in range(n_frames):
        theta = np.radians(360.0 * i / n_frames)
        c, s = np.cos(theta), np.sin(theta)
        Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        VR = (V @ Ry.T) @ Rx.T

        tri = VR[F]
        v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
        normals = np.cross(v1 - v0, v2 - v0)
        ln = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(ln, 1e-8)
        front = normals[:, 2] > 0.0
        if not front.any():
            frames.append(np.full((H, W, 3), 255, dtype=np.uint8))
            continue
        v0f, v1f, v2f, nf = v0[front], v1[front], v2[front], normals[front]

        x0 = cx + v0f[:, 0] * half; y0 = cy - v0f[:, 1] * half; z0 = v0f[:, 2]
        x1 = cx + v1f[:, 0] * half; y1 = cy - v1f[:, 1] * half; z1 = v1f[:, 2]
        x2 = cx + v2f[:, 0] * half; y2 = cy - v2f[:, 1] * half; z2 = v2f[:, 2]
        depth = (z0 + z1 + z2) / 3.0
        order = np.argsort(depth)

        intensity = np.clip(nf @ light, 0.18, 1.0)
        face_rgb = (np.clip(base[None, :] * intensity[:, None], 0, 1) * 255).astype(np.uint8)

        img = Image.new("RGB", (W, H), (255, 255, 255))
        drw = ImageDraw.Draw(img, "RGB")
        for idx in order:
            r, g, b = int(face_rgb[idx, 0]), int(face_rgb[idx, 1]), int(face_rgb[idx, 2])
            drw.polygon(
                [(x0[idx], y0[idx]), (x1[idx], y1[idx]), (x2[idx], y2[idx])],
                fill=(r, g, b),
            )
        frames.append(np.asarray(img))

    imageio.mimsave(out_mp4, frames, fps=fps, codec="libx264", quality=8, macro_block_size=1)
    return out_mp4


# --------------------------------------------------------------------------- #
# Inline result grid
# --------------------------------------------------------------------------- #
def _b64(path: Path, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


def result_grid_html(
    inputs: list[tuple[str, Path]],
    meshes: dict[str, Path],
    turntable_videos: dict[str, Path],
    wan_videos: dict[str, Path],
) -> str:
    """Build a 3-row HTML grid (input photo / mesh turntable / AI video) as base64.

    The returned string is self-contained, so the rendered notebook needs no
    external files.
    """
    img_style = "width:280px;border:1px solid #ddd;border-radius:4px"
    cap_style = "font-family:monospace;font-size:11px;margin-top:6px;line-height:1.5"

    def photo(name: str, asset: Path) -> str:
        return (
            '<td style="padding:6px;text-align:center;vertical-align:bottom">'
            f'<img src="{_b64(asset, "image/png")}" style="{img_style}"/>'
            f'<div style="{cap_style}">{name}</div></td>'
        )

    def video(mp4: Path | None, caption: str, empty: str) -> str:
        if mp4 is None or not Path(mp4).is_file():
            return f'<td style="padding:6px;text-align:center;color:#888">{empty}</td>'
        return (
            '<td style="padding:6px;text-align:center;vertical-align:top">'
            f'<video src="{_b64(mp4, "video/mp4")}" controls loop autoplay muted style="{img_style}"></video>'
            f'<div style="{cap_style}">{caption}</div></td>'
        )

    n = len(inputs)
    photo_row = "".join(photo(name, asset) for name, asset in inputs)
    mesh_row = "".join(
        video(turntable_videos.get(name),
              f'<code>{meshes[name].name}</code>' if meshes.get(name) else "mesh",
              "(no mesh turntable)")
        for name, _ in inputs
    )
    video_row = "".join(
        video(wan_videos.get(name), "81 frames @ 24 fps", "(no video)")
        for name, _ in inputs
    )

    def header(text: str) -> str:
        return (f'<tr><th colspan="{n}" style="text-align:left;padding:14px 6px 8px;'
                f'font-weight:600">{text}</th></tr>')

    return (
        '<table style="border-collapse:collapse;margin:0 auto;font-family:sans-serif">'
        + header("Input photographs (768x768)") + f"<tr>{photo_row}</tr>"
        + header("Hunyuan3D v2.1 mesh - deterministic CPU turntable render") + f"<tr>{mesh_row}</tr>"
        + header("Wan2.2 5B image-to-video - AI-generated camera orbit") + f"<tr>{video_row}</tr>"
        + "</table>"
    )


# --------------------------------------------------------------------------- #
# Part 3 benchmark harness (measures YOUR hardware; no published numbers)
# --------------------------------------------------------------------------- #
def _prep_workflow(wf_template: dict, image_input: str | None) -> dict:
    wf = deepcopy(wf_template)
    for node in wf.values():
        if image_input is not None and node.get("class_type") == "LoadImage":
            node["inputs"]["image"] = image_input
        if node.get("class_type") in ("KSampler", "KSamplerAdvanced") and "seed" in node.get("inputs", {}):
            node["inputs"]["seed"] = int(time.time() * 1_000_000) % (2**63 - 1)
    return wf


def benchmark_workflow(
    client: ComfyUIClient,
    workflow_path: Path,
    checkpoints: list[Path],
    image_input: str | None,
    iterations: int,
    out_dir: Path,
) -> dict | None:
    """Run one warm-up pass plus ``iterations`` timed passes; return local stats.

    Returns ``None`` (and prints why) if the workflow JSON or any checkpoint is
    missing. Timings are wall-clock seconds on the caller's own hardware.
    """
    workflow_path = Path(workflow_path)
    if not workflow_path.is_file():
        print(f"  skipped - workflow JSON not found: {workflow_path}")
        return None
    missing = require_files(checkpoints)
    if missing:
        print(f"  skipped - missing checkpoint(s): {[str(c) for c in missing]}")
        return None

    with workflow_path.open() as f:
        wf_template = json.load(f)

    print("  warm-up...")
    client.run_workflow(_prep_workflow(wf_template, image_input), Path(out_dir) / "_warmup")

    times: list[float] = []
    for it in range(iterations):
        t0 = time.perf_counter()
        client.run_workflow(_prep_workflow(wf_template, image_input), Path(out_dir) / f"iter{it}")
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"  iter {it}: {dt:.2f} s")

    mean = sum(times) / len(times)
    var = sum((t - mean) ** 2 for t in times) / len(times)
    return {"times_s": times, "mean_s": mean, "std_s": var ** 0.5}
