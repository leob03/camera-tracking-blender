"""Train and review a static Gaussian environment with a separately installed LichtFeld."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from worker import digest, write_status
from splat_dataset import prepare_dataset


def settings_checked(settings):
    defaults = {
        "splat_iterations": 15000,
        "splat_max_dimension": 1536,
        "splat_frame_step": 3,
        "splat_max_gaussians": 1000000,
    }
    limits = {
        "splat_iterations": (100, 60000),
        "splat_max_dimension": (256, 2048),
        "splat_frame_step": (1, 20),
        "splat_max_gaussians": (50000, 1500000),
    }
    result = {}
    for key, default in defaults.items():
        v = settings.get(key, default)
        if type(v) is not int or not limits[key][0] <= v <= limits[key][1]:
            raise ValueError("Invalid " + key)
        result[key] = v
    return result


def trainer_command(executable, dataset, run, settings):
    s = settings_checked(settings)
    return [
        str(executable),
        "--headless",
        "--no-splash",
        "--safe-mode",
        "--no-download",
        "-d",
        str(dataset),
        "-o",
        str(run),
        "--iter",
        str(s["splat_iterations"]),
        "--strategy",
        "mrnf",
        "-r",
        "1",
        "--max-width",
        str(s["splat_max_dimension"]),
        "--test-every",
        "8",
        "--eval",
        "--eval-steps",
        str(s["splat_iterations"]),
        "--mask-mode",
        "ignore",
        "--min-track-length",
        "0",
        "--centralize",
        "off",
        "--max-cap",
        str(s["splat_max_gaussians"]),
        "--enable-mip",
        "--background-improvements",
        "--no-normal-auto-generate",
        "--export",
        "ply",
        "--output-name",
        "scene",
    ]


def run(request_path, config_path):
    request = json.loads(request_path.read_text())
    cfg = json.loads(config_path.read_text())
    work = request_path.parent
    out = Path(request["output"])
    s = settings_checked(request["settings"])
    executable = Path(cfg.get("lichtfeld_executable", ""))
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError(
            "Configure lichtfeld_executable on the worker before training splats."
        )
    if shutil.disk_usage(work).free < 5 * 1024**3:
        raise RuntimeError(
            "Splat training needs at least 5 GiB free on its work filesystem."
        )
    start = time.monotonic()
    before = digest(out / "camera.npz")
    if digest(Path(request["source"])) != request["source_sha256"]:
        raise ValueError("The source plate changed after the camera solve.")
    write_status(work, "running", "Preparing splat training dataset", 10)
    dataset = work / "splat_dataset"
    report = prepare_dataset(
        out,
        request["source"],
        work / "solve_masks.zip",
        dataset,
        s,
        progress=lambda msg: print("[ViPE splat] " + msg, flush=True),
    )
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.update(
        LFS_HOME=str(work / "splat_home"),
        XDG_CACHE_HOME=str(work / "splat_cache"),
        OMP_NUM_THREADS=str(cfg.get("cpu_threads", 2)),
        OPENBLAS_NUM_THREADS="2",
        PYTHONNOUSERSITE="1",
    )
    run_dir = work / "splat_training"
    command = trainer_command(executable, dataset, run_dir, s)
    write_status(work, "running", "Training Gaussian splat", 30)
    print("[ViPE splat] Command: " + json.dumps(command), flush=True)
    subprocess.run(command, cwd=work, env=env, check=True)
    ply = run_dir / "scene.ply"
    if not ply.is_file():
        alternate = run_dir / f"splat_{s['splat_iterations']}.ply"
        if alternate.is_file():
            ply = alternate
        else:
            raise RuntimeError("LichtFeld completed without the expected Gaussian PLY.")
    target = out / "splat"
    target.mkdir(exist_ok=True)
    shutil.copyfile(ply, target / "scene.ply")
    shutil.copyfile(dataset / "masks_preview.jpg", target / "masks_preview.jpg")
    shutil.copyfile(dataset / "dataset.json", target / "dataset.json")
    write_status(work, "running", "Rendering Gaussian splat previews", 85)
    width, height = report["width"], report["height"]
    preview_scale = min(1.0, 1280 / width)
    width = max(2, round(width * preview_scale / 2) * 2)
    height = max(2, round(height * preview_scale / 2) * 2)
    for source, name in [
        ("camera_path.json", "camera_preview.mp4"),
        ("nearby_path.json", "nearby_preview.mp4"),
    ]:
        command = [
            str(executable),
            "--render-camera-path",
            str(dataset / source),
            "--render-load",
            str(target / "scene.ply"),
            "--render-output",
            str(target / name),
            "--render-width",
            str(width),
            "--render-height",
            str(height),
            "--render-fps",
            str(max(1, round(report["fps"]))),
            "--render-crf",
            "18",
            "--no-download",
        ]
        subprocess.run(command, cwd=work, env=env, check=True)
    if digest(out / "camera.npz") != before:
        raise RuntimeError("Splat stage changed the solved camera.")
    m = json.loads((out / "result.json").read_text())
    m["version"] = 4
    m["splat"] = dict(
        schema="vipe-gaussian-splat",
        version=1,
        trainer="LichtFeld Studio",
        strategy="mrnf",
        file="splat/scene.ply",
        previews=["splat/camera_preview.mp4", "splat/nearby_preview.mp4"],
        mask_preview="splat/masks_preview.jpg",
        dataset_report="splat/dataset.json",
        camera_sha256=before,
        camera_unchanged=True,
        settings=s,
        elapsed_seconds=time.monotonic() - start,
        coordinates="LichtFeld COLMAP world: diag(1,-1,-1) @ ViPE world; rotate 180 degrees around X to align with ViPE points",
        note="Static scene appearance. Moving subjects are ignored. Depth initializes geometry; no depth loss or camera refinement.",
    )
    for name in [
        "scene.ply",
        "camera_preview.mp4",
        "nearby_preview.mp4",
        "masks_preview.jpg",
        "dataset.json",
    ]:
        f = target / name
        m["files"]["splat/" + name] = {"sha256": digest(f), "bytes": f.stat().st_size}
    if sum(x["bytes"] for x in m["files"].values()) > 500 * 1024**2:
        raise RuntimeError(
            "Splat result exceeds the 500 MiB bundle budget; reduce Gaussian count or iterations."
        )
    (out / "result.json").write_text(json.dumps(m, indent=2))
    write_status(work, "running", "Gaussian splat and previews ready", 94)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--request", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    a = p.parse_args()
    run(a.request, a.config)
