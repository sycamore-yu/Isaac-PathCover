"""Install the pinned Isaac Lab source components and simulation dependencies."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
CHECKOUT = ROOT / ".runtime" / "isaaclab"
REPOSITORY = "https://github.com/isaac-sim/IsaacLab.git"
REVISION = "e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4"
COMPONENTS = ("isaaclab", "isaaclab_newton", "isaaclab_contrib", "isaaclab_physx")


def git(*arguments, capture=False):
    command = ["git", "-C", str(CHECKOUT), *arguments]
    if capture:
        return subprocess.check_output(command, text=True).strip()
    subprocess.run(command, check=True)


def prepare_checkout():
    if not CHECKOUT.resolve().is_relative_to(ROOT):
        raise RuntimeError("Dependency checkout must remain inside this repository")
    if not CHECKOUT.exists():
        CHECKOUT.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", str(CHECKOUT)], check=True)
        git("remote", "add", "origin", REPOSITORY)
        git("sparse-checkout", "init", "--cone")
        git("sparse-checkout", "set", "apps", *(f"source/{name}" for name in COMPONENTS))
        git("fetch", "--depth=1", "origin", REVISION)
        git("checkout", "--detach", "FETCH_HEAD")
    if not (CHECKOUT / ".git").exists():
        raise RuntimeError(f"Existing dependency directory is not a Git checkout: {CHECKOUT}")
    revision = git("rev-parse", "HEAD", capture=True)
    if revision != REVISION:
        raise RuntimeError(f"Expected Isaac Lab {REVISION}, found {revision}; existing checkout was not changed")
    git("diff", "--quiet", "HEAD", "--")
    for name in COMPONENTS:
        if not (CHECKOUT / "source" / name / "pyproject.toml").is_file():
            raise RuntimeError(f"Required Isaac Lab component is missing: {name}")
    if not (CHECKOUT / "apps").is_dir():
        raise RuntimeError("Required Isaac Lab apps directory is missing")
    print(f"Isaac Lab v3.0.0 verified at {CHECKOUT} ({revision})", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", help="Fetch and verify source without modifying a Python environment")
    parser.add_argument("--venv", type=Path, default=ROOT / ".venv", help="Target environment inside this repository")
    parser.add_argument("--uv", default="uv", help="uv executable, including an absolute path when it is not on PATH")
    args = parser.parse_args()
    environment = (ROOT / args.venv).resolve()
    if not environment.is_relative_to(ROOT):
        parser.error("--venv must remain inside this repository")
    prepare_checkout()
    if args.prepare_only:
        return
    executable = shutil.which(args.uv)
    if executable is None:
        parser.error("uv was not found; install uv or provide --uv /absolute/path/to/uv")
    process_environment = dict(os.environ)
    process_environment["UV_PROJECT_ENVIRONMENT"] = str(environment)
    subprocess.run(
        [executable, "sync", "--locked", "--python", "3.12", "--extra", "simulation", "--extra", "test"],
        cwd=ROOT, env=process_environment, check=True,
    )
    print(f"Simulation environment installed at {environment}", flush=True)


if __name__ == "__main__":
    main()
