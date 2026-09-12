"""
Run nanochat's runs/speedrun.sh on Modal instead of Lambda.

Usage:
    modal run modal_speedrun.py
    # with wandb logging:
    modal run modal_speedrun.py --wandb-run=speedrun
"""

import os
import subprocess

import modal

app = modal.App("nanochat-speedrun")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "curl", "build-essential")
    .pip_install("uv")
    .add_local_dir(
        ".",
        remote_path="/root/nanochat",
        copy=True,
        ignore=[".venv", "__pycache__", "*.pyc", ".git", "wandb"],
    )
)

# Persists ~/.cache/nanochat (data shards, tokenizer, checkpoints) across runs
cache_volume = modal.Volume.from_name("nanochat-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100:8",
    timeout=4 * 60 * 60,  # speedrun.sh takes ~1.5-2h; leave headroom
    volumes={"/root/.cache/nanochat": cache_volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def speedrun(wandb_run: str = "dummy"):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["WANDB_RUN"] = wandb_run
    subprocess.run(
        ["bash", "runs/speedrun.sh"],
        cwd="/root/nanochat",
        env=env,
        check=True,
    )
    cache_volume.commit()


@app.function(
    image=image,
    gpu="H100:1",
    timeout=10 * 60,
    volumes={"/root/.cache/nanochat": cache_volume},
)
def chat(prompt: str) -> str:
    result = subprocess.run(
        ["python", "-m", "scripts.chat_cli", "-p", prompt],
        cwd="/root/nanochat",
        env={**os.environ, "OMP_NUM_THREADS": "1"},
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


@app.local_entrypoint()
def main(wandb_run: str = "dummy", prompt: str = ""):
    if prompt:
        print(chat.remote(prompt))
    else:
        speedrun.remote(wandb_run)
