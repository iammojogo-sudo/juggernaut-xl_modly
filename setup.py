"""
SDXL Turbo — extension setup script.

Creates an isolated venv and installs all required dependencies.
Called by Modly at extension install time with:

    python setup.py <json_args>

where json_args contains:
    python_exe   — path to Modly's embedded Python (used to create the venv)
    ext_dir      — absolute path to this extension directory
    torch_flavor — Flavor of torch to use (cuda, rocm - defaults to cuda)
    gpu_sm       — GPU compute capability as integer (e.g. 61 for Pascal, 86 for Ampere; 0 on macOS)
    cuda_version — CUDA major/minor encoded as integer (e.g. 124, 128)
    accelerator  — "mps" | "cuda" | "cpu"  (passed by Electron since Modly 1.x)
    platform     — Electron's process.platform string ("win32", "darwin", "linux")

Example (manual test):
    python setup.py '{"python_exe":"C:/…/python.exe","ext_dir":"C:/…/sdxl-turbo","torch_flavor":"cuda","gpu_sm":86,"cuda_version":128}'
"""
import json
import platform
import subprocess
import sys
from pathlib import Path


def pip(venv: Path, *args: str) -> None:
    is_win = platform.system() == "Windows"
    pip_exe = venv / ("Scripts/pip.exe" if is_win else "bin/pip")
    subprocess.run([str(pip_exe), *args], check=True)


def _venv_python(ext_dir: Path) -> Path:
    is_win = platform.system() == "Windows"
    return ext_dir / "venv" / ("Scripts/python.exe" if is_win else "bin/python")


def resolve_models_dir(model_dir: str, ext_dir: Path) -> Path:
    import os
    md = model_dir or os.environ.get("MODELS_DIR")
    if md:
        return Path(md)
    return ext_dir.parent.parent / "models"


def download_weights(ext_dir: Path, models_dir: Path) -> None:
    import os
    manifest = json.loads((ext_dir / "manifest.json").read_text(encoding="utf-8"))
    ext_id = manifest["id"]
    nodes = manifest.get("nodes", []) or [manifest]

    for node in nodes:
        hf_repo = node.get("hf_repo")
        if not hf_repo:
            continue
        node_id = node.get("id", ext_id)
        target = models_dir / ext_id / node_id
        target.mkdir(parents=True, exist_ok=True)

        include = node.get("hf_include_prefixes") or []
        skip = node.get("hf_skip_prefixes") or []
        allow = [p for p in include if p]
        ignore = list(skip) + [
            "*.md", "LICENSE", "NOTICE", "Notice.txt", ".gitattributes",
        ]

        print(f"[setup] Downloading {hf_repo} -> {target}")
        snippet = (
            "import os;"
            "from huggingface_hub import snapshot_download;"
            f"snapshot_download(repo_id={hf_repo!r}, local_dir={str(target)!r}, "
            f"allow_patterns={allow!r}, ignore_patterns={ignore!r});"
            "print('DOWNLOAD_DONE')"
        )
        venv_py = _venv_python(ext_dir)
        env = os.environ.copy()
        env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        env.pop("HF_HUB_OFFLINE", None)
        subprocess.run([str(venv_py), "-c", snippet], check=True, env=env)
        print(f"[setup] Weights ready: {target}")


def setup(
    python_exe:    str,
    ext_dir:       Path,
    gpu_sm:        int,
    cuda_version:  int = 0,
    torch_flavor:  str = "cuda",
    accelerator:   str = "",
    platform_name: str = "",
    model_dir:     str = "",
) -> None:
    venv = ext_dir / "venv"
    machine = platform.machine().lower()
    is_win = platform.system() == "Windows"
    is_mac = platform.system() == "Darwin" or platform_name == "darwin"
    is_linux_arm64 = platform.system() == "Linux" and machine in {"aarch64", "arm64"}

    if not accelerator:
        if is_mac:
            accelerator = "mps" if machine == "arm64" else "cpu"
        elif gpu_sm > 0:
            accelerator = "cuda"
        else:
            accelerator = "cpu"

    print(f"[setup] accelerator={accelerator}  gpu_sm={gpu_sm}  cuda_version={cuda_version}")
    print(f"[setup] Creating venv at {venv} …")
    subprocess.run([python_exe, "-m", "venv", str(venv)], check=True)

    if is_mac:
        print("[setup] macOS -> PyTorch from standard PyPI (includes MPS)")
        pip(venv, "install", "torch", "torchvision")
    elif torch_flavor == "rocm":
        if is_win:
            print("[setup] WARNING: ROCm unsupported on Windows, using CPU PyTorch.")
            torch_index = "https://download.pytorch.org/whl/cpu"
            torch_pkgs  = ["torch==2.6.0", "torchvision==0.21.0"]
        else:
            torch_index = "https://download.pytorch.org/whl/rocm7.2"
            torch_pkgs  = ["torch", "torchvision"]
            print("[setup] -> PyTorch + ROCm 7.2")
        pip(venv, "install", *torch_pkgs, "--index-url", torch_index)
    elif gpu_sm >= 100 or cuda_version >= 128:
        torch_index = "https://download.pytorch.org/whl/cu128"
        torch_pkgs  = ["torch==2.7.0", "torchvision==0.22.0"]
        print(f"[setup] GPU SM {gpu_sm}, CUDA {cuda_version} -> PyTorch 2.7 + CUDA 12.8")
        pip(venv, "install", *torch_pkgs, "--index-url", torch_index)
    elif gpu_sm >= 70:
        torch_index = "https://download.pytorch.org/whl/cu124"
        torch_pkgs  = ["torch==2.6.0", "torchvision==0.21.0"]
        print(f"[setup] GPU SM {gpu_sm} -> PyTorch 2.6 + CUDA 12.4")
        pip(venv, "install", *torch_pkgs, "--index-url", torch_index)
    else:
        torch_index = "https://download.pytorch.org/whl/cu118"
        torch_pkgs  = ["torch==2.5.1", "torchvision==0.20.1"]
        print(f"[setup] GPU SM {gpu_sm} (legacy) -> PyTorch 2.5 + CUDA 11.8")
        pip(venv, "install", *torch_pkgs, "--index-url", torch_index)

    print("[setup] Installing Juggernaut XL dependencies …")
    pip(venv, "install",
        "Pillow",
        "numpy",
        "huggingface_hub",
        "diffusers>=0.27.0",
        "transformers>=4.36.0",
        "accelerate",
        "safetensors",
        "scipy",
    )

    # Download model weights into Modly's models/ folder so the extension is
    # ready to run with no manual steps and no network at generation time.
    models_dir = resolve_models_dir(model_dir, ext_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] Models dir: {models_dir}")
    download_weights(ext_dir, models_dir)

    print("[setup] Done. Venv + weights ready at:", venv)


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        setup(
            python_exe   = sys.argv[1],
            ext_dir      = Path(sys.argv[2]),
            gpu_sm       = int(sys.argv[3]),
            cuda_version = int(sys.argv[4]) if len(sys.argv) >= 5 else 0,
            torch_flavor = sys.argv[5] if len(sys.argv) >= 6 else "cuda",
            model_dir    = sys.argv[6] if len(sys.argv) >= 7 else "",
        )
    elif len(sys.argv) == 2:
        args = json.loads(sys.argv[1])
        setup(
            python_exe    = args["python_exe"],
            ext_dir       = Path(args["ext_dir"]),
            gpu_sm        = int(args["gpu_sm"]),
            cuda_version  = int(args.get("cuda_version", 0)),
            torch_flavor  = args.get("torch_flavor", "cuda"),
            accelerator   = args.get("accelerator", ""),
            platform_name = args.get("platform", ""),
            model_dir     = args.get("model_dir", ""),
        )
    else:
        print("Usage: python setup.py <python_exe> <ext_dir> <gpu_sm> [cuda_version] [torch_flavor]")
        print('   or: python setup.py \'{"python_exe":"...","ext_dir":"...","gpu_sm":86,"cuda_version":128}\'')
        sys.exit(1)
