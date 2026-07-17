"""
SDXL Turbo / Juggernaut XL — extension setup script.

Creates an isolated venv and installs all required dependencies (including
torch). Model weights are downloaded separately by Modly's model-install step
(driven by the manifest's hf_repo / hf_include_prefixes / download_check),
into Modly's models/ folder, and are read OFFLINE at generation time.

Called by Modly at extension install time with:

    python setup.py '<json_args>'

where json_args contains:
    python_exe   — path to Modly's embedded Python (used to create the venv)
    ext_dir      — absolute path to this extension directory
    gpu_sm       — GPU compute capability as integer
    cuda_version — CUDA major/minor encoded as integer
    torch_flavor — Flavor of torch to use (cuda, rocm - defaults to cuda)
    accelerator  — "mps" | "cuda" | "cpu"
    platform     — Electron's process.platform string
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

    print("[setup] Done. Venv ready at:", venv)
    print("[setup] Model weights are installed via Modly's model-download step.")


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
