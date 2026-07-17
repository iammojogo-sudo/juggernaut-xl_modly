import random
import sys
import os
import time
import threading
import uuid
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from services.generators.base import BaseGenerator, smooth_progress, GenerationCancelled

import json as _modly_json
import time as _modly_time
import uuid as _modly_uuid
from pathlib import Path as _modly_Path


def _modly_session_file(outputs_dir):
    return _modly_Path(outputs_dir) / ".modly_run.json"


def _modly_new_run_folder(outputs_dir):
    outputs_dir = _modly_Path(outputs_dir)
    run = outputs_dir / f"run_{int(_modly_time.time())}_{_modly_uuid.uuid4().hex[:8]}"
    run.mkdir(parents=True, exist_ok=True)
    _modly_session_file(outputs_dir).write_text(
        _modly_json.dumps({"run_folder": str(run)}), encoding="utf-8")
    return run


def _modly_current_run_folder(outputs_dir, params=None, input_path=None):
    outputs_dir = _modly_Path(outputs_dir)
    params = params or {}
    rf = params.get("run_folder") or ""
    if rf and _modly_Path(rf).is_dir():
        return _modly_Path(rf)
    src = input_path or params.get("input_path") or ""
    if src:
        p = _modly_Path(src)
        for cand in ([p] + list(p.parents)):
            if cand.is_dir() and cand.name.startswith("run_"):
                return cand
            if (cand / "source.png").exists() or (cand / "mesh.glb").exists() or (cand / "views").is_dir():
                return cand
    sf = _modly_session_file(outputs_dir)
    if sf.exists():
        try:
            data = _modly_json.loads(sf.read_text(encoding="utf-8"))
            p = _modly_Path(data.get("run_folder", ""))
            if p.is_dir():
                return p
        except Exception:
            pass
    return _modly_new_run_folder(outputs_dir)


_HF_REPO_ID = "RunDiffusion/Juggernaut-XL-v9"


class JuggernautXLGenerator(BaseGenerator):
    MODEL_ID     = "juggernaut-xl"
    DISPLAY_NAME = "Juggernaut XL"
    VRAM_GB      = 6

    def is_downloaded(self) -> bool:
        if self.download_check:
            return (self.model_dir / self.download_check).exists()
        return self.model_dir.exists() and any(self.model_dir.iterdir())

    def load(self) -> None:
        if self._model is not None:
            return

        if not self.is_downloaded():
            self._auto_download()

        # Never re-fetch from the network at runtime — weights live in Modly's
        # models/ folder, populated by setup.py / the manifest download.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        import torch
        from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler

        if sys.platform == "darwin":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            dtype = torch.float32
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32

        print(f"[JuggernautXL] Loading pipeline from {self.model_dir}…")
        pipe = AutoPipelineForText2Image.from_pretrained(
            str(self.model_dir),
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if dtype == torch.float16 else None,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++"
        )

        if device == "cuda":
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()
            # Attempt xformers memory-efficient attention. In this venv xformers
            # is not installed, so this is a no-op that falls back to PyTorch's
            # built-in SDPA (also memory-efficient). If xformers is added later
            # (or the pipeline runs in another env), this swaps the attention
            # processors to xformers for a small speed/memory win. Harmless if
            # unavailable — guarded so we keep SDPA.
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("[JuggernautXL] xformers memory-efficient attention enabled")
            except Exception as e:
                print(f"[JuggernautXL] xformers not available, using SDPA: {e}")
            # Adaptive device strategy. On >=8GB pin everything to GPU (no
            # per-step UNet shuffle); on 6GB keep CPU offload (the 4.9GB UNet
            # can't be pinned resident) and rely on torch.compile for speed.
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if total_gb >= 8:
                pipe = pipe.to("cuda")
            else:
                try:
                    pipe.enable_model_cpu_offload()
                except Exception:
                    pipe = pipe.to("cuda")
            # torch.compile the UNet (per-step compute speedup). Under offload
            # the original hook is preserved so the UNet still shuffles safely.
            # Needs the Triton backend, which is absent in the portable env, so
            # guard on it (compile would otherwise crash on the first forward).
            import importlib.util as _ilu
            if _ilu.find_spec("triton") is None:
                print("[JuggernautXL] UNet compile skipped (triton not installed)")
            else:
                try:
                    hook = getattr(pipe.unet, "_hf_hook", None)
                    pipe.unet = torch.compile(pipe.unet, dynamic=True)
                    if hook is not None:
                        pipe.unet._hf_hook = hook
                    print("[JuggernautXL] Compiled UNet (torch.compile)")
                except Exception as e:
                    print(f"[JuggernautXL] UNet compile skipped: {e}")
        else:
            pipe = pipe.to(device)

        self._model = pipe
        print(f"[JuggernautXL] Loaded on {device}.")

    def unload(self) -> None:
        super().unload()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except ImportError:
            pass

    def generate(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        import torch

        if self._model is None:
            self.load()

        prompt          = str(params.get("prompt", ""))
        negative_prompt = str(params.get("negative_prompt",
                                          ""))
        num_steps       = int(params.get("num_inference_steps", 30))
        guidance_scale  = float(params.get("guidance_scale", 5.0))
        size_str        = str(params.get("image_size", "768"))
        size            = int(size_str)
        seed            = int(params.get("seed", -1))
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        if not prompt:
            raise ValueError("A text prompt is required for generation.")

        self._report(progress_cb, 5, "Preparing generation…")
        self._check_cancelled(cancel_event)

        camera_view = str(params.get("camera_view", "none"))
        CAMERA_SUFFIXES = {
            "none": "",
            "front": ", front view, facing camera, centered, symmetrical",
            "right": ", right side view, facing right, profile",
            "back": ", back view, viewed from behind, facing away, rear",
            "left": ", left side view, facing left, profile",
        }
        camera_suffix = CAMERA_SUFFIXES.get(camera_view, "")

        self._report(progress_cb, 15, "Generating image…")
        stop_evt = threading.Event()
        if progress_cb:
            t = threading.Thread(
                target=smooth_progress,
                args=(progress_cb, 15, 90, "Generating image…", stop_evt),
                daemon=True,
            )
            t.start()

        try:
            generator = torch.Generator(device=self._model.device).manual_seed(seed)

            result = self._model(
                prompt=prompt + camera_suffix,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                width=size,
                height=size,
                generator=generator,
                output_type="pil",
            )
            image = result.images[0]
        finally:
            stop_evt.set()

        self._check_cancelled(cancel_event)

        self._report(progress_cb, 95, "Saving image…")
        run = _modly_new_run_folder(self.outputs_dir)
        path = run / "source.png"
        image.save(str(path), "PNG")

        self._report(progress_cb, 100, "Done")
        self.unload()
        return path

    @classmethod
    def params_schema(cls) -> list:
        return [
            {
                "id":      "prompt",
                "label":   "Prompt",
                "type":    "string",
                "default": "",
                "tooltip": "Describe the asset. For game characters: 'a full body human male character in tpose'.",
            },
            {
                "id":      "negative_prompt",
                "label":   "Negative Prompt",
                "type":    "string",
                "default": "",
                "tooltip": "Optional. Juggernaut XL works best with minimal or no negative prompt.",
            },
            {
                "id":      "num_inference_steps",
                "label":   "Quality Steps",
                "type":    "select",
                "default": 30,
                "options": [
                    {"value": 20, "label": "20 (Fast)"},
                    {"value": 30, "label": "30 (Balanced)"},
                    {"value": 40, "label": "40 (High)"},
                    {"value": 50, "label": "50 (Maximum)"},
                ],
                "tooltip": "DPM++ SDE Karras works best at 30-40 steps.",
            },
            {
                "id":      "guidance_scale",
                "label":   "Prompt Guidance",
                "type":    "float",
                "default": 5.0,
                "min":     1.0,
                "max":     15.0,
                "step":    0.5,
                "tooltip": "3-7 is standard for Juggernaut XL. Lower = more realistic.",
            },
            {
                "id":      "image_size",
                "label":   "Image Size",
                "type":    "select",
                "default": "768",
                "options": [
                    {"value": "768", "label": "768x768 (Safe for 6GB)"},
                    {"value": "896", "label": "896x896 (Balanced)"},
                    {"value": "1024", "label": "1024x1024 (Full, may need 8GB)"},
                ],
                "tooltip": "Larger = more detail for Zero123++, more VRAM.",
            },
            {
                "id":      "camera_view",
                "label":   "Camera View",
                "type":    "select",
                "default": "none",
                "options": [
                    {"value": "none", "label": "None (Free)"},
                    {"value": "front", "label": "Front View"},
                    {"value": "right", "label": "Right Side"},
                    {"value": "back", "label": "Back View"},
                    {"value": "left", "label": "Left Side"},
                ],
                "tooltip": "Augments prompt with camera direction for consistent Zero123++ input.",
            },
            {
                "id":      "seed",
                "label":   "Seed",
                "type":    "int",
                "default": -1,
                "min":     -1,
                "max":     2147483647,
                "tooltip": "Random seed (-1 for random).",
            },
        ]
