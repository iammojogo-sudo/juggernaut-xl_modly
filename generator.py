import io
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

_PLACEHOLDER_PATH = Path(__file__).parent / "placeholder.png"


def _placeholder_image_bytes() -> bytes:
    """1x1 image used when no input image is wired into the node."""
    try:
        return _PLACEHOLDER_PATH.read_bytes()
    except OSError:
        buf = io.BytesIO()
        Image.new("RGB", (1, 1), (0, 0, 0)).save(buf, "PNG")
        return buf.getvalue()


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

        # Force UTF-8 for stdout so the base class's Unicode print() works on Windows
        os.environ["PYTHONIOENCODING"] = "utf-8"
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

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
            variant="fp16",
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++"
        )

        if device == "cuda":
            pipe.enable_attention_slicing()
            try:
                pipe.enable_vae_slicing()
            except AttributeError:
                pass
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
        self._device = device
        self._dtype = dtype
        print(f"[JuggernautXL] Loaded on {device}.")

    def unload(self) -> None:
        super().unload()
        self._controlnet = None
        self._controlnet_loaded_repo = None
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
        # No image wired into the node? Fall back to the bundled 1x1
        # placeholder so image-consuming modes still run instead of failing.
        if not image_bytes:
            image_bytes = _placeholder_image_bytes()

        mode = str(params.get("mode", "generate")).lower()

        if mode == "img2img":
            return self._generate_img2img(image_bytes, params, progress_cb, cancel_event)
        elif mode == "inpaint":
            return self._generate_inpaint(image_bytes, params, progress_cb, cancel_event)

        return self._generate_text2img(image_bytes, params, progress_cb, cancel_event)

    def _generate_text2img(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        import torch

        if self._model is None:
            self.load()

        prompt = str(params.get("prompt", "")).strip()
        # Keep sampler settings stable so the node stays simple to use.
        num_steps = 30
        guidance_scale = 7.0
        width = height = 1024

        if not prompt:
            raise ValueError("A text prompt is required for generation.")

        self._report(progress_cb, 5, "Preparing generation…")
        self._check_cancelled(cancel_event)

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
            result = self._model(
                prompt=prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                output_type="pil",
            )
            image = result.images[0]
        finally:
            stop_evt.set()

        self._check_cancelled(cancel_event)

        self.unload()

        self._report(progress_cb, 95, "Saving image…")
        run = _modly_new_run_folder(self.outputs_dir)
        path = run / "source.png"
        image.save(str(path), "PNG")

        self._report(progress_cb, 100, "Done")
        return path

    def _generate_img2img(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        import torch
        from diffusers import StableDiffusionXLImg2ImgPipeline

        if self._model is None:
            self.load()

        prompt = str(params.get("prompt", "")).strip()
        strength = 0.7
        num_steps = 30
        guidance_scale = 7.0

        if not prompt:
            raise ValueError("A text prompt is required.")

        self._report(progress_cb, 5, "Preparing image…")
        self._check_cancelled(cancel_event)

        init_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        pipe = StableDiffusionXLImg2ImgPipeline(
            vae=self._model.vae,
            text_encoder=self._model.text_encoder,
            text_encoder_2=self._model.text_encoder_2,
            tokenizer=self._model.tokenizer,
            tokenizer_2=self._model.tokenizer_2,
            unet=self._model.unet,
            scheduler=self._model.scheduler,
            image_encoder=getattr(self._model, "image_encoder", None),
            feature_extractor=getattr(self._model, "feature_extractor", None),
        ).to(self._device)
        pipe.enable_attention_slicing()
        try:
            pipe.enable_vae_slicing()
        except AttributeError:
            pass

        self._report(progress_cb, 15, "Editing image…")
        stop_evt = threading.Event()
        if progress_cb:
            t = threading.Thread(
                target=smooth_progress,
                args=(progress_cb, 15, 90, "Editing image…", stop_evt),
                daemon=True,
            )
            t.start()

        try:
            result = pipe(
                prompt=prompt,
                image=init_image,
                strength=strength,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                output_type="pil",
            )
            image = result.images[0]
        finally:
            stop_evt.set()

        self._check_cancelled(cancel_event)

        self._report(progress_cb, 95, "Saving…")
        run = _modly_new_run_folder(self.outputs_dir)
        path = run / "edited.png"
        image.save(str(path), "PNG")

        self._report(progress_cb, 100, "Done")
        return path

    def _generate_inpaint(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        import torch
        from diffusers import StableDiffusionXLInpaintPipeline

        if self._model is None:
            self.load()

        prompt = str(params.get("prompt", "")).strip()
        strength = 0.95
        num_steps = 30
        guidance_scale = 7.5

        if not prompt:
            raise ValueError("A text prompt is required for inpainting.")

        self._report(progress_cb, 5, "Preparing image and mask…")
        self._check_cancelled(cancel_event)

        init_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        mask_path = params.get("mask_path") or params.get("image_path_2") or ""
        if not mask_path or not Path(mask_path).exists():
            raise ValueError("Inpainting requires a mask image. Wire a second image (mask) into the node inputs.")
        mask_image = Image.open(mask_path).convert("L")
        if mask_image.size != init_image.size:
            mask_image = mask_image.resize(init_image.size, Image.LANCZOS)

        pipe = StableDiffusionXLInpaintPipeline(
            vae=self._model.vae,
            text_encoder=self._model.text_encoder,
            text_encoder_2=self._model.text_encoder_2,
            tokenizer=self._model.tokenizer,
            tokenizer_2=self._model.tokenizer_2,
            unet=self._model.unet,
            scheduler=self._model.scheduler,
            image_encoder=getattr(self._model, "image_encoder", None),
            feature_extractor=getattr(self._model, "feature_extractor", None),
        ).to(self._device)
        pipe.enable_attention_slicing()
        try:
            pipe.enable_vae_slicing()
        except AttributeError:
            pass

        self._report(progress_cb, 15, "Inpainting…")
        stop_evt = threading.Event()
        if progress_cb:
            t = threading.Thread(
                target=smooth_progress,
                args=(progress_cb, 15, 90, "Inpainting…", stop_evt),
                daemon=True,
            )
            t.start()

        try:
            result = pipe(
                prompt=prompt,
                image=init_image,
                mask_image=mask_image,
                strength=strength,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                output_type="pil",
            )
            image = result.images[0]
        finally:
            stop_evt.set()

        self._check_cancelled(cancel_event)

        self._report(progress_cb, 95, "Saving…")
        run = _modly_new_run_folder(self.outputs_dir)
        path = run / "inpainted.png"
        image.save(str(path), "PNG")

        self._report(progress_cb, 100, "Done")
        return path

    @classmethod
    def params_schema(cls) -> list:
        return [
            {
                "id": "mode",
                "label": "Mode",
                "type": "select",
                "default": "generate",
                "options": [
                    {"value": "generate", "label": "Generate Image"},
                    {"value": "img2img", "label": "Edit Image"},
                    {"value": "inpaint", "label": "Inpaint Image"},
                ],
                "tooltip": "Create a new image, edit a connected image, or replace a masked area.",
            },
            {
                "id": "prompt",
                "label": "Prompt",
                "type": "string",
                "default": "",
                "tooltip": "Describe the image to create or the changes to make.",
            },
        ]
