import io
import os
import random
import sys
import threading
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

    # ------------------------------------------------------------------ #
    # Node / path helpers
    # ------------------------------------------------------------------ #

    def _node_id(self) -> str:
        """Node id is the trailing component of model_dir (set by Modly)."""
        return Path(self.model_dir).name.lower()

    def _base_model_dir(self) -> Path:
        """Weights dir for this node.

        Every node (generate/edit/inpaint) keeps its own folder, but they all
        use the same SDXL checkpoint — so share whichever node folder actually
        holds the weights instead of downloading ~7GB three times.
        """
        if self.download_check:
            if (self.model_dir / self.download_check).exists():
                return self.model_dir
            try:
                for candidate in sorted(self.model_dir.parent.iterdir()):
                    if candidate.is_dir() and (candidate / self.download_check).exists():
                        return candidate
            except OSError:
                pass
        return self.model_dir

    def is_downloaded(self) -> bool:
        base = self._base_model_dir()
        return base.exists() and any(base.iterdir())

    # ------------------------------------------------------------------ #
    # Model lifecycle
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        if self._model is not None:
            return

        # Force UTF-8 so the base class's Unicode print() works on Windows.
        os.environ["PYTHONIOENCODING"] = "utf-8"
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

        if not self.is_downloaded():
            self._auto_download()

        # Weights live in Modly's models/ folder (populated by the manifest
        # download) — never fetch from the network at generation time.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        import torch

        if sys.platform == "darwin":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            dtype = torch.float32
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32

        node = self._node_id()

        # The "segment" node runs CLIPSeg, not SDXL — no diffusers pipeline.
        if node == "segment":
            self._load_clipseg(device)
            self._device = device
            self._dtype = torch.float32
            print(f"[JuggernautXL] Loaded 'segment' on {device}.")
            return

        from diffusers import (
            AutoPipelineForText2Image,
            StableDiffusionXLImg2ImgPipeline,
            StableDiffusionXLInpaintPipeline,
            DPMSolverMultistepScheduler,
        )

        base = str(self._base_model_dir())
        load_kwargs = dict(
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16",
        )

        if node in ("edit", "img2img"):
            print(f"[JuggernautXL] Loading img2img pipeline from {base}…")
            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(base, **load_kwargs)
        elif node == "inpaint":
            print(f"[JuggernautXL] Loading inpaint pipeline from {base}…")
            pipe = StableDiffusionXLInpaintPipeline.from_pretrained(base, **load_kwargs)
        else:
            print(f"[JuggernautXL] Loading text2img pipeline from {base}…")
            pipe = AutoPipelineForText2Image.from_pretrained(base, **load_kwargs)

        # Juggernaut XL v9 recommends DPM++ 2M Karras.
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True
        )
        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass

        if device == "cuda":
            pipe.enable_attention_slicing()
            try:
                pipe.enable_vae_slicing()
            except AttributeError:
                pass
            # xformers isn't installed in this venv, so this falls back to
            # PyTorch SDPA. Harmless no-op if unavailable.
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("[JuggernautXL] xformers memory-efficient attention enabled")
            except Exception as e:
                print(f"[JuggernautXL] xformers not available, using SDPA: {e}")
            # >=8GB: pin everything to GPU. 6GB: the 4.9GB UNet can't stay
            # resident, so keep CPU offload and rely on torch.compile for speed.
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if total_gb >= 8:
                pipe = pipe.to("cuda")
            else:
                try:
                    pipe.enable_model_cpu_offload()
                except Exception:
                    pipe = pipe.to("cuda")
            # torch.compile the UNet. Needs Triton, absent in the portable env,
            # so guard on it (compile would otherwise crash on first forward).
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
        print(f"[JuggernautXL] Loaded '{node or 'generate'}' on {device}.")

    def _load_clipseg(self, device: str) -> None:
        """Loads CLIPSeg for the 'segment' node (text-guided masking, ~600MB)."""
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

        base = str(self._base_model_dir())
        print(f"[JuggernautXL] Loading CLIPSeg from {base}…")
        processor = CLIPSegProcessor.from_pretrained(base)
        model = CLIPSegForImageSegmentation.from_pretrained(base)
        model.eval()
        model = model.to(device)
        self._model = (processor, model)

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

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def generate(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        # No image wired in? Use the bundled 1x1 placeholder so text-to-image
        # (and REST/MCP callers) never fail on a missing input.
        if not image_bytes:
            image_bytes = _placeholder_image_bytes()

        node = self._node_id()
        if node == "segment":
            return self._generate_segment(image_bytes, params, progress_cb, cancel_event)
        if node in ("edit", "img2img"):
            return self._generate_img2img(image_bytes, params, progress_cb, cancel_event)
        if node == "inpaint":
            return self._generate_inpaint(image_bytes, params, progress_cb, cancel_event)
        return self._generate_text2img(image_bytes, params, progress_cb, cancel_event)

    @staticmethod
    def _num(params: dict, key: str, default: float) -> float:
        value = params.get(key, default)
        if value is None or value == "":
            return float(default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @classmethod
    def _seed(cls, params: dict) -> int:
        seed = int(cls._num(params, "seed", -1))
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        return seed

    @staticmethod
    def _fit_image(image: Image.Image, max_side: int = 1216) -> Image.Image:
        """Resize to a multiple of 8 that SDXL's VAE accepts (<= max_side)."""
        width, height = image.size
        scale = min(1.0, max_side / max(width, height))
        width = max(64, round(width * scale))
        height = max(64, round(height * scale))
        width = max(64, (width // 8) * 8)
        height = max(64, (height // 8) * 8)
        if (width, height) != image.size:
            image = image.resize((width, height), Image.LANCZOS)
        return image

    def _run_progress(self, progress_cb, start, end, label):
        stop_evt = threading.Event()
        if progress_cb:
            t = threading.Thread(
                target=smooth_progress,
                args=(progress_cb, start, end, label, stop_evt),
                daemon=True,
            )
            t.start()
        return stop_evt

    def _generate_segment(self, image_bytes, params, progress_cb=None, cancel_event=None) -> Path:
        """CLIPSeg: turn a text description into a white-on-black mask image."""
        import numpy as np
        import torch

        if self._model is None:
            self.load()
        processor, model = self._model

        region = str(params.get("region", "")).strip()
        if not region:
            raise ValueError("Type what to select (for example 'sky' or 'the bicycle').")
        threshold = min(max(self._num(params, "threshold", 0.5), 0.05), 0.95)
        invert = str(params.get("invert", "no")).lower() in ("yes", "true", "1", "on")
        expand = max(0, int(self._num(params, "expand", 8)))
        feather = max(0, int(self._num(params, "feather", 4)))

        self._report(progress_cb, 10, "Preparing image…")
        self._check_cancelled(cancel_event)

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        self._report(progress_cb, 25, f"Finding '{region}'…")
        stop_evt = self._run_progress(progress_cb, 25, 80, f"Finding '{region}'…")
        try:
            inputs = processor(text=[region], images=[image], padding=True, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
            probs = torch.sigmoid(logits).detach().float().cpu()
            while probs.dim() > 2:
                probs = probs[0]
            prob_img = Image.fromarray((probs.numpy() * 255).astype("uint8"), mode="L")
            prob_map = np.asarray(
                prob_img.resize(image.size, Image.LANCZOS), dtype=np.float32
            ) / 255.0
        finally:
            stop_evt.set()

        self._check_cancelled(cancel_event)

        self._report(progress_cb, 85, "Building mask…")
        mask = prob_map >= threshold
        if invert:
            mask = ~mask
        if expand > 0:
            from scipy import ndimage
            mask = ndimage.binary_dilation(mask, iterations=expand)
        if feather > 0:
            from scipy import ndimage
            soft = ndimage.gaussian_filter(mask.astype(np.float32), sigma=feather)
            out = (np.clip(soft, 0.0, 1.0) * 255).astype("uint8")
        else:
            out = mask.astype("uint8") * 255
        mask_img = Image.fromarray(out, mode="L")

        self.unload()

        self._report(progress_cb, 95, "Saving mask…")
        run = _modly_new_run_folder(self.outputs_dir)
        path = run / "mask.png"
        mask_img.save(str(path), "PNG")

        self._report(progress_cb, 100, "Done")
        return path

    def _generate_text2img(self, image_bytes, params, progress_cb=None, cancel_event=None) -> Path:
        import torch

        if self._model is None:
            self.load()

        prompt = str(params.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("A text prompt is required for generation.")
        negative_prompt = str(params.get("negative_prompt", "") or "").strip()
        num_steps = int(self._num(params, "num_inference_steps", 30))
        guidance_scale = self._num(params, "guidance_scale", 5.0)
        seed = self._seed(params)

        aspect = str(params.get("aspect_ratio", "1024x1024") or "1024x1024").lower()
        try:
            width, height = (int(part) for part in aspect.split("x"))
        except Exception:
            width = height = 1024
        width = max(64, (width // 8) * 8)
        height = max(64, (height // 8) * 8)

        self._report(progress_cb, 5, "Preparing generation…")
        self._check_cancelled(cancel_event)

        self._report(progress_cb, 15, "Generating image…")
        stop_evt = self._run_progress(progress_cb, 15, 90, "Generating image…")
        try:
            generator = torch.Generator(device=self._device).manual_seed(seed)
            result = self._model(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator,
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

    def _generate_img2img(self, image_bytes, params, progress_cb=None, cancel_event=None) -> Path:
        import torch

        if self._model is None:
            self.load()

        prompt = str(params.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("A text prompt is required for the edit.")
        negative_prompt = str(params.get("negative_prompt", "") or "").strip()
        strength = min(max(self._num(params, "strength", 0.7), 0.0), 1.0)
        num_steps = int(self._num(params, "num_inference_steps", 30))
        guidance_scale = self._num(params, "guidance_scale", 5.0)
        seed = self._seed(params)

        self._report(progress_cb, 5, "Preparing image…")
        self._check_cancelled(cancel_event)

        init_image = self._fit_image(Image.open(io.BytesIO(image_bytes)).convert("RGB"))

        self._report(progress_cb, 15, "Editing image…")
        stop_evt = self._run_progress(progress_cb, 15, 90, "Editing image…")
        try:
            generator = torch.Generator(device=self._device).manual_seed(seed)
            result = self._model(
                prompt=prompt,
                image=init_image,
                strength=strength,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt or None,
                generator=generator,
                output_type="pil",
            )
            image = result.images[0]
        finally:
            stop_evt.set()

        self._check_cancelled(cancel_event)
        self.unload()

        self._report(progress_cb, 95, "Saving…")
        run = _modly_new_run_folder(self.outputs_dir)
        path = run / "edited.png"
        image.save(str(path), "PNG")

        self._report(progress_cb, 100, "Done")
        return path

    def _resolve_mask_path(self, params: dict) -> str:
        """Mask may arrive as mask_path, image_path_2, or the 2nd wired image."""
        mask_path = params.get("mask_path") or params.get("image_path_2") or ""
        if not mask_path:
            extra = params.get("extra_image_paths") or []
            if isinstance(extra, (list, tuple)) and extra:
                mask_path = str(extra[0] or "")
        return str(mask_path).strip(' "\'')

    @staticmethod
    def _alpha_mask(image_bytes: bytes, size, background: bool) -> Image.Image:
        """Build a mask from an image's transparency (see-through area)."""
        from PIL import ImageOps

        src = Image.open(io.BytesIO(image_bytes))
        if "A" not in src.getbands():
            raise ValueError(
                "This image has no see-through area. Remove the background first "
                "(Image Editor → Remove Background), or connect a mask."
            )
        alpha = src.getchannel("A").convert("L")
        if alpha.size != size:
            alpha = alpha.resize(size, Image.LANCZOS)
        if background:
            alpha = ImageOps.invert(alpha)
        return alpha

    def _generate_inpaint(self, image_bytes, params, progress_cb=None, cancel_event=None) -> Path:
        import torch

        if self._model is None:
            self.load()

        prompt = str(params.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("A text prompt is required for inpainting.")
        negative_prompt = str(params.get("negative_prompt", "") or "").strip()
        strength = min(max(self._num(params, "strength", 0.9), 0.0), 1.0)
        num_steps = int(self._num(params, "num_inference_steps", 30))
        guidance_scale = self._num(params, "guidance_scale", 5.0)
        seed = self._seed(params)

        self._report(progress_cb, 5, "Preparing image and mask…")
        self._check_cancelled(cancel_event)

        init_image = self._fit_image(Image.open(io.BytesIO(image_bytes)).convert("RGB"))

        mask_source = str(params.get("mask_source", "connected") or "connected").lower()
        if mask_source.startswith("alpha"):
            mask_image = self._alpha_mask(
                image_bytes, init_image.size, background="background" in mask_source
            )
        else:
            mask_path = self._resolve_mask_path(params)
            if not mask_path or not Path(mask_path).exists():
                raise ValueError(
                    "Inpaint needs a mask. Connect a mask to the Mask input, or set "
                    "Mask Source to a transparency option."
                )
            mask_image = Image.open(mask_path).convert("L")
        if mask_image.size != init_image.size:
            mask_image = mask_image.resize(init_image.size, Image.LANCZOS)

        self._report(progress_cb, 15, "Inpainting…")
        stop_evt = self._run_progress(progress_cb, 15, 90, "Inpainting…")
        try:
            generator = torch.Generator(device=self._device).manual_seed(seed)
            result = self._model(
                prompt=prompt,
                image=init_image,
                mask_image=mask_image,
                width=init_image.width,
                height=init_image.height,
                strength=strength,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt or None,
                generator=generator,
                output_type="pil",
            )
            image = result.images[0]
        finally:
            stop_evt.set()

        self._check_cancelled(cancel_event)
        self.unload()

        self._report(progress_cb, 95, "Saving…")
        run = _modly_new_run_folder(self.outputs_dir)
        path = run / "inpainted.png"
        image.save(str(path), "PNG")

        self._report(progress_cb, 100, "Done")
        return path
