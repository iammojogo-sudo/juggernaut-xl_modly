# Juggernaut XL for Modly

Juggernaut XL v9 is an SDXL image generator focused on photorealistic output,
character references, and prompt adherence. The extension supports text to
image, img2img, inpainting, and optional 2x or 4x upscaling.

## Install

1. In Modly, open **Extensions** and choose **Install from GitHub**.
2. Paste `https://github.com/iammojogo-sudo/juggernaut-xl_modly`.
3. Wait for the extension setup to finish. `setup.py` creates the isolated
   `venv` and installs PyTorch, Diffusers, Transformers, and the optional
   Real-ESRGAN upscaler.
4. Open the Modly model/weights view and download the weights for the
   **Juggernaut XL** node. Model weights are separate from the Python setup.
5. Wait until the weight entry reports **Installed** before generating.

The model is loaded from Modly's local model directory and runs offline after
the weight download. The completion marker is
`unet/diffusion_pytorch_model.fp16.safetensors`.

## Requirements

- NVIDIA CUDA GPU recommended.
- 6 GB VRAM minimum; 8 GB or more is recommended.
- Extra disk space for the SDXL model and the extension virtual environment.
- macOS can use MPS when supported, but generation is slower than CUDA.

## Use

Add the **Juggernaut XL** node and select one of these modes:

| Mode | Input | Result |
| --- | --- | --- |
| Generate Image | Text prompt | New image at 512, 768, or 1024 square resolution |
| img2img | Source image and prompt | Prompt-guided edit of the source image |
| Inpaint | Source image, mask image, and prompt | Replaces the masked area |

### Generate Image

Use `Prompt` for a normal text-to-image prompt. The optional profile fields
`Physical Profile Details`, `Material Surface Ideas`, `Wear and Tear State`,
`Target Art Style`, and `Color Theme` can be used instead to assemble a full
figure prompt automatically. If any of those profile fields are filled, they
take precedence over the normal `Prompt` field.

Useful controls:

| Control | Description |
| --- | --- |
| Quality Steps | 20, 30, 40, 50, or 100 denoising steps. 30-40 is a good default. |
| Prompt Guidance | CFG strength. 3-7 is a normal range. |
| Output Resolution | 512, 768, or native 1024 square output. |
| Upscale Output | None, 2x, or 4x. Real-ESRGAN is used when available; otherwise a high-quality resize is used. |
| Subject Pose | T-pose, A-pose, neutral standing, or free pose. |
| Camera View | Front, right, back, left, or free camera direction. |
| Seed | Use `-1` for a random seed or set a value for repeatable results. |

### img2img

Wire an image into the node, select `img2img`, enter a prompt, and adjust
`Edit Strength`. Lower values preserve more of the source; higher values allow
larger changes. The normal negative prompt, steps, guidance, and seed controls
also apply.

### Inpaint

Wire the source image and a second grayscale mask image into the node. White
mask areas are replaced and black areas are preserved. A mask is required;
without the second image the node reports an error. `Edit Strength` controls
how strongly the masked region is regenerated.

## Troubleshooting

- If the node still says **Install** after downloading, reload Modly so it
  refreshes the manifest and model status.
- If generation reports missing weights, download the weights from the Modly
  model view rather than relying on an online Hugging Face fetch.
- For CUDA out-of-memory errors, use 512 or 768 resolution, fewer steps, and
  avoid 4x upscaling until the base image has completed.
- If Real-ESRGAN is unavailable, upscaling still works with the built-in
  Lanczos fallback.

## Credits

- [Juggernaut XL v9](https://huggingface.co/RunDiffusion/Juggernaut-XL-v9) by RunDiffusion
- SDXL by Stability AI
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- iammojogo

## License

MIT
