# Juggernaut XL for Modly

Juggernaut XL v9 is a general-purpose SDXL image generator. The extension
supports text-to-image, img2img, and inpainting through one prompt field.

## Install

1. In Modly, open **Extensions** and choose **Install from GitHub**.
2. Paste `https://github.com/iammojogo-sudo/juggernaut-xl_modly`.
3. Wait for the extension setup to finish.
4. Open the Modly model/weights view and download the weights for the
   **Juggernaut XL** node.
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

Add the **Juggernaut XL** node, choose a mode, and enter one prompt:

| Mode | Inputs | Result |
| --- | --- | --- |
| Generate Image | Prompt | New 1024 square image |
| img2img | Source image and prompt | Prompt-guided edit of the source image |
| Inpaint | Source image, mask image, and prompt | Replaces the masked area |

### Prompt

Use the single `Prompt` field for the subject, composition, style, lighting,
camera, or requested edit. Sampler settings, resolution, guidance, and edit
strength use stable internal defaults so the node remains simple and consistent.

For img2img, wire an image into the node and select `img2img`. For inpainting,
wire the source image and a second grayscale mask image into the node. White mask
areas are replaced and black areas are preserved.

## Troubleshooting

- If the node still says **Install** after downloading, reload Modly.
- If generation reports missing weights, download the weights from the Modly
  model view rather than relying on an online Hugging Face fetch.
- For CUDA out-of-memory errors, use a smaller source image for editing or run
  the extension on a machine with more available VRAM.

## Credits

- [Juggernaut XL v9](https://huggingface.co/RunDiffusion/Juggernaut-XL-v9) by RunDiffusion
- SDXL by Stability AI
- iammojogo

## License

MIT
