# Juggernaut XL for Modly

Juggernaut XL v9 is a photorealistic SDXL model. The extension exposes three
nodes that all share one set of weights: **Generate** (text to image),
**Edit** (image to image), and **Inpaint** (replace a masked area).

## Install

1. In Modly, open **Extensions** and choose **Install from GitHub**.
2. Paste `https://github.com/iammojogo-sudo/juggernaut-xl_modly`.
3. Wait for the extension setup to finish.
4. Open the Modly model/weights view and download the weights for any of the
   **Juggernaut XL** nodes. The three nodes share the same files, so you only
   download once.
5. Wait until the weight entry reports **Installed** before generating.

The model is loaded from Modly's local model directory and runs offline after
the weight download. The completion marker is
`unet/diffusion_pytorch_model.fp16.safetensors`.

## Requirements

- NVIDIA CUDA GPU recommended.
- 6 GB VRAM minimum; 8 GB or more is recommended.
- Extra disk space for the SDXL model and the extension virtual environment.
- macOS can use MPS when supported, but generation is slower than CUDA.

## Nodes

| Node | Inputs | Result |
| --- | --- | --- |
| Juggernaut XL | Prompt | New image from text |
| Juggernaut XL Edit | Source image and prompt | Prompt-guided edit of the source image |
| Juggernaut XL Inpaint | Source image, mask image, and prompt | Replaces the masked area |

### Generate

Add the **Juggernaut XL** node and enter a prompt. The node has a text input
so you can wire a prompt from another node, but the field on the node works on
its own. Sampler settings follow the model card: DPM++ 2M Karras, 30 steps,
guidance 5, native SDXL resolutions.

### Edit (img2img)

Wire an image into the **Juggernaut XL Edit** node and enter a prompt. Use
**Edit Strength** to control how much changes: low values (0.3-0.5) preserve
the source, higher values (0.6-0.8) allow larger changes.

### Inpaint

Wire the source image into the **first** input and a grayscale **mask** into the
**second** input. White mask areas are replaced and black areas are preserved.
The mask must come from your own image (any image editor that can save a black
and white PNG works). **Inpaint Strength** controls how strongly the masked
region is regenerated; 0.8-1.0 is typical.

If no mask is wired in, the node reports a clear error instead of guessing.

## Troubleshooting

- If a node still says **Install** after downloading, reload Modly.
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
