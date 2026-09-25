# Juggernaut XL for Modly

Juggernaut XL v9 makes photorealistic images. This extension adds three nodes:

- **Juggernaut XL** — make a new image from text.
- **Juggernaut XL Edit** — change an image you already have.
- **Juggernaut XL Inpaint** — replace part of an image.

All three use the same model files. You only download the model once.

## Install

1. Open Modly.
2. Go to **Extensions**.
3. Click **Install from GitHub**.
4. Paste this link:
   `https://github.com/iammojogo-sudo/juggernaut-xl_modly`
5. Wait for the setup to finish.
6. Go to the **Models** page.
7. Download the weights for **Juggernaut XL**.
8. Wait until it says **Installed**.

Now you can use the nodes.

## Requirements

- An NVIDIA GPU is recommended.
- 6 GB of VRAM is the minimum. 8 GB or more is better.
- Free disk space for the model and the extension.
- On a Mac it can use MPS, but it is slower than NVIDIA.

## How to use

### Make an image

1. Add the **Juggernaut XL** node.
2. Type what you want in the **Prompt** box.
3. Click generate.

You don't have to connect anything.

### Edit an image

1. Add the **Juggernaut XL Edit** node.
2. Connect your image to it.
3. Type what to change in the **Prompt** box.
4. Use **Edit Strength** to pick how much changes:
   - Low (0.3–0.5) keeps most of the original.
   - High (0.6–0.8) changes more.

### Inpaint an image

1. Add the **Juggernaut XL Inpaint** node.
2. Connect your image to the first input, **Image**.
3. Connect a mask to the second input, **Mask**.
4. Type what should appear in the masked spot.

A mask is just a black-and-white image:

- **White** parts get replaced.
- **Black** parts stay the same.

Make the mask in any image editor and save it as a PNG. If you don't connect a
mask, the node tells you it needs one.

## Tips

- Keep the default settings at first. They match the model's recommended ones.
- Leave the **Negative Prompt** empty to start. This model does better without
  long negative prompts.
- If you run out of memory, use a smaller image.

## Troubleshooting

- Still says **Install** after downloading? Reload Modly.
- Says the weights are missing? Download them from the **Models** page.
- Out of memory? Use a smaller image, or a machine with more VRAM.

## Credits

- [Juggernaut XL v9](https://huggingface.co/RunDiffusion/Juggernaut-XL-v9) by RunDiffusion
- SDXL by Stability AI
- iammojogo

## License

MIT
