# Juggernaut XL for Modly

Juggernaut XL v9 makes photorealistic images. This extension adds three nodes:

- **Juggernaut XL** — make a new image from text.
- **Juggernaut XL Restyle** — redraw a whole image from a prompt. It does not
  keep parts unchanged.
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

### Restyle an image

This node redraws the **whole** image from your prompt. It cannot keep parts of
the image unchanged. If you only want to change one area, use Inpaint instead.

1. Add the **Juggernaut XL Restyle** node.
2. Connect your image to it.
3. Type a prompt that describes the whole picture you want.
4. Use **Restyle Strength** to pick how much changes:
   - Low (0.3–0.5) stays closer to the original.
   - High (0.6–0.8) changes more.

Even at low strength the whole image is redrawn, so small details can shift.

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

## FAQ

**Why did my restyle change everything?**
The Restyle node redraws the whole image. It can't keep parts of the image the
same, even if your prompt says to.

**How do I change just one thing, like the sky?**
Use **Inpaint**. Make a mask that is white on the part you want to change and
black everywhere else. Only the white part changes.

**Why can't I say "keep everything the same"?**
The model uses your prompt for the whole picture. It has no way to attach words
to specific spots, so that instruction does nothing.

**How do I remove a background?**
Use the **Image Editor → Remove Background** node. It is made for that, and it
keeps the subject exactly the same.

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
