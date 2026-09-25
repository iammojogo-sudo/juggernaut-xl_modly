# Juggernaut XL for Modly

Juggernaut XL v9 makes photorealistic images. This extension adds four nodes:

- **Juggernaut XL** — make a new image from text.
- **Juggernaut XL Restyle** — redraw a whole image from a prompt. It does not
  keep parts unchanged.
- **Select by Text** — make a mask by typing what to select, like "sky".
- **Juggernaut XL Inpaint** — replace part of an image using a mask.

The three Juggernaut nodes share one model, so you only download it once. The
**Select by Text** node uses a small extra model.

## Install

1. Open Modly.
2. Go to **Extensions**.
3. Click **Install from GitHub**.
4. Paste this link:
   `https://github.com/iammojogo-sudo/juggernaut-xl_modly`
5. Wait for the setup to finish.
6. Go to the **Models** page.
7. Download **Juggernaut XL**. If you want to make masks by text, also download
   **Select by Text**.
8. Wait until they say **Installed**.

## Requirements

- An NVIDIA GPU is recommended.
- 6 GB of VRAM is the minimum. 8 GB or more is better.
- Free disk space for the models and the extension.
- On a Mac it can use MPS, but it is slower than NVIDIA.

## How to use

### Make an image

1. Add the **Juggernaut XL** node.
2. Type what you want in the **Prompt** box.
3. Click generate.

You don't have to connect anything.

### Restyle an image

This node redraws the **whole** image from your prompt. It cannot keep parts of
the image unchanged. To change only one area, use Select by Text + Inpaint.

1. Add the **Juggernaut XL Restyle** node.
2. Connect your image to it.
3. Type a prompt that describes the whole picture you want.
4. Use **Restyle Strength** to pick how much changes:
   - Low (0.3–0.5) stays closer to the original.
   - High (0.6–0.8) changes more.

### Make a mask with text

No painting needed.

1. Add the **Select by Text** node.
2. Connect your image to it.
3. Type what to select in **What to select**, like `sky` or `the bicycle`.
4. The node outputs a mask image: white where that part is, black elsewhere.

Useful options:

- **Threshold** — higher selects less, lower selects more.
- **Invert** — select everything except the part you typed. Handy for keeping
  a subject and changing the background.
- **Expand** and **Feather** — grow and soften the mask edges.

### Inpaint an image

1. Add the **Juggernaut XL Inpaint** node.
2. Connect your image to the first input, **Image**.
3. Connect a mask to the second input, **Mask**. (Use **Select by Text** to make
   one, or connect any black-and-white image.)
4. Type what should appear in the masked spot.
5. Pick a **Mask Source**:
   - **Connected mask** — use the mask you connected.
   - **Transparency: replace subject** / **replace background** — use the
     see-through area of the image instead of a mask. This works with images
     from **Remove Background**.

In the mask, **white** parts get replaced and **black** parts stay the same.

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
Use **Select by Text** with `sky`, connect its output to the **Mask** input of
**Inpaint**, and prompt for what you want there (for example `red sky`). Only
the white part of the mask changes.

**How do I make the mask myself?**
Any black-and-white image works: white on the part to change, black elsewhere.
Save it as a PNG and connect it to the **Mask** input.

**Why can't I say "keep everything the same"?**
The model uses your prompt for the whole picture. It has no way to attach words
to specific spots, so that instruction does nothing.

**How do I remove a background?**
Use the **Image Editor → Remove Background** node. It keeps the subject exactly
the same.

## Troubleshooting

- Still says **Install** after downloading? Reload Modly.
- Says the weights are missing? Download them from the **Models** page.
- Out of memory? Use a smaller image, or a machine with more VRAM.

## Credits

- [Juggernaut XL v9](https://huggingface.co/RunDiffusion/Juggernaut-XL-v9) by RunDiffusion
- [CLIPSeg](https://huggingface.co/CIDAS/clipseg-rd64-refined) by CIDAS for text-based masking
- SDXL by Stability AI
- iammojogo

## License

MIT
