# Images

This folder contains the **source images** and the **weight masks** used to produce
the figures of the IPOL article. Every *result* image shown in
the article (reductions, enlargements, removals, and the energy/seam diagnostics) is
generated from these files with the `seam_carving` tool; the exact commands are listed in
[Reproducing the manuscript figures](#reproducing-the-manuscript-figures).

The four photographs are downscaled working copies (960 px wide) of full-resolution
originals from [Wikimedia Commons](https://commons.wikimedia.org/), reproduced here in
compliance with their respective licenses. The masks are the author's own work.

---

## Source images and credits

### `portal_orig.jpg` - Portal of the City Palace, Jaipur (960 × 1167)
- **Author:** Jakub Hałun
- **Source:** [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Portal,_Pitam_Niwas_Chowk,_City_Palace,_Jaipur,_20191218_1000_9059.jpg)
- **Year:** 2019
- **License:** [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

### `birds_orig.png` - Cormorant, egret and gadwall at Taudaha Lake (960 × 640)
- **Author:** Prasan Shrestha
- **Source:** [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Phalacrocorax_carbo,_Egretta_garzetta_and_Mareca_strepera_in_Taudha_Lake.jpg)
- **Year:** 2019
- **License:** [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

### `pont_orig.jpg` - Pont de Bir-Hakeim, Paris (960 × 688)
- **Author:** Daniel Vorndran / DXR
- **Source:** [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Pont_de_Bir-Hakeim_and_view_on_the_16th_Arrondissement_of_Paris_140124_1.jpg)
- **Year:** 2014
- **License:** [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)

### `aurora_orig.jpg` - Aurora over an abandoned weather station, Teriberka (960 × 1200)
- **Author:** Olga Maksimova
- **Source:** [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:%D0%9F%D0%BE%D0%BB%D1%8F%D1%80%D0%BD%D0%BE%D0%B5_%D1%81%D0%B8%D1%8F%D0%BD%D0%B8%D0%B5_%D0%BD%D0%B0%D0%B4_%D0%B7%D0%B0%D0%B1%D1%80%D0%BE%D1%88%D0%B5%D0%BD%D0%BD%D1%8B%D0%BC_%D0%B7%D0%B4%D0%B0%D0%BD%D0%B8%D0%B5%D0%BC_%D0%BC%D0%B5%D1%82%D0%B5%D0%BE%D1%81%D1%82%D0%B0%D0%BD%D1%86%D0%B8%D0%B5%D0%B9.jpg)
- **Year:** 2019
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

---

## Masks

The masks are created by the author for this article (own work). A mask must have the
same dimensions as the image it is applied to.

The tool interprets masks in two ways:

- **Colour mask:** the per-pixel weight is proportional to the difference `G − R`
  between the green and red channels: **green protects** (positive weight, seams avoid
  the region) and **red removes** (negative weight, seams are attracted there first).
- **Greyscale mask:** sufficiently **bright** pixels protect, sufficiently **dark**
  pixels are removed, and intermediate grey leaves the cost unchanged.

| File | Type | Applies to | Role |
|------|------|-----------|------|
| `aurora_mask_both.png` | colour | `aurora_orig.jpg` | green = protect the house, red = remove a patch of aurora |
| `aurora_mask_gray.png`  | greyscale | `aurora_orig.jpg` | greyscale equivalent of `aurora_mask_both.png` |
| `birds_mask.png`        | colour | `birds_orig.png`  | red = remove the duck on the right |
| `birds_mask_gray.png`   | greyscale | `birds_orig.png`  | greyscale equivalent of `birds_mask.png` |

The greyscale masks are provided to document the greyscale mode: each was built from its
colour counterpart by mapping the **sign of `G − R`** (white where green dominates,
black where red dominates, mid-grey `128` elsewhere). They produce the same result as
the colour masks: the `aurora` output is byte-identical, and the `birds` output differs
only imperceptibly (≤ 7/255 on a few pixels) due to floating-point tie-breaking.

---

## Reproducing the manuscript figures

Build the tool first (see the top-level [`README.txt`](../README.txt)); the binary is
then at `../build/seam_carving` (append `.exe` on Windows). The commands below are run
from inside this `images/` folder. Backward energy and BT.601-luminance gradients are
the defaults; `--forward` selects the forward-energy criterion. The visualisation flags
(`--dump-energy`, `--dump-seams`, `--seam-on-gray`) write their figures from the input
*before* carving, so the carved output of those runs is incidental and can be discarded.

**Figure 2: energy map and selected seams (`portal_orig.jpg`)**
```sh
# energy map (viridis) and the 320 seams removed to reach width 640, over greyscale;
# the carved output portal_scratch.png is unused and may be deleted
../build/seam_carving portal_orig.jpg portal_scratch.png -w 640 \
    --dump-energy portal_energy.png \
    --dump-seams  portal_seams.png --seam-on-gray
```

**Figure 3: content-aware resizing (`portal_orig.jpg`)**
```sh
../build/seam_carving portal_orig.jpg portal_enlarge_w1280.png -w 1280
../build/seam_carving portal_orig.jpg portal_backward_w640.png -w 640
../build/seam_carving portal_orig.jpg portal_forward_w640.png  -w 640 --forward
```

**Figure 4: backward vs forward energy (`pont_orig.jpg`)**
```sh
../build/seam_carving pont_orig.jpg pont_backward_w560.png -w 560
../build/seam_carving pont_orig.jpg pont_forward_w560.png  -w 560 --forward
```

**Figure 5: seam selection, backward vs forward (`pont_orig.jpg`)**
```sh
# first 160 seams each criterion would remove (target width 800), over greyscale;
# the carved outputs are unused figures and may be deleted
../build/seam_carving pont_orig.jpg pont_scratch_back.png -w 800 \
    --dump-seams pont_seams_backward.png --seam-on-gray
../build/seam_carving pont_orig.jpg pont_scratch_fwd.png  -w 800 --forward \
    --dump-seams pont_seams_forward.png --seam-on-gray
```

**Figure 6: object removal (`birds_orig.png` + `birds_mask.png`)**
```sh
# carve the width to 700 with the removal mask: the duck is carved away
../build/seam_carving birds_orig.png birds_removed_w700.png -w 700 -m birds_mask.png
# enlarge the result back to the original width 960 by seam insertion
../build/seam_carving birds_removed_w700.png birds_removed_back960.png -w 960
```

**Figure 7: protection and simultaneous removal (`aurora_orig.jpg` + `aurora_mask_both.png`)**
```sh
# height reduced to 600 WITHOUT a mask: the house is squeezed and distorted
../build/seam_carving aurora_orig.jpg aurora_nomask_h600.png -h 600
# same reduction WITH the mask: the house is protected, the aurora patch is removed
../build/seam_carving aurora_orig.jpg aurora_both_h600.png  -h 600 -m aurora_mask_both.png
```

The greyscale masks are drop-in replacements, e.g. `-m aurora_mask_gray.png` or
`-m birds_mask_gray.png`, and yield equivalent results.
