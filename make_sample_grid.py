"""Generate a 2x3 grid of representative dataset samples with their captions."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT  = Path(__file__).resolve().parent

# A hand-picked set: one each of overlapping, mixed, multi-object, ambiguous-ish
PICKS = [
    "img_3063.png",   # 3 objects, no overlap
    "img_5932.png",   # overlap
    "img_11262.png",  # 4 objects
    "img_3016.png",   # spatial spread
    "img_13137.png",  # 5 objects, complex
    "img_8836.png",   # 4 objects, similar colours
]

labels = pd.read_csv(ROOT / "labels.csv").set_index("file_name")

import textwrap

def short_cap(desc: str) -> str:
    a, b = desc.split(" | ")
    # Each clause has form "a SIZE COLOUR SHAPE is REL a SIZE COLOUR SHAPE".
    # We'll keep the full text but wrap to ~38 chars per line.
    a_w = "\n".join(textwrap.wrap(a, width=42))
    b_w = "\n".join(textwrap.wrap(b, width=42))
    return f"{a_w}\n{b_w}"

fig, axes = plt.subplots(2, 3, figsize=(8.5, 7.5))
for ax, fname in zip(axes.flat, PICKS):
    img = Image.open(ROOT / "images" / fname)
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    desc = labels.loc[fname, "description"]
    ax.set_xlabel(short_cap(desc), fontsize=7.5)

fig.suptitle("Example dataset samples with their generated captions", fontsize=11)
fig.tight_layout()
fig.subplots_adjust(hspace=0.55, wspace=0.15, top=0.94)
fig.savefig(OUT / "fig0_samples.png", bbox_inches="tight", dpi=180)
print("[ok] fig0_samples.png")
