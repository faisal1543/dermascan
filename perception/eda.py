"""DERM-8: metadata EDA for HAM10000.

Usage:  python -m perception.eda --metadata data/HAM10000_metadata.csv --out results/eda
Writes PNG figures and summary.md. Uses metadata only (no images), keyed by image_id.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NAMES = {"akiec": "Actinic keratosis (akiec)", "bcc": "Basal cell carcinoma (bcc)",
         "bkl": "Benign keratosis (bkl)", "df": "Dermatofibroma (df)", "mel": "Melanoma (mel)",
         "nv": "Melanocytic nevus (nv)", "vasc": "Vascular lesion (vasc)"}
MALIGNANT = {"akiec", "bcc", "mel"}
BAR, INK, MUTED, GRID = "#3366cc", "#1f2328", "#59636e", "#e6e8eb"


def _style(ax, title):
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def hbar(series, title, xlabel, path, fmt="{:,.0f}", suffix=""):
    s = series.sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(s) + 1.2))
    ax.barh(s.index, s.values, color=BAR, height=0.6)
    for y, v in enumerate(s.values):
        ax.text(v, y, " " + fmt.format(v) + suffix, va="center", fontsize=9, color=INK)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    _style(ax, title)
    ax.set_xlim(0, s.max() * 1.18)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(metadata, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(metadata)
    df["name"] = df.dx.map(NAMES)

    counts = df.dx.value_counts().reindex(CLASSES)
    pct = (counts / len(df) * 100).round(1)
    hbar(df.name.value_counts(), "Images per class: benign moles dominate",
         "Images", out / "class_distribution.png")

    per_lesion = df.groupby("lesion_id").size()
    ipl = per_lesion.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(ipl.index.astype(str), ipl.values, color=BAR, width=0.6)
    for x, v in enumerate(ipl.values):
        ax.text(x, v, f"{v:,}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xlabel("Images of the same lesion", color=MUTED, fontsize=9)
    ax.set_ylabel("Lesions", color=MUTED, fontsize=9)
    _style(ax, "Many lesions have several images, so splits must group by lesion")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out / "images_per_lesion.png", dpi=160)
    plt.close(fig)

    ages = [df.loc[df.dx == c, "age"].dropna() for c in CLASSES]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bp = ax.boxplot(ages, orientation="horizontal", patch_artist=True, widths=0.5,
                    medianprops=dict(color=INK), flierprops=dict(markersize=3, markeredgecolor=MUTED))
    for b in bp["boxes"]:
        b.set(facecolor="#c9d7f2", edgecolor=BAR)
    ax.set_yticks(range(1, len(CLASSES) + 1), CLASSES)
    ax.set_xlabel("Age (years)", color=MUTED, fontsize=9)
    _style(ax, "Age by class: malignant classes skew older")
    fig.tight_layout()
    fig.savefig(out / "age_by_class.png", dpi=160)
    plt.close(fig)

    known_sex = df[df.sex.isin(["male", "female"])]
    female = (known_sex.groupby("dx").sex.apply(lambda s: (s == "female").mean()) * 100).reindex(CLASSES)
    hbar(female, "Share of female patients by class", "% female (known sex only)",
         out / "female_share_by_class.png", fmt="{:.0f}", suffix="%")

    histo = (df.groupby("dx").dx_type.apply(lambda s: (s == "histo").mean()) * 100).reindex(CLASSES)
    hbar(histo, "Biopsy-confirmed labels by class", "% confirmed by histopathology",
         out / "histo_confirmed_by_class.png", fmt="{:.0f}", suffix="%")

    hbar(df.localization.value_counts().head(15), "Body site of lesions (top 15)", "Images",
         out / "body_site.png")

    mal_share = df.dx.isin(MALIGNANT).mean() * 100
    rows = "\n".join(f"| {c} | {NAMES[c]} | {counts[c]:,} | {pct[c]}% | {histo[c]:.0f}% |" for c in CLASSES)
    summary = f"""# HAM10000 metadata EDA (DERM-8)

Source: `{Path(metadata).name}` — {len(df):,} images, {df.lesion_id.nunique():,} lesions.

| Class | Name | Images | Share | Histopathology-confirmed |
|---|---|---:|---:|---:|
{rows}

## Key findings
- **Imbalance:** nv is {pct['nv']}% of images; df ({counts['df']}) and vasc ({counts['vasc']}) are the rarest. Largest/smallest class ratio = {counts.max() / counts.min():.0f}:1.
  A model that always predicts nv scores {pct['nv']}% accuracy, so we report balanced accuracy and per-class recall.
- **Malignant share:** akiec + bcc + mel = {mal_share:.1f}% of images.
- **Several images per lesion:** {(per_lesion > 1).sum():,} of {len(per_lesion):,} lesions ({(per_lesion > 1).mean() * 100:.0f}%) have more than one image (max {per_lesion.max()}). The split must group by `lesion_id` (DERM-10).
- **Label quality:** malignant classes are {', '.join(f'{c} {histo[c]:.0f}%' for c in sorted(MALIGNANT))} biopsy-confirmed; nv only {histo['nv']:.0f}% (the rest by follow-up or consensus), so label noise sits on the benign side.
- **Age:** median {df.age.median():.0f} years overall; {df.age.isna().sum()} missing. Median by class: {", ".join(f"{c} {df.loc[df.dx == c, 'age'].median():.0f}" for c in CLASSES)}.
- **Sex:** {(df.sex == 'male').sum():,} male, {(df.sex == 'female').sum():,} female, {(df.sex == 'unknown').sum()} unknown.
- **Body site:** {df.localization.nunique()} sites; most common: {", ".join(df.localization.value_counts().head(3).index)}. {(df.localization == 'unknown').sum()} unknown.

Figures: class_distribution, images_per_lesion, age_by_class, female_share_by_class,
histo_confirmed_by_class, body_site (PNG, this folder).
"""
    (out / "summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", default="data/HAM10000_metadata.csv")
    p.add_argument("--out", default="results/eda")
    a = p.parse_args()
    main(a.metadata, a.out)
