"""
scripts/add_captions.py
Add caption text below each figure PNG.

Usage:
    python scripts/add_captions.py --input-dir output --output-dir output/captioned
"""
import argparse, os
from pathlib import Path

CAPTIONS = {
    "fig1_heatmap.png": (
        "Fig. 1.  Signal landscape heatmap: migraine preventives in adolescent females (FAERS 2004\u20132024). "
        "Color = IC025 (lower Bayesian credible bound); cell labels = ROR where lower 95% CI > 1, n \u2265 3."
    ),
    "fig2_volcano.png": (
        "Fig. 2.  Volcano plot of all drug-event pairs tested. X-axis: log\u2082(ROR); "
        "Y-axis: \u2212log\u2081\u2080(p-value). Point size \u221aN. Labeled points: top BH-significant pairs."
    ),
    "fig3_amplified.png": (
        "Fig. 3.  Adverse events with amplified signals in adolescent females vs. general FAERS population. "
        "IC025 amplification delta = IC025(adolescent female) \u2212 IC025(full FAERS)."
    ),
    "fig4_timeline.png": (
        "Fig. 4.  Annual FAERS report counts by drug for adolescent females with migraine, 2004\u20132024. "
        "Dotted vertical = CGRP mAb FDA approval year (2018)."
    ),
    "fig5_ebgm_ror.png": (
        "Fig. 5.  EBGM vs. ROR for topiramate. Points below the identity line are Bayesian-shrunk "
        "toward null. Point size and color \u221aN."
    ),
}

def add_caption(img_path, caption_text, out_path):
    try:
        from PIL import Image, ImageDraw, ImageFont
        import textwrap

        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        pad = 60
        line_h = 22
        wrapped = textwrap.wrap(caption_text, width=int(W / 8))
        cap_h = len(wrapped) * line_h + pad

        new_img = Image.new("RGB", (W, H + cap_h), "white")
        new_img.paste(img, (0, 0))

        draw = ImageDraw.Draw(new_img)
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/times.ttf", 18)
        except Exception:
            font = ImageFont.load_default()

        y = H + pad // 2
        for line in wrapped:
            draw.text((40, y), line, fill="#222222", font=font)
            y += line_h

        new_img.save(out_path, dpi=(300, 300))
        print(f"  Saved: {out_path}")
    except ImportError:
        print("Pillow not installed. Run: pip install Pillow")
        raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="output")
    parser.add_argument("--output-dir", default="output/captioned")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for fname, caption in CAPTIONS.items():
        src = Path(args.input_dir) / fname
        dst = Path(args.output_dir) / fname
        if src.exists():
            add_caption(str(src), caption, str(dst))
        else:
            print(f"  Not found: {src}")

if __name__ == "__main__":
    main()
