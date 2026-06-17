# QCS6490 detector benchmark — 2026-06-11

Slide deck + figures for the model × backend benchmark on the QCS6490 (Hexagon v68).

## Files
- `benchmark_results.csv` — raw results (9 model/backend pairs × 50 images × 3 cycles).
- `make_plots.py` — uv inline-script: prints the verification report + writes `plots/*.png`.
- `slides.md` — Marp deck (embeds `plots/*.png`).
- `plots/` — generated figures.

## Regenerate
```bash
# figures + printed report (speedups, means, FPS):
uv run docs/presentations/update_2026-06-11/make_plots.py

# render the deck:
marp docs/presentations/update_2026-06-11/slides.md -o slides.pdf
# or live preview:
marp docs/presentations/update_2026-06-11/slides.md --preview
```
