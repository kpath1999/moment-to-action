#!/usr/bin/env bash
# Regenerate figures + render the slide deck.
# Requires: uv, marp, and a browser (firefox/chrome) for PDF export.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

echo "==> Generating figures + report"
uv run ./make_plots.py

echo "==> Rendering slides.html"
marp slides.md -o slides.html --allow-local-files

echo "==> Rendering slides.pdf (via firefox)"
marp slides.md -o slides.pdf --allow-local-files --browser firefox

echo "==> Done: $HERE/{slides.html,slides.pdf}"
