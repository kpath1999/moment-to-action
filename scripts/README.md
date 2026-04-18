## Two-command workflow

**Step 1 - Mac (MPS), generate pseudo-GT labels:**

```
uv run python scripts/run_coco_eval.py \
    --model oracle \
    --oracle-unit gpu \
    --n-images 500
```

Currently, the terminal command above runs GroundingDINO + SigLIP on MPS.

**Step 2 - Rubik Pi, evaluate edge models against those labels:**

```
uv run python scripts/run_coco_eval.py \
    --model both \
    --skip-oracle \
    --edge-unit npu
```

Currently, we are running YOLO vs GroundingDINO & MobileCLIP vs SigLIP, printing the comparison JSON. Pass `--output results.json` to save it.
