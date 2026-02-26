"""
MoViNet Streaming Violence Detection
=====================================
Model class mapping (from RWF-2000 training):
  index 0 = Fight   → displayed as "Violence"
  index 1 = NonFight → displayed as "NonViolence"

Usage:
    python3 movinet_violence_detection.py \
        --model movinet_model.tflite \
        --video path/to/video.mp4 \
        --output annotated.mp4 \
        --graph --no-display

Flags:
    --output FILE     Save annotated video
    --no-display      Headless mode (no OpenCV window)
    --graph           Scrolling confidence graph panel
    --threshold F     Violence confidence threshold (default: 0.5)
    --skip N          Infer every N frames (default: 1)
"""

import argparse
import collections
import time

import cv2
import numpy as np

try:
    import ai_edge_litert.interpreter as litert
    Interpreter = litert.Interpreter
    print("[INFO] Using ai_edge_litert")
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
        print("[INFO] Using tflite_runtime")
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        print("[INFO] Using TensorFlow Lite (full TF)")

# Model outputs index 0 = Fight, index 1 = NonFight
# We remap to human-readable names for display
VIOLENCE_IDX    = 0   # model class index for violence/fight
NONVIOLENCE_IDX = 1   # model class index for non-violence

DISPLAY_LABELS = {
    VIOLENCE_IDX:    "Violence",
    NONVIOLENCE_IDX: "NonViolence",
}

COLORS = {
    "Violence":    (0,   0, 220),   # red
    "NonViolence": (50, 205,  50),  # green
    "bar_bg":      (40,  40,  40),
    "graph_line":  (255, 200,   0),
    "graph_bg":    (20,  20,  20),
}

GRAPH_HEIGHT   = 120
GRAPH_WIDTH_PX = 600


def softmax(x):
    x = x.astype(np.float64)
    e = np.exp(x - np.max(x))
    return e / e.sum()


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_label_box(frame, label, confidence):
    color = COLORS[label]
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (360, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    cv2.putText(frame, label, (20, 62),
                cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"{confidence*100:.1f}%", (260, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)


def draw_confidence_bars(frame, violence_prob, nonviolence_prob):
    bar_x, bar_y = 10, 105
    bar_w_max, bar_h, gap = 260, 20, 6

    for label, prob in [("Violence", violence_prob), ("NonViolence", nonviolence_prob)]:
        y = bar_y if label == "Violence" else bar_y + bar_h + gap
        color = COLORS[label]
        cv2.rectangle(frame, (bar_x, y),
                      (bar_x + bar_w_max, y + bar_h), COLORS["bar_bg"], -1)
        fill = int(bar_w_max * prob)
        if fill > 0:
            cv2.rectangle(frame, (bar_x, y),
                          (bar_x + fill, y + bar_h), color, -1)
        cv2.putText(frame, f"{label}: {prob*100:.1f}%",
                    (bar_x + bar_w_max + 8, y + bar_h - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)


def draw_hud(frame, frame_idx, total_frames, fps_val):
    h, w = frame.shape[:2]
    txt = f"Frame {frame_idx}/{total_frames}  |  {fps_val:.1f} FPS"
    (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, txt, (w - tw - 10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    bh = 5
    if total_frames > 0:
        fill = int(w * frame_idx / total_frames)
        cv2.rectangle(frame, (0, h - bh), (w, h), (60, 60, 60), -1)
        cv2.rectangle(frame, (0, h - bh), (fill, h), (100, 180, 255), -1)


def build_graph_panel(history, panel_w, panel_h, threshold):
    panel = np.full((panel_h, panel_w, 3), COLORS["graph_bg"], dtype=np.uint8)
    for level in [0.25, 0.5, 0.75]:
        y = int(panel_h - level * panel_h)
        cv2.line(panel, (0, y), (panel_w, y), (60, 60, 60), 1)
        cv2.putText(panel, f"{level:.0%}", (2, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)
    cv2.putText(panel, "Violence confidence (per frame)", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    if len(history) > 1:
        n = len(history)
        pts = [(int(i * panel_w / n),
                max(0, min(panel_h - 1, int(panel_h - v * panel_h))))
               for i, v in enumerate(history)]
        for i in range(1, len(pts)):
            cv2.line(panel, pts[i-1], pts[i], COLORS["graph_line"], 2, cv2.LINE_AA)
    ty = int(panel_h * (1.0 - threshold))
    cv2.line(panel, (0, ty), (panel_w, ty), (180, 80, 80), 1)
    cv2.putText(panel, f"threshold {threshold:.0%}", (panel_w - 130, ty - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 80, 80), 1)
    return panel


# ── Model helpers ─────────────────────────────────────────────────────────────

def find_image_input(interpreter):
    for t in interpreter.get_input_details():
        if "image" in t["name"]:
            return t
    raise RuntimeError("Could not find image input tensor")


def find_logit_output(interpreter):
    for t in interpreter.get_output_details():
        if list(t["shape"]) == [1, 2]:
            return t
    raise RuntimeError("Could not find [1,2] logit output tensor")


def build_state_map(interpreter):
    img_idx   = find_image_input(interpreter)["index"]
    logit_idx = find_logit_output(interpreter)["index"]

    inp_by_idx = {t["index"]: t for t in interpreter.get_input_details()}

    try:
        sig_list = interpreter.get_signature_list()
        sig_key  = list(sig_list.keys())[0]
        sig      = sig_list[sig_key]
        sig_inputs  = sig["inputs"]
        sig_outputs = sig["outputs"]

        out_to_in = {}
        states    = {}
        in_dtypes = {}

        for name, out_idx in sig_outputs.items():
            if out_idx == logit_idx:
                continue
            if name in sig_inputs:
                in_idx = sig_inputs[name]
                t = inp_by_idx[in_idx]
                states[in_idx]     = np.zeros(t["shape"], dtype=t["dtype"])
                in_dtypes[in_idx]  = t["dtype"]
                out_to_in[out_idx] = in_idx

        print(f"[INFO] Mapped {len(out_to_in)} state pairs via model signature ✓")

    except Exception as e:
        print(f"[WARN] Signature mapping failed ({e}), falling back to shape matching")

        from collections import defaultdict
        state_inputs  = [t for t in interpreter.get_input_details()
                         if t["index"] != img_idx]
        state_outputs = [t for t in interpreter.get_output_details()
                         if t["index"] != logit_idx]

        inp_groups = defaultdict(list)
        for t in state_inputs:
            key = (tuple(t["shape"]), np.dtype(t["dtype"]).name)
            inp_groups[key].append(t)

        out_to_in   = {}
        states      = {}
        in_dtypes   = {}
        used_inputs = set()

        for o in state_outputs:
            key = (tuple(o["shape"]), np.dtype(o["dtype"]).name)
            candidates = [t for t in inp_groups.get(key, [])
                          if t["index"] not in used_inputs]
            if not candidates:
                continue
            t = candidates[0]
            used_inputs.add(t["index"])
            states[t["index"]]    = np.zeros(t["shape"], dtype=t["dtype"])
            in_dtypes[t["index"]] = t["dtype"]
            out_to_in[o["index"]] = t["index"]

        print(f"[INFO] Mapped {len(out_to_in)} state pairs via shape matching")

    dtype_counts = {}
    for in_idx in states:
        k = np.dtype(in_dtypes[in_idx]).name
        dtype_counts[k] = dtype_counts.get(k, 0) + 1
    print(f"[INFO] State dtype breakdown: {dtype_counts}")

    return states, out_to_in, in_dtypes


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args):
    print(f"[INFO] Loading model: {args.model}")
    interpreter = Interpreter(model_path=args.model)
    interpreter.allocate_tensors()

    img_inp   = find_image_input(interpreter)
    logit_out = find_logit_output(interpreter)
    states, out_to_in, in_dtypes = build_state_map(interpreter)

    _, _, H, W, _ = img_inp["shape"]
    print(f"[INFO] Image input  : index={img_inp['index']}  shape={img_inp['shape']}")
    print(f"[INFO] Logit output : index={logit_out['index']}  shape={logit_out['shape']}")
    print(f"[INFO] State pairs  : {len(states)}")
    print(f"[INFO] Frame size   : {W}x{H}")

    print(f"[INFO] Opening: {args.video}")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Video: {src_w}x{src_h} @ {src_fps:.1f} FPS, {total_frames} frames\n")

    writer = None
    out_h  = src_h + (GRAPH_HEIGHT if args.graph else 0)
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, src_fps, (src_w, out_h))
        print(f"[INFO] Saving → {args.output}")

    violence_history = collections.deque(maxlen=GRAPH_WIDTH_PX)
    # probs[VIOLENCE_IDX] = violence score, probs[NONVIOLENCE_IDX] = nonviolence score
    current_probs = np.array([0.5, 0.5])
    current_label = "NonViolence"
    frame_idx     = 0
    t_prev        = time.time()
    fps_display   = 0.0

    print(f"{'Frame':>7}  {'Label':<14}  {'Violence':>9}  {'NonViolence':>12}")
    print("─" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % args.skip == 0:
            img = cv2.resize(frame, (W, H))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = (img.astype(np.float32) / 255.0)[np.newaxis, np.newaxis, ...]

            for in_idx, arr in states.items():
                interpreter.set_tensor(in_idx, arr)
            interpreter.set_tensor(img_inp["index"], img)
            interpreter.invoke()

            logits = interpreter.get_tensor(logit_out["index"])[0]
            current_probs = softmax(logits)
            # probs[0] = Violence (Fight), probs[1] = NonViolence (NonFight)
            current_label = DISPLAY_LABELS[int(np.argmax(current_probs))]

            for out_idx, in_idx in out_to_in.items():
                raw = interpreter.get_tensor(out_idx).copy()
                states[in_idx] = raw.astype(in_dtypes[in_idx])

        violence_score    = float(current_probs[VIOLENCE_IDX])
        nonviolence_score = float(current_probs[NONVIOLENCE_IDX])
        violence_history.append(violence_score)

        now = time.time()
        fps_display = 0.9 * fps_display + 0.1 / max(now - t_prev, 1e-6)
        t_prev = now

        if frame_idx % 10 == 0 or frame_idx == 1:
            print(f"{frame_idx:>7}  {current_label:<14}  "
                  f"{violence_score*100:>8.1f}%  "
                  f"{nonviolence_score*100:>11.1f}%")

        annotated = frame.copy()
        draw_label_box(annotated, current_label, violence_score)
        draw_confidence_bars(annotated, violence_score, nonviolence_score)
        draw_hud(annotated, frame_idx, total_frames, fps_display)

        if args.graph:
            panel = build_graph_panel(
                violence_history, src_w, GRAPH_HEIGHT, args.threshold)
            annotated = np.vstack([annotated, panel])

        if not args.no_display:
            cv2.imshow("MoViNet Violence Detection", annotated)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                print("\n[INFO] Quit by user.")
                break

        if writer:
            writer.write(annotated)

    cap.release()
    if writer:
        writer.release()
        print(f"\n[INFO] Saved → {args.output}")
    if not args.no_display:
        cv2.destroyAllWindows()
    print(f"[INFO] Done. {frame_idx}/{total_frames} frames processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      required=True)
    parser.add_argument("--video",      required=True)
    parser.add_argument("--output",     default=None)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--graph",      action="store_true")
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--skip",       type=int,   default=1)
    args = parser.parse_args()
    run(args)
