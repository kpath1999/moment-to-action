"""
MoViNet Violence Detection - Video File Inference
=================================================
Usage:
    python movinet_violence_detection.py --model movinet_violence.tflite --video input.mp4

Optional flags:
    --output output.mp4         Save annotated video
    --no-display                Don't show live window (useful if no display / headless)
    --clip-len 8                Number of frames per inference window (default: 8)
    --img-size 172              Square frame size fed to the model (default: 172)
    --threshold 0.5             Confidence threshold to flag Violence (default: 0.5)
    --graph                     Show a scrolling confidence graph overlay
"""

import argparse
import collections
import time

import cv2
import numpy as np

# Import order: ai_edge_litert (Rubik Pi 3 / aarch64) → tflite_runtime → full TensorFlow
def softmax(x):
    e = np.exp(x.astype(np.float64) - np.max(x))
    return e / e.sum()

try:
    import ai_edge_litert.interpreter as litert
    Interpreter = litert.Interpreter
    print("[INFO] Using ai_edge_litert (Rubik Pi 3 / LiteRT runtime)")
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
        print("[INFO] Using tflite_runtime")
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        print("[INFO] Using TensorFlow Lite (full TF)")

# ──────────────────────────────────────────────
# Config / labels
# ──────────────────────────────────────────────
LABELS = ["NonViolence", "Violence"]
VIOLENCE_IDX = 1
COLORS = {
    "NonViolence": (50, 205, 50),   # green
    "Violence":    (0,  0,  220),   # red
    "bar_bg":      (40, 40,  40),
    "bar_v":       (0,  0,  220),
    "bar_nv":      (50, 205, 50),
    "graph_line":  (255, 200, 0),
    "graph_bg":    (20,  20,  20),
}
GRAPH_HEIGHT = 120   # pixels for the confidence graph panel
GRAPH_WIDTH  = 400   # number of historical points shown


# ──────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────

def draw_label_box(frame, label, confidence, threshold):
    """Big semi-transparent label + confidence in top-left corner."""
    h, w = frame.shape[:2]
    color = COLORS[label]
    overlay = frame.copy()

    # Background box
    box_h, box_w = 80, 320
    cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    # Label text
    cv2.putText(frame, label, (20, 55),
                cv2.FONT_HERSHEY_DUPLEX, 1.4, color, 2, cv2.LINE_AA)

    # Confidence percentage
    pct = f"{confidence * 100:.1f}%"
    cv2.putText(frame, pct, (230, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)


def draw_confidence_bars(frame, probs):
    """Horizontal confidence bars for each class."""
    bar_x, bar_y = 10, 105
    bar_w_max, bar_h, gap = 260, 18, 6

    for i, (lbl, p) in enumerate(zip(LABELS, probs)):
        y = bar_y + i * (bar_h + gap)
        fill_w = int(bar_w_max * p)
        color = COLORS[lbl]

        # Background
        cv2.rectangle(frame, (bar_x, y),
                      (bar_x + bar_w_max, y + bar_h), COLORS["bar_bg"], -1)
        # Fill
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, y),
                          (bar_x + fill_w, y + bar_h), color, -1)
        # Label
        cv2.putText(frame, f"{lbl}: {p*100:.1f}%",
                    (bar_x + bar_w_max + 8, y + bar_h - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)


def draw_frame_counter(frame, frame_idx, total_frames, fps_val):
    """Bottom-right: frame counter and live FPS."""
    h, w = frame.shape[:2]
    txt = f"Frame {frame_idx}/{total_frames}  |  {fps_val:.1f} FPS"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, txt, (w - tw - 10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)


def draw_timeline_bar(frame, frame_idx, total_frames):
    """Thin progress bar along the very bottom of the frame."""
    h, w = frame.shape[:2]
    bar_h = 5
    if total_frames > 0:
        fill = int(w * frame_idx / total_frames)
        cv2.rectangle(frame, (0, h - bar_h), (w, h), (60, 60, 60), -1)
        cv2.rectangle(frame, (0, h - bar_h), (fill, h), (100, 180, 255), -1)


def build_graph_panel(history, panel_w, panel_h):
    """Return a panel image with a scrolling violence-confidence line graph."""
    panel = np.full((panel_h, panel_w, 3), COLORS["graph_bg"], dtype=np.uint8)

    # Grid lines at 0.25, 0.5, 0.75
    for level in [0.25, 0.5, 0.75]:
        y = int(panel_h - level * panel_h)
        cv2.line(panel, (0, y), (panel_w, y), (60, 60, 60), 1)
        cv2.putText(panel, f"{level:.0%}", (2, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)

    # Title
    cv2.putText(panel, "Violence confidence", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Plot the line
    if len(history) > 1:
        pts = []
        for i, val in enumerate(history):
            x = int(i * panel_w / len(history))
            y = int(panel_h - val * panel_h)
            pts.append((x, y))
        for i in range(1, len(pts)):
            cv2.line(panel, pts[i - 1], pts[i], COLORS["graph_line"], 2, cv2.LINE_AA)

    # Threshold line at 0.5
    thresh_y = int(panel_h * 0.5)
    cv2.line(panel, (0, thresh_y), (panel_w, thresh_y), (180, 80, 80), 1)

    return panel


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def run(args):
    # ── Load model ──────────────────────────────
    print(f"[INFO] Loading model: {args.model}")
    interpreter = Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    inp = input_details[0]
    print(f"[INFO] Model input  shape : {inp['shape']}  dtype: {inp['dtype']}")
    print(f"[INFO] Model output shape : {output_details[0]['shape']}")

    # Determine expected clip length & spatial size from model metadata if possible
    # Otherwise fall back to CLI args
    model_shape = inp['shape']  # e.g. [1, T, H, W, 3]  or  [1, H, W, 3]
    if len(model_shape) == 5:
        _, T, H, W, _ = model_shape
        clip_len  = T if T > 0 else args.clip_len
        img_size  = (W if W > 0 else args.img_size,
                     H if H > 0 else args.img_size)
    else:
        clip_len = args.clip_len
        img_size = (args.img_size, args.img_size)

    print(f"[INFO] Clip length: {clip_len} frames  |  Frame size: {img_size}")

    # ── Open video ──────────────────────────────
    print(f"[INFO] Opening video: {args.video}")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Video: {src_w}x{src_h} @ {src_fps:.1f} FPS, {total_frames} frames")

    # ── Output video writer ──────────────────────
    writer = None
    out_h  = src_h + (GRAPH_HEIGHT if args.graph else 0)
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, src_fps, (src_w, out_h))
        print(f"[INFO] Saving annotated video → {args.output}")

    # ── Inference state ──────────────────────────
    frame_buffer   = collections.deque(maxlen=clip_len)
    violence_history = collections.deque(maxlen=GRAPH_WIDTH)
    current_probs  = np.array([0.5, 0.5])
    current_label  = "NonViolence"

    frame_idx  = 0
    t_prev     = time.time()
    fps_display = 0.0

    print("[INFO] Starting inference — press Q to quit\n")
    print(f"{'Frame':>7}  {'Label':<14}  {'NonViolence':>11}  {'Violence':>9}")
    print("─" * 48)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # ── Preprocess & buffer ──────────────────
        resized    = cv2.resize(frame, img_size)
        rgb        = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        frame_buffer.append(normalized)

        # ── Inference (once buffer is full) ──────
        if len(frame_buffer) == clip_len:
            clip = np.expand_dims(np.array(frame_buffer), axis=0)  # [1,T,H,W,3]

            # Cast to model's expected dtype
            if inp['dtype'] == np.uint8:
                clip = (clip * 255).astype(np.uint8)

            interpreter.set_tensor(inp['index'], clip)
            interpreter.invoke()
            logits = interpreter.get_tensor(output_details[0]['index'])[0]
            current_probs = softmax(logits.astype(np.float64))
            current_label = LABELS[int(np.argmax(current_probs))]

        violence_history.append(float(current_probs[VIOLENCE_IDX]))

        # ── FPS tracking ─────────────────────────
        now = time.time()
        fps_display = 0.9 * fps_display + 0.1 * (1.0 / max(now - t_prev, 1e-6))
        t_prev = now

        # ── Console log (every 10 frames) ────────
        if frame_idx % 10 == 0 or frame_idx == 1:
            print(f"{frame_idx:>7}  {current_label:<14}  "
                  f"{current_probs[0]*100:>10.1f}%  "
                  f"{current_probs[1]*100:>8.1f}%")

        # ── Draw overlays ─────────────────────────
        annotated = frame.copy()
        draw_label_box(annotated, current_label,
                       current_probs[VIOLENCE_IDX], args.threshold)
        draw_confidence_bars(annotated, current_probs)
        draw_frame_counter(annotated, frame_idx, total_frames, fps_display)
        draw_timeline_bar(annotated, frame_idx, total_frames)

        # Optional graph panel
        if args.graph:
            panel = build_graph_panel(violence_history, src_w, GRAPH_HEIGHT)
            annotated = np.vstack([annotated, panel])

        # ── Display ──────────────────────────────
        if not args.no_display:
            cv2.imshow("MoViNet Violence Detection", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("\n[INFO] Quit by user.")
                break

        # ── Write output ──────────────────────────
        if writer:
            writer.write(annotated)

    # ── Cleanup ───────────────────────────────────
    cap.release()
    if writer:
        writer.release()
        print(f"\n[INFO] Saved annotated video → {args.output}")
    if not args.no_display:
        cv2.destroyAllWindows()

    print(f"\n[INFO] Done. Processed {frame_idx} frames.")


# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MoViNet Violence Detection")
    parser.add_argument("--model",      required=True,  help="Path to .tflite model")
    parser.add_argument("--video",      required=True,  help="Path to input video file")
    parser.add_argument("--output",     default=None,   help="Save annotated video to this path")
    parser.add_argument("--no-display", action="store_true", help="Disable live preview window")
    parser.add_argument("--clip-len",   type=int, default=8,   help="Frames per inference clip")
    parser.add_argument("--img-size",   type=int, default=172, help="Square input size for model")
    parser.add_argument("--threshold",  type=float, default=0.5, help="Violence confidence threshold")
    parser.add_argument("--graph",      action="store_true", help="Show scrolling confidence graph")
    args = parser.parse_args()
    run(args)
