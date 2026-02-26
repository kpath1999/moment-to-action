#!/usr/bin/env python3
"""
YAMNet + MoViNet Real-Time Violence Monitor
============================================
- USB webcam captures live video
- Aux mic captures live audio → YAMNet monitors continuously
- When YAMNet threat score >= threshold, MoViNet analyses
  the last N seconds of buffered frames and reports
- Live annotated window on HDMI + saves to file

Usage:
    python3 realtime_monitor.py

Flags:
    --camera N              Camera device index (default: 0)
    --yamnet-threshold F    Threat score to trigger MoViNet (default: 0.4)
    --movinet-threshold F   Violence confidence threshold (default: 0.5)
    --window-sec N          Seconds of frames MoViNet analyses on trigger (default: 5)
    --cooldown N            Seconds before MoViNet can re-trigger (default: 10)
    --output FILE           Save annotated video (default: output.mp4)
    --no-display            Disable live window (headless/SSH mode)
    --use-npu               Use NPU for YAMNet
    --skip N                MoViNet: analyse every Nth buffered frame (default: 2)
    --mic-device N          Audio input device index (default: system default)
    --list-devices          List available camera and audio devices then exit
"""

import argparse
import csv
import os
import queue
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np
import sounddevice as sd

from ai_edge_litert.interpreter import Interpreter, load_delegate
from yamnet_class_groups import ClassGrouper, score_threat

# ── Constants ─────────────────────────────────────────────────────────────────
SAMPLE_RATE  = 16000
WINDOW_SIZE  = 15600
HOP_SIZE     = 7680
CHANNELS     = 1

MOVINET_MODEL = "movinet/movinet_model.tflite"
YAMNET_MODEL  = "models/yamnet/yamnet_quantized.tflite"
YAMNET_LABELS = "models/yamnet/yamnet_class_map.csv"

VIOLENCE_IDX    = 0
NONVIOLENCE_IDX = 1

COLORS = {
    "Violence":    (0,   0, 220),
    "NonViolence": (50, 205,  50),
    "idle":        (180, 180, 180),
    "alert":       (0,   140, 255),
    "bar_bg":      (40,   40,  40),
    "graph_line":  (255, 200,   0),
    "graph_bg":    (20,   20,  20),
}

GRAPH_HEIGHT = 100


# ── Softmax ───────────────────────────────────────────────────────────────────

def softmax(x):
    x = x.astype(np.float64)
    e = np.exp(x - np.max(x))
    return e / e.sum()


# ── MoViNet helpers ───────────────────────────────────────────────────────────

def movinet_find_image_input(interp):
    for t in interp.get_input_details():
        if "image" in t["name"]:
            return t
    raise RuntimeError("MoViNet image input not found")


def movinet_find_logit_output(interp):
    for t in interp.get_output_details():
        if list(t["shape"]) == [1, 2]:
            return t
    raise RuntimeError("MoViNet [1,2] logit output not found")


def movinet_build_state_map(interp):
    img_idx    = movinet_find_image_input(interp)["index"]
    logit_idx  = movinet_find_logit_output(interp)["index"]
    inp_by_idx = {t["index"]: t for t in interp.get_input_details()}
    try:
        sig    = interp.get_signature_list()
        sig_io = sig[list(sig.keys())[0]]
        sig_in, sig_out = sig_io["inputs"], sig_io["outputs"]
        out_to_in = {}
        states    = {}
        in_dtypes = {}
        for name, out_idx in sig_out.items():
            if out_idx == logit_idx:
                continue
            if name in sig_in:
                in_idx = sig_in[name]
                t = inp_by_idx[in_idx]
                states[in_idx]     = np.zeros(t["shape"], dtype=t["dtype"])
                in_dtypes[in_idx]  = t["dtype"]
                out_to_in[out_idx] = in_idx
    except Exception:
        from collections import defaultdict
        state_inputs  = [t for t in interp.get_input_details()  if t["index"] != img_idx]
        state_outputs = [t for t in interp.get_output_details() if t["index"] != logit_idx]
        groups = defaultdict(list)
        for t in state_inputs:
            groups[(tuple(t["shape"]), np.dtype(t["dtype"]).name)].append(t)
        out_to_in = {}
        states    = {}
        in_dtypes = {}
        used      = set()
        for o in state_outputs:
            key = (tuple(o["shape"]), np.dtype(o["dtype"]).name)
            candidates = [t for t in groups.get(key, []) if t["index"] not in used]
            if not candidates:
                continue
            t = candidates[0]
            used.add(t["index"])
            states[t["index"]]    = np.zeros(t["shape"], dtype=t["dtype"])
            in_dtypes[t["index"]] = t["dtype"]
            out_to_in[o["index"]] = t["index"]
    return states, out_to_in, in_dtypes


def movinet_reset_states(states, in_dtypes):
    for in_idx in states:
        states[in_idx] = np.zeros(states[in_idx].shape, dtype=in_dtypes[in_idx])


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_status_bar(frame, yamnet_threat, movinet_label, movinet_conf,
                    movinet_running, trigger_count):
    """Top overlay: YAMNet threat bar + MoViNet result."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # YAMNet threat bar
    bar_w   = int((w - 20) * min(yamnet_threat, 1.0))
    bar_col = (0, 0, 220) if yamnet_threat >= 0.4 else (0, 140, 255) \
              if yamnet_threat >= 0.2 else (50, 205, 50)
    cv2.rectangle(frame, (10, 10), (w - 10, 30), COLORS["bar_bg"], -1)
    if bar_w > 0:
        cv2.rectangle(frame, (10, 10), (10 + bar_w, 30), bar_col, -1)
    cv2.putText(frame, f"YAMNet threat: {yamnet_threat*100:.1f}%",
                (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    # MoViNet status
    if movinet_running:
        status_txt = "MoViNet: ANALYSING..."
        status_col = COLORS["alert"]
    elif movinet_label:
        conf_txt   = f"{movinet_conf*100:.1f}%"
        status_txt = f"MoViNet: {movinet_label}  {conf_txt}"
        status_col = COLORS.get(movinet_label, COLORS["idle"])
    else:
        status_txt = "MoViNet: idle"
        status_col = COLORS["idle"]

    cv2.putText(frame, status_txt, (10, 60),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, status_col, 2, cv2.LINE_AA)

    cv2.putText(frame, f"Triggers: {trigger_count}",
                (w - 130, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)


def draw_timestamp(frame, elapsed):
    h, w = frame.shape[:2]
    mins = int(elapsed) // 60
    secs = int(elapsed) % 60
    cv2.putText(frame, f"{mins:02d}:{secs:02d}",
                (w - 70, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)


def build_threat_graph(history, panel_w, panel_h, threshold):
    panel = np.full((panel_h, panel_w, 3), COLORS["graph_bg"], dtype=np.uint8)
    for level in [0.25, 0.5, 0.75]:
        y = int(panel_h - level * panel_h)
        cv2.line(panel, (0, y), (panel_w, y), (60, 60, 60), 1)
    cv2.putText(panel, "YAMNet threat score (live)", (4, 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    if len(history) > 1:
        n   = len(history)
        pts = [(int(i * panel_w / n),
                max(0, min(panel_h - 1, int(panel_h - v * panel_h))))
               for i, v in enumerate(history)]
        for i in range(1, len(pts)):
            cv2.line(panel, pts[i-1], pts[i], COLORS["graph_line"], 2, cv2.LINE_AA)
    ty = int(panel_h * (1.0 - threshold))
    cv2.line(panel, (0, ty), (panel_w, ty), (0, 0, 220), 1)
    cv2.putText(panel, f"trigger {threshold:.0%}",
                (panel_w - 90, ty - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 0, 220), 1)
    return panel


# ── Main monitor ──────────────────────────────────────────────────────────────

class RealtimeMonitor:

    def __init__(self, args):
        self.args = args

        # ── Shared state ──────────────────────────────────────────────────────
        self.lock             = threading.Lock()
        self.running          = True

        self.yamnet_threat    = 0.0          # latest YAMNet threat score
        self.detection_streak = 0

        self.movinet_label    = None         # last MoViNet result
        self.movinet_conf     = 0.0
        self.movinet_running  = False
        self.trigger_count    = 0
        self.last_trigger_t   = 0.0

        self.threat_history   = deque(maxlen=300)   # for graph

        # ── Frame ring buffer (for MoViNet to consume on trigger) ─────────────
        fps_estimate = 30
        max_frames   = int(fps_estimate * args.window_sec * 2)  # 2× safety margin
        self.frame_buffer = deque(maxlen=max_frames)
        self.frame_lock   = threading.Lock()
        self.frame_fps    = fps_estimate

        # ── Queues ────────────────────────────────────────────────────────────
        self.audio_queue   = deque(maxlen=WINDOW_SIZE * 4)
        self.audio_lock    = threading.Lock()
        self.movinet_queue = queue.Queue()   # trigger payloads

        # ── Load models ───────────────────────────────────────────────────────
        self._load_yamnet()
        self._load_movinet()

        self.start_time = time.time()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_yamnet(self):
        print("[YAMNet] Loading...")
        delegates = []
        if self.args.use_npu:
            try:
                delegates = [load_delegate("libQnnTFLiteDelegate.so",
                                           options={"backend_type": "htp"})]
                print("[YAMNet] ✓ NPU delegate loaded")
            except Exception as e:
                print(f"[YAMNet] NPU failed ({e}), using CPU")

        self.yamnet_interp = Interpreter(
            model_path=YAMNET_MODEL,
            experimental_delegates=delegates
        )
        self.yamnet_interp.allocate_tensors()
        self.yamnet_inp = self.yamnet_interp.get_input_details()
        self.yamnet_out = self.yamnet_interp.get_output_details()

        labels = {}
        with open(YAMNET_LABELS) as f:
            for row in csv.DictReader(f):
                labels[int(row["index"])] = row["display_name"]
        self.yamnet_labels = labels
        self.grouper = ClassGrouper(labels)

        # Warmup
        dummy = np.zeros(WINDOW_SIZE, dtype=np.float32)
        self.yamnet_interp.set_tensor(self.yamnet_inp[0]["index"], dummy)
        self.yamnet_interp.invoke()
        print("[YAMNet] ✓ Ready")

    def _load_movinet(self):
        print("[MoViNet] Loading...")
        self.movinet_interp = Interpreter(model_path=MOVINET_MODEL)
        self.movinet_interp.allocate_tensors()
        self.movinet_img_inp   = movinet_find_image_input(self.movinet_interp)
        self.movinet_logit_out = movinet_find_logit_output(self.movinet_interp)
        self.movinet_states, self.movinet_out_to_in, self.movinet_in_dtypes = \
            movinet_build_state_map(self.movinet_interp)
        _, _, H, W, _ = self.movinet_img_inp["shape"]
        self.movinet_H = H
        self.movinet_W = W
        print(f"[MoViNet] ✓ Ready  ({W}x{H})")

    # ── Threads ───────────────────────────────────────────────────────────────

    def _audio_callback(self, indata, frames, time_info, status):
        with self.audio_lock:
            self.audio_queue.extend(indata.flatten())

    def _yamnet_thread(self):
        """Continuously process audio and compute threat score."""
        print("[YAMNet] Audio thread started")
        audio_buf = deque(maxlen=WINDOW_SIZE * 2)

        while self.running:
            with self.audio_lock:
                new_samples = list(self.audio_queue)
                self.audio_queue.clear()

            audio_buf.extend(new_samples)

            if len(audio_buf) >= WINDOW_SIZE:
                window = np.array(list(audio_buf)[:WINDOW_SIZE], dtype=np.float32)

                self.yamnet_interp.set_tensor(self.yamnet_inp[0]["index"], window)
                self.yamnet_interp.invoke()
                scores = self.yamnet_interp.get_tensor(self.yamnet_out[0]["index"])
                if len(scores.shape) > 1:
                    scores = scores[0] if scores.shape[0] == 1 \
                             else np.mean(scores, axis=0)

                group_scores  = self.grouper.aggregate(scores)
                threat_result = score_threat(group_scores)
                threat_score  = threat_result["threat_score"]

                with self.lock:
                    self.yamnet_threat = threat_score
                    self.threat_history.append(threat_score)

                    if threat_score >= self.args.yamnet_threshold:
                        self.detection_streak += 1
                    else:
                        self.detection_streak = 0

                    should_trigger = (
                        self.detection_streak >= 2 and
                        not self.movinet_running and
                        (time.time() - self.last_trigger_t) >= self.args.cooldown
                    )

                    if should_trigger:
                        self.last_trigger_t = time.time()
                        self.trigger_count += 1
                        # Snapshot the current frame buffer for MoViNet
                        with self.frame_lock:
                            frames_snapshot = list(self.frame_buffer)
                        self.movinet_queue.put(frames_snapshot)
                        print(f"\n[YAMNet] 🔴 THREAT {threat_score:.3f} "
                              f"— triggering MoViNet #{self.trigger_count}")

                # Advance buffer by hop size
                for _ in range(min(HOP_SIZE, len(audio_buf))):
                    audio_buf.popleft()
            else:
                time.sleep(0.02)

    def _movinet_thread(self):
        """Wait for trigger, run MoViNet on buffered frames, report."""
        print("[MoViNet] Worker thread started")
        H, W = self.movinet_H, self.movinet_W
        target_frames = int(self.frame_fps * self.args.window_sec)

        while self.running:
            try:
                frames_snapshot = self.movinet_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            with self.lock:
                self.movinet_running = True

            # Take the most recent `window_sec` worth of frames
            frames_to_use = frames_snapshot[-target_frames:]
            if not frames_to_use:
                with self.lock:
                    self.movinet_running = False
                continue

            print(f"[MoViNet] Analysing {len(frames_to_use)} frames "
                  f"({self.args.window_sec}s window)...")
            t0 = time.time()

            movinet_reset_states(self.movinet_states, self.movinet_in_dtypes)
            violence_scores = []

            for i, frame in enumerate(frames_to_use):
                if i % self.args.skip != 0:
                    continue
                img = cv2.resize(frame, (W, H))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = (img.astype(np.float32) / 255.0)[np.newaxis, np.newaxis, ...]

                for in_idx, arr in self.movinet_states.items():
                    self.movinet_interp.set_tensor(in_idx, arr)
                self.movinet_interp.set_tensor(self.movinet_img_inp["index"], img)
                self.movinet_interp.invoke()

                logits = self.movinet_interp.get_tensor(
                    self.movinet_logit_out["index"])[0]
                probs  = softmax(logits)
                violence_scores.append(float(probs[VIOLENCE_IDX]))

                for out_idx, in_idx in self.movinet_out_to_in.items():
                    raw = self.movinet_interp.get_tensor(out_idx).copy()
                    self.movinet_states[in_idx] = \
                        raw.astype(self.movinet_in_dtypes[in_idx])

            elapsed = time.time() - t0

            if violence_scores:
                avg_v  = float(np.mean(violence_scores))
                label  = "Violence" if avg_v >= self.args.movinet_threshold \
                         else "NonViolence"
                icon   = "🔴" if label == "Violence" else "🟢"
                print(f"\n[MoViNet] {icon} {label}  "
                      f"({avg_v*100:.1f}% violence)  "
                      f"[{len(violence_scores)} frames, {elapsed:.1f}s]\n")
            else:
                avg_v = 0.0
                label = "NonViolence"

            with self.lock:
                self.movinet_label   = label
                self.movinet_conf    = avg_v
                self.movinet_running = False

    def _camera_thread(self, cap):
        """Continuously read frames from webcam into the ring buffer."""
        print("[Camera] Capture thread started")
        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self.frame_lock:
                self.frame_buffer.append(frame.copy())

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        # Open camera
        print(f"\n[Camera] Opening device {self.args.camera}...")
        cap = cv2.VideoCapture(self.args.camera)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera {self.args.camera}")
            sys.exit(1)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS,           30)
        self.frame_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Camera] ✓ {src_w}x{src_h} @ {self.frame_fps:.0f} FPS")

        # Video writer
        writer     = None
        out_h      = src_h + GRAPH_HEIGHT
        output_path = self.args.output
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, self.frame_fps,
                                 (src_w, out_h))
        print(f"[Output] Saving → {output_path}")

        # Start background threads
        threading.Thread(target=self._camera_thread,
                         args=(cap,), daemon=True).start()
        threading.Thread(target=self._yamnet_thread,  daemon=True).start()
        threading.Thread(target=self._movinet_thread, daemon=True).start()

        # Start audio stream
        mic_kwargs = dict(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            callback=self._audio_callback,
            blocksize=WINDOW_SIZE // 4,
            dtype="float32",
        )
        if self.args.mic_device is not None:
            mic_kwargs["device"] = self.args.mic_device

        print(f"\n[System] All threads running. Press Q or Ctrl+C to stop.\n")
        print(f"  YAMNet trigger threshold : {self.args.yamnet_threshold}")
        print(f"  MoViNet window           : {self.args.window_sec}s")
        print(f"  Cooldown                 : {self.args.cooldown}s\n")

        self.start_time = time.time()

        try:
            with sd.InputStream(**mic_kwargs):
                while self.running:
                    # Grab latest frame for display
                    with self.frame_lock:
                        if not self.frame_buffer:
                            time.sleep(0.01)
                            continue
                        display_frame = self.frame_buffer[-1].copy()

                    # Read shared state
                    with self.lock:
                        threat    = self.yamnet_threat
                        mv_label  = self.movinet_label
                        mv_conf   = self.movinet_conf
                        mv_run    = self.movinet_running
                        trig_cnt  = self.trigger_count
                        history   = list(self.threat_history)

                    # Annotate
                    elapsed = time.time() - self.start_time
                    draw_status_bar(display_frame, threat,
                                    mv_label, mv_conf, mv_run, trig_cnt)
                    draw_timestamp(display_frame, elapsed)

                    graph = build_threat_graph(
                        history, src_w, GRAPH_HEIGHT,
                        self.args.yamnet_threshold)
                    combined = np.vstack([display_frame, graph])

                    if not self.args.no_display:
                        cv2.imshow("Violence Monitor", combined)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), 27):
                            print("\n[System] Quit by user.")
                            break

                    if writer:
                        writer.write(combined)

        except KeyboardInterrupt:
            print("\n[System] Stopped.")
        finally:
            self.running = False
            cap.release()
            if writer:
                writer.release()
                print(f"[Output] Saved → {output_path}")
            if not self.args.no_display:
                cv2.destroyAllWindows()

            elapsed = time.time() - self.start_time
            print(f"\n{'='*60}")
            print(f"  SESSION SUMMARY")
            print(f"{'='*60}")
            print(f"  Duration         : {elapsed:.1f}s")
            print(f"  MoViNet triggers : {self.trigger_count}")
            if self.movinet_label:
                print(f"  Last result      : {self.movinet_label} "
                      f"({self.movinet_conf*100:.1f}%)")
            print(f"{'='*60}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def list_devices():
    print("\n── Camera devices ──────────────────────────────")
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  [{i}] {w}x{h}")
            cap.release()
    print("\n── Audio input devices ─────────────────────────")
    print(sd.query_devices())
    print()


def main():
    parser = argparse.ArgumentParser(
        description="YAMNet + MoViNet Real-Time Violence Monitor")
    parser.add_argument("--camera",             type=int,   default=0)
    parser.add_argument("--yamnet-threshold",   type=float, default=0.4)
    parser.add_argument("--movinet-threshold",  type=float, default=0.5)
    parser.add_argument("--window-sec",         type=int,   default=5)
    parser.add_argument("--cooldown",           type=float, default=10.0)
    parser.add_argument("--output",             default="output.mp4")
    parser.add_argument("--no-display",         action="store_true")
    parser.add_argument("--use-npu",            action="store_true")
    parser.add_argument("--skip",               type=int,   default=2)
    parser.add_argument("--mic-device",         type=int,   default=None)
    parser.add_argument("--list-devices",       action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    monitor = RealtimeMonitor(args)
    monitor.run()


if __name__ == "__main__":
    main()
