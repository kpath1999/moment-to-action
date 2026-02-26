#!/usr/bin/env python3
"""
YAMNet + MoViNet Integrated Violence Detection
===============================================
YAMNet monitors audio continuously. When its threat score exceeds
the trigger threshold, MoViNet is activated to analyse the video.

Architecture:
  - YAMNet runs on audio in its own thread (always on)
  - MoViNet runs in a separate thread, gated by a threading.Event
  - When YAMNet fires → MoViNet thread wakes up and processes the video
  - MoViNet result is printed alongside the YAMNet alert

Usage:
    python3 yamnet_movinet_integrated.py \
        --video  fight.mp4 \
        --yamnet-threshold 0.4 \
        --movinet-threshold 0.5

Flags:
    --video FILE              MP4 file to process (audio + video)
    --yamnet-threshold F      Threat score to trigger MoViNet (default: 0.4)
    --movinet-threshold F     Violence confidence threshold (default: 0.5)
    --use-npu                 Use NPU for YAMNet
    --detect-screams          Enable YAMNet scream detection mode
    --skip N                  MoViNet: process every Nth frame (default: 2)
    --debug                   Show full YAMNet group report each chunk
"""

import argparse
import csv
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from ai_edge_litert.interpreter import Interpreter, load_delegate
import sounddevice as sd
import soundfile as sf

from yamnet_class_groups import ClassGrouper, score_threat, print_group_report

# ── YAMNet config ─────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
WINDOW_SIZE = 15600
HOP_SIZE    = 7680
CHANNELS    = 1

# ── MoViNet config ────────────────────────────────────────────────────────────
MOVINET_MODEL  = "movinet/movinet_model.tflite"
VIOLENCE_IDX   = 0   # class 0 = Fight/Violence
NONVIOLENCE_IDX = 1

COLORS = {
    "Violence":    (0,   0, 220),
    "NonViolence": (50, 205,  50),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def softmax(x):
    x = x.astype(np.float64)
    e = np.exp(x - np.max(x))
    return e / e.sum()


def extract_audio(mp4_path):
    wav_path = mp4_path.replace(".mp4", "_audio.wav")
    cmd = [
        "ffmpeg", "-i", mp4_path, "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(SAMPLE_RATE),
        "-ac", str(CHANNELS),
        "-y", wav_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"[YAMNet] ✓ Extracted audio → {wav_path}")
        return wav_path
    except subprocess.CalledProcessError:
        print("[YAMNet] ✗ ffmpeg failed — is ffmpeg installed?")
        return None


# ── MoViNet model helpers ─────────────────────────────────────────────────────

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
    img_idx   = movinet_find_image_input(interp)["index"]
    logit_idx = movinet_find_logit_output(interp)["index"]
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


# ── MoViNet worker thread ─────────────────────────────────────────────────────

class MoViNetWorker:
    """
    Sits idle. When trigger_event is set by YAMNet, processes the full
    video and reports the average Violence confidence.
    """

    def __init__(self, video_path, movinet_threshold, skip, trigger_event, result_queue):
        self.video_path        = video_path
        self.movinet_threshold = movinet_threshold
        self.skip              = skip
        self.trigger_event     = trigger_event   # threading.Event
        self.result_queue      = result_queue    # to send result back
        self.running           = False

        print("[MoViNet] Loading model...")
        self.interp = Interpreter(model_path=MOVINET_MODEL)
        self.interp.allocate_tensors()
        self.img_inp   = movinet_find_image_input(self.interp)
        self.logit_out = movinet_find_logit_output(self.interp)
        self.states, self.out_to_in, self.in_dtypes = movinet_build_state_map(self.interp)
        _, _, H, W, _ = self.img_inp["shape"]
        self.H = H
        self.W = W
        print(f"[MoViNet] ✓ Ready  ({W}x{H} input, {len(self.states)} state pairs)")

    def start(self):
        self.running = True
        self.thread  = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.trigger_event.set()   # unblock the wait so thread can exit

    def _run(self):
        while self.running:
            # Block until YAMNet triggers us (or stop() is called)
            self.trigger_event.wait()
            self.trigger_event.clear()

            if not self.running:
                break

            print("\n[MoViNet] 🎬 Triggered — analysing video...")
            t0 = time.time()

            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                print(f"[MoViNet] ✗ Cannot open video: {self.video_path}")
                continue

            movinet_reset_states(self.states, self.in_dtypes)
            violence_scores = []
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                if frame_idx % self.skip != 0:
                    continue

                img = cv2.resize(frame, (self.W, self.H))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = (img.astype(np.float32) / 255.0)[np.newaxis, np.newaxis, ...]

                for in_idx, arr in self.states.items():
                    self.interp.set_tensor(in_idx, arr)
                self.interp.set_tensor(self.img_inp["index"], img)
                self.interp.invoke()

                logits = self.interp.get_tensor(self.logit_out["index"])[0]
                probs  = softmax(logits)
                violence_scores.append(float(probs[VIOLENCE_IDX]))

                for out_idx, in_idx in self.out_to_in.items():
                    raw = self.interp.get_tensor(out_idx).copy()
                    self.states[in_idx] = raw.astype(self.in_dtypes[in_idx])

            cap.release()
            elapsed = time.time() - t0

            if not violence_scores:
                print("[MoViNet] ✗ No frames processed")
                continue

            avg_violence = float(np.mean(violence_scores))
            label = "Violence" if avg_violence >= self.movinet_threshold else "NonViolence"
            color_code = "🔴" if label == "Violence" else "🟢"

            result = dict(
                avg_violence=avg_violence,
                label=label,
                frames_processed=len(violence_scores),
                elapsed=elapsed,
            )
            self.result_queue.put(result)

            print(f"\n{'='*70}")
            print(f"  MoViNet RESULT")
            print(f"  {color_code} {label}  ({avg_violence*100:.1f}% violence confidence)")
            print(f"  Frames analysed : {len(violence_scores)}")
            print(f"  Time taken      : {elapsed:.1f}s")
            print(f"{'='*70}\n")


# ── YAMNet (adapted from your existing script) ────────────────────────────────

class IntegratedYAMNet:
    """YAMNet with a MoViNet trigger wired in."""

    def __init__(self, args, trigger_event):
        self.args          = args
        self.trigger_event = trigger_event
        self.running       = False
        self.audio_queue   = queue.Queue()
        self.result_queue  = queue.Queue()
        self.audio_buffer  = deque(maxlen=WINDOW_SIZE * 2)

        self.total_inferences = 0
        self.threat_count     = 0
        self.detection_streak = 0
        self.start_time       = None
        self.processing_start_time = None
        self.stream_simulator = None

        # YAMNet trigger cooldown — don't re-trigger MoViNet too fast
        self.last_trigger_time = 0
        self.trigger_cooldown  = args.cooldown

        self.labels  = self._load_labels("models/yamnet/yamnet_class_map.csv")
        self.grouper = ClassGrouper(self.labels)
        self._load_model()

    def _load_labels(self, path):
        labels = {}
        with open(path, "r") as f:
            for row in csv.DictReader(f):
                labels[int(row["index"])] = row["display_name"]
        return labels

    def _load_model(self):
        print("[YAMNet] Loading model...")
        delegates = []
        if self.args.use_npu:
            try:
                delegates = [load_delegate("libQnnTFLiteDelegate.so",
                                           options={"backend_type": "htp"})]
                print("[YAMNet] ✓ NPU delegate loaded")
            except Exception as e:
                print(f"[YAMNet] NPU failed ({e}), using CPU")

        self.interp = Interpreter(
            model_path="models/yamnet/yamnet_quantized.tflite",
            experimental_delegates=delegates
        )
        self.interp.allocate_tensors()
        self.inp_det = self.interp.get_input_details()
        self.out_det = self.interp.get_output_details()

        # Warmup
        dummy = np.zeros(WINDOW_SIZE, dtype=np.float32)
        self.interp.set_tensor(self.inp_det[0]["index"], dummy)
        self.interp.invoke()
        print("[YAMNet] ✓ Ready\n")

    def _run_inference(self, audio):
        self.interp.set_tensor(self.inp_det[0]["index"], audio.astype(np.float32))
        self.interp.invoke()
        return self.interp.get_tensor(self.out_det[0]["index"])

    def _audio_callback(self, indata, frames, time_info, status):
        self.audio_queue.put(indata.copy())

    def _processing_thread(self):
        while self.running:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                self.audio_buffer.extend(chunk.flatten())

                if len(self.audio_buffer) >= WINDOW_SIZE:
                    window    = np.array(list(self.audio_buffer)[:WINDOW_SIZE])
                    output    = self._run_inference(window)
                    scores    = output[0] if len(output.shape) > 1 else output

                    group_scores  = self.grouper.aggregate(scores)
                    threat_result = score_threat(group_scores)
                    threat_score  = threat_result["threat_score"]

                    top5 = [(self.labels[i], float(scores[i]))
                            for i in np.argsort(scores)[-5:][::-1]]

                    # Temporal confirmation
                    if threat_score >= self.args.yamnet_threshold:
                        self.detection_streak += 1
                    else:
                        self.detection_streak = 0

                    is_threat = self.detection_streak >= 2

                    self.total_inferences += 1
                    elapsed = time.time() - self.processing_start_time

                    if is_threat:
                        self.threat_count += 1
                        now = time.time()
                        if now - self.last_trigger_time >= self.trigger_cooldown:
                            self.last_trigger_time = now
                            print(f"\n[YAMNet] 🔴 THREAT DETECTED "
                                  f"(score={threat_score:.3f}) — triggering MoViNet...")
                            self.trigger_event.set()

                    self.result_queue.put(dict(
                        inference_num=self.total_inferences,
                        timestamp=elapsed,
                        threat_score=threat_score,
                        threat_result=threat_result,
                        group_scores=group_scores,
                        is_threat=is_threat,
                        predictions=top5,
                    ))

                    for _ in range(HOP_SIZE):
                        if self.audio_buffer:
                            self.audio_buffer.popleft()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[YAMNet] Processing error: {e}")

    def _display_thread(self):
        while self.running:
            try:
                r = self.result_queue.get(timeout=0.1)
                ts = r["threat_score"]
                bar = "█" * min(30, int(ts * 30))
                print(f"\n{'='*70}")
                print(f"[YAMNet] #{r['inference_num']:4d} | "
                      f"t={r['timestamp']:6.2f}s | "
                      f"threat={ts:.3f} [{bar:<30s}]")

                if self.args.debug:
                    print_group_report(r["group_scores"], r["threat_result"])

                active = {g: s for g, s in r["group_scores"].items()
                          if s >= 0.08 and g not in ("ignore", "unmatched")}
                if active:
                    for g, s in sorted(active.items(), key=lambda x: -x[1]):
                        print(f"         {g:30s} {s:.3f}")

                if r["is_threat"]:
                    print(f"         🔴 ALERT → MoViNet triggered")
                elif ts >= self.args.yamnet_threshold:
                    print(f"         🟡 Building... (streak={self.detection_streak})")

                print("Top 5:")
                for i, (lbl, sc) in enumerate(r["predictions"], 1):
                    print(f"  {i}. {lbl:35s} {sc*100:5.1f}%")

            except queue.Empty:
                continue

    def start(self, audio_file=None):
        self.running = True
        self.start_time = time.time()
        self.processing_start_time = time.time()

        threading.Thread(target=self._processing_thread, daemon=True).start()
        threading.Thread(target=self._display_thread,    daemon=True).start()

        if audio_file:
            # Simulated streaming from file
            audio_data, sr = sf.read(audio_file, dtype="float32")
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            if sr != SAMPLE_RATE:
                from scipy import signal as scipy_signal
                n = int(len(audio_data) * SAMPLE_RATE / sr)
                audio_data = scipy_signal.resample(audio_data, n)

            chunk_size = int(0.1 * SAMPLE_RATE)
            pos = 0
            print(f"[YAMNet] Streaming {len(audio_data)/SAMPLE_RATE:.1f}s of audio...\n")
            try:
                while self.running and pos < len(audio_data):
                    chunk = audio_data[pos:pos + chunk_size]
                    if len(chunk) < chunk_size:
                        chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                    self._audio_callback(chunk.reshape(-1, 1), chunk_size, None, None)
                    pos += chunk_size
                    time.sleep(0.1)
                time.sleep(2)   # let processing finish
            except KeyboardInterrupt:
                pass
        else:
            # Live microphone
            print("[YAMNet] Listening on microphone...\n")
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                                callback=self._audio_callback,
                                blocksize=WINDOW_SIZE // 4):
                try:
                    while self.running:
                        time.sleep(0.1)
                except KeyboardInterrupt:
                    pass

        self.running = False
        print(f"\n[YAMNet] Done. {self.total_inferences} inferences, "
              f"{self.threat_count} threats detected.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="YAMNet + MoViNet Integrated Violence Detection")
    parser.add_argument("--video",               required=True,
                        help="MP4 video file (audio extracted automatically)")
    parser.add_argument("--yamnet-threshold",    type=float, default=0.4,
                        help="YAMNet threat score to trigger MoViNet (default: 0.4)")
    parser.add_argument("--movinet-threshold",   type=float, default=0.5,
                        help="MoViNet violence threshold (default: 0.5)")
    parser.add_argument("--skip",                type=int,   default=2,
                        help="MoViNet: process every Nth frame (default: 2)")
    parser.add_argument("--cooldown",            type=float, default=10.0,
                        help="Seconds before MoViNet can be re-triggered (default: 10)")
    parser.add_argument("--use-npu",             action="store_true")
    parser.add_argument("--detect-screams",      action="store_true")
    parser.add_argument("--debug",               action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"[ERROR] Video not found: {args.video}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print("  YAMNet + MoViNet Integrated Violence Detection")
    print(f"{'='*70}")
    print(f"  Video              : {args.video}")
    print(f"  YAMNet threshold   : {args.yamnet_threshold}")
    print(f"  MoViNet threshold  : {args.movinet_threshold}")
    print(f"  MoViNet frame skip : every {args.skip} frames")
    print(f"  Trigger cooldown   : {args.cooldown}s")
    print(f"{'='*70}\n")

    # Extract audio from MP4
    wav_file = extract_audio(args.video)
    if not wav_file:
        sys.exit(1)

    # Shared trigger mechanism
    trigger_event  = threading.Event()
    movinet_results = queue.Queue()

    # Start MoViNet worker (waits for trigger)
    movinet_worker = MoViNetWorker(
        video_path=args.video,
        movinet_threshold=args.movinet_threshold,
        skip=args.skip,
        trigger_event=trigger_event,
        result_queue=movinet_results,
    )
    movinet_worker.start()

    # Start YAMNet (drives everything)
    yamnet = IntegratedYAMNet(args, trigger_event)
    yamnet.start(audio_file=wav_file)

    # Cleanup
    movinet_worker.stop()

    # Final summary
    print(f"\n{'='*70}")
    print("  FINAL SUMMARY")
    print(f"{'='*70}")
    all_results = []
    while not movinet_results.empty():
        all_results.append(movinet_results.get())

    if all_results:
        for i, r in enumerate(all_results, 1):
            label = r["label"]
            icon  = "🔴" if label == "Violence" else "🟢"
            print(f"  MoViNet trigger {i}: {icon} {label}  "
                  f"({r['avg_violence']*100:.1f}%)  "
                  f"[{r['frames_processed']} frames, {r['elapsed']:.1f}s]")
    else:
        print("  MoViNet was never triggered.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
