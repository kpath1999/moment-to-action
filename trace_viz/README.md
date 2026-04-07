# Trace Visualizer

Lightweight single-file visualizer for pipeline metrics reports (metrics_report.json).

**Overview**
- Browser-based tool to inspect traces, spans, resource usage and latency.
- No backend required: `trace_viz/index.html` is the UI and `serve.py` provides a tiny local server to serve an initial report.

**Quick start**
1. From the repo root, start the local server (auto-opens browser on supported platforms):

```bash
./trace_viz/serve.py metrics_report.json
# or: ./trace_viz/serve.py                      # serve UI only; load a file via UI
```

2. In the UI you can:
- Drag & drop a `metrics_report.json` file, or use the `Load file…` button in the top toolbar.
- If the server is serving a report at `/api/report`, press the `↻ Server` button to reload it.
- Select a trace from the `Trace:` dropdown in the toolbar.

**Pages / Controls**
- **Overview**: summary cards, cross-trace latency chart, slow-trace alert.
- **Timeline**: Grafana-style collapsible span tree.
  - Zoom: mouse wheel (vertical) or `+` / `-` buttons.
  - Pan: click-and-drag on the bar area or horizontal two-finger swipe.
  - Reset zoom: `↩` button.
  - Expand / Collapse: per-node toggles and `Expand all` / `Collapse all` buttons.
  - Hover a bar to see a tooltip (span name, type, duration, start/end offsets). The label on the colored bar shows the span's reported duration (from `latency_ns`). The bar position/width uses recorded start/end timestamps when available.
- **Resources**: process CPU, device usage, memory, power; a small colored activity strip shows which span was active at each sample (hover to inspect).
- **Latency**: gauge vs budget, top slow spans, sunburst and bar breakdowns.

**Data format (high-level)**
- The UI expects a JSON object with `traces: []` and other metadata like `latency_budget_ms` and `session_id`.
- Each `trace` should contain:
  - `id`: numeric or string id
  - `start`: ISO timestamp (string)
  - `end` (optional): ISO timestamp (string). If missing, `start + latency_ns/1e6` is used.
  - `latency_ns`: duration in nanoseconds (used for numeric computations & hover labels)
  - `spans`: array of spans with fields: `id`, `parent_id` (nullable), `name`, `type`, `start` (ISO), optional `end`, and `latency_ns`.
  - `resource_usage_samples`: array of sample objects { timestamp, proc_cpu_usage, mem_usage: { rss_bytes }, running_span_id, cpu_usage/gpu_usage/npu_usage/dsp_usage (optional) }

Using both `start`/`end` timestamps and `latency_ns` produces the best visual results: timestamps position the bars, `latency_ns` is used for precise duration values shown in labels and other charts.

**Notes & troubleshooting**
- If bars appear to overflow at 100% zoom, refresh the report. The timeline uses the trace `start` as left anchor and extends the right edge to cover span end times; missing `end` fields fall back to `start + latency_ns/1e6`.
- If the server `./trace_viz/serve.py metrics_report.json` reports `No report loaded`, check the file path and permissions.
- The UI performs heavy Plotly rendering on some charts; switching traces can block for a moment — a small "Rendering…" overlay shows progress.

**Development**
- `trace_viz/index.html` is a single-file app (vanilla JS + Plotly). Edit it directly for tweaks.
- `serve.py` is a tiny HTTP server that serves `/api/report` (if started with a filename) and static files from the `trace_viz` dir.

**Want more?**
- Search/filter spans, save view state, or export visible time window as a CSV are natural next improvements.

---
Created for local interactive inspection of `metrics_report.json` files. Feedback welcome.
