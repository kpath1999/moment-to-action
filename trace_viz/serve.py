#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "orjson",
# ]
# ///
"""Trace Visualizer — development server.

Usage:
    uv run trace_viz/serve.py                         # serve, user uploads file via UI
    uv run trace_viz/serve.py metrics_report.json     # auto-load this report
    uv run trace_viz/serve.py --port 9000 report.json # custom port + auto-load
    uv run trace_viz/serve.py --no-browser report.json  # don't open browser
"""

# ruff: noqa: T201

from __future__ import annotations

import argparse
import os
import sys
import threading
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import orjson

# Path to the currently loaded report JSON file, if any
_report_path: Path | None = None

# Cache the latest report data to avoid re-reading from disk on every request
_latest_report_data: tuple[float, bytes] | None = None


def _load_report() -> bytes | None:
    """Load and cache the report JSON data, if a report is loaded."""
    if _report_path is None:
        return None

    stat = _report_path.stat()

    global _latest_report_data  # noqa: PLW0603
    if _latest_report_data is not None and stat.st_mtime <= _latest_report_data[0]:
        return _latest_report_data[1]

    with _report_path.open(encoding="utf-8") as f:
        data: bytes = orjson.dumps(orjson.loads(f.read()))  # Re-encode to bytes for serving

    _latest_report_data = (stat.st_mtime, data)
    return data


class Handler(SimpleHTTPRequestHandler):
    """Serves static files from trace_viz/ and handles /api/report."""

    def do_GET(self) -> None:
        """Handle GET requests."""
        if self.path == "/api/report":
            if _report_path is not None:
                data = _load_report()
                if data is not None:
                    self._serve_bytes(data, "application/json; charset=utf-8")
                else:
                    self.send_error(404, "No report loaded - upload a file via the UI")
            else:
                self.send_error(404, "No report loaded - upload a file via the UI")
            return

        if self.path in ("/", ""):
            self.path = "/index.html"

        super().do_GET()

    def _serve_bytes(self, data: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Trace Visualizer — interactive pipeline metrics viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "report",
        nargs="?",
        metavar="REPORT_JSON",
        help="Path to a metrics_report.json to auto-load on startup",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't automatically open the browser",
    )
    args = parser.parse_args()

    if args.report:
        path = Path(args.report).resolve()
        if not path.exists():
            print(f"Error: file not found: {path}")
            sys.exit(1)

        global _report_path  # noqa: PLW0603
        _report_path = path

        print(f"Loaded report: {path}")

    # Serve files from the directory containing this script (trace_viz/)
    os.chdir(Path(__file__).parent)

    server = HTTPServer(("localhost", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"\nTrace Visualizer → {url}")
    print("Press Ctrl+C to stop.\n")

    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
