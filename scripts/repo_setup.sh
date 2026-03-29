#!/usr/bin/env bash
set -e

# Check if uv is installed; install it automatically if not
if ! command -v uv &> /dev/null; then
    echo "📦 uv not found — installing via the official installer..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # The installer places the binary in ~/.local/bin (Linux) or ~/.cargo/bin
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    if ! command -v uv &> /dev/null; then
        echo "❌ uv installation failed. Please install manually:"
        echo "   https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
    echo "✅ uv installed — you may need to restart your shell or run:"
    echo "   source \$HOME/.local/bin/env  # or add ~/.local/bin to PATH"
fi

echo "✅ uv is installed"

# Sync dependencies
echo ""
echo "📦 Syncing dependencies..."
if [[ "$(uname -m)" == "aarch64" ]]; then
    # NVIDIA's Jetson torch wheel has mismatched filename/metadata version fields.
    export UV_SKIP_WHEEL_FILENAME_CHECK=1
    uv sync
else
    uv sync
fi

# Install pre-commit hooks
echo ""
echo "🪝 Installing pre-commit hooks..."
uv run pre-commit install

echo ""
echo "✅ Development environment setup complete!"
echo ""
echo "Next steps:"
echo "  just test          # Run tests"
echo "  just test-all      # Run all tests (including slow integration tests)"
echo "  just lint          # Check code quality"
echo "  just format        # Auto-format code"
