#!/usr/bin/env bash
set -e

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed."
    echo ""
    echo "Install uv from: https://docs.astral.sh/uv/getting-started/installation/"
    echo ""
    echo "Quick install:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo ""
    exit 1
fi

echo "✅ uv is installed"

# Sync dependencies
echo ""
echo "📦 Syncing dependencies..."
uv sync

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
