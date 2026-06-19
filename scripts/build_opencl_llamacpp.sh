#!/usr/bin/env bash
# Build OpenCL headers + ICD loader + llama.cpp (OpenCL backend) from source.
# All build artifacts go under $DEV_DIR; final binaries are installed to $INSTALL_PREFIX.
# Symlinks to system paths require sudo.
set -euo pipefail

# ---------------------------------------------------------------------------
# Config — override via env vars
# ---------------------------------------------------------------------------
DEV_DIR="${DEV_DIR:-$HOME/dev/llm}"
OPENCL_PREFIX="${OPENCL_PREFIX:-$DEV_DIR/opencl}"
INSTALL_PREFIX="${INSTALL_PREFIX:-/opt/llm}"
OPENCL_LIB="${OPENCL_LIB:-/lib/aarch64-linux-gnu/libOpenCL.so.1.0.0}"

OPENCL_HEADERS_REV="${OPENCL_HEADERS_REV:-5d52989617e7ca7b8bb83d7306525dc9f58cdd46}"
OPENCL_ICD_REV="${OPENCL_ICD_REV:-02134b05bdff750217bf0c4c11a9b13b63957b04}"
LLAMA_CPP_REV="${LLAMA_CPP_REV:-f6da8cb86a28f0319b40d9d2a957a26a7d875f8c}"

JOBS="${JOBS:-$(nproc)}"

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

log()  { echo -e "${CYAN}${BOLD}==>${RESET} ${BOLD}$*${RESET}"; }
ok()   { echo -e "${GREEN}✔${RESET} $*"; }
warn() { echo -e "${YELLOW}⚠${RESET}  $*"; }
die()  { echo -e "${RED}✘${RESET}  $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
clone_or_update() {
    local url="$1" dir="$2" rev="$3"
    if [[ -d "$dir/.git" ]]; then
        warn "$(basename "$dir") already cloned — fetching"
        git -C "$dir" fetch --quiet
    else
        git clone --quiet "$url" "$dir"
    fi
    git -C "$dir" checkout --quiet "$rev"
    ok "$(basename "$dir") @ ${rev:0:12}"
}

cmake_build_install() {
    local src="$1"; shift
    local build_dir="$src/build"
    mkdir -p "$build_dir"
    cmake -S "$src" -B "$build_dir" -G Ninja "$@" 2>&1 | grep -v "^--"
    cmake --build "$build_dir" --target install -- -j"$JOBS"
}

# ---------------------------------------------------------------------------
# 0. System package check
# ---------------------------------------------------------------------------
log "Checking system packages"

REQUIRED_PKGS=(
    git
    cmake
    ninja-build
    pkg-config
    build-essential
    ca-certificates
)

MISSING=()
for pkg in "${REQUIRED_PKGS[@]}"; do
    if ! dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "install ok installed"; then
        MISSING+=("$pkg")
    fi
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
    warn "Missing packages: ${MISSING[*]}"
    echo -e "  Install with: ${CYAN}sudo apt-get install -y ${MISSING[*]}${RESET}"
    read -rp "  Install now? [y/N] " ans
    if [[ "${ans,,}" == y ]]; then
        sudo apt-get install -y "${MISSING[@]}"
    else
        die "Aborted — install missing packages first."
    fi
else
    ok "All required packages present"
fi

# Verify key binaries exist after package check
for cmd in git cmake ninja pkg-config; do
    command -v "$cmd" &>/dev/null || die "Command not found after install: $cmd"
done

mkdir -p "$DEV_DIR" "$INSTALL_PREFIX/bin"
log "Dev dir:         $DEV_DIR"
log "OpenCL prefix:   $OPENCL_PREFIX"
log "Install prefix:  $INSTALL_PREFIX"

# ---------------------------------------------------------------------------
# 1. Symlink libOpenCL.so
# ---------------------------------------------------------------------------
log "Symlinking libOpenCL.so"
if [[ ! -f "$OPENCL_LIB" ]]; then
    die "OpenCL shared library not found: $OPENCL_LIB"
fi
sudo rm -f /usr/lib/libOpenCL.so
sudo ln -s "$OPENCL_LIB" /usr/lib/libOpenCL.so
ok "libOpenCL.so → $OPENCL_LIB"

# ---------------------------------------------------------------------------
# 2. OpenCL Headers
# ---------------------------------------------------------------------------
log "Building OpenCL-Headers"
HEADERS_DIR="$DEV_DIR/OpenCL-Headers"
clone_or_update \
    https://github.com/KhronosGroup/OpenCL-Headers \
    "$HEADERS_DIR" \
    "$OPENCL_HEADERS_REV"

cmake_build_install "$HEADERS_DIR" \
    -DBUILD_TESTING=OFF \
    -DOPENCL_HEADERS_BUILD_TESTING=OFF \
    -DOPENCL_HEADERS_BUILD_CXX_TESTS=OFF \
    -DCMAKE_INSTALL_PREFIX="$OPENCL_PREFIX"
ok "OpenCL headers → $OPENCL_PREFIX"

# ---------------------------------------------------------------------------
# 3. OpenCL ICD Loader
# ---------------------------------------------------------------------------
log "Building OpenCL-ICD-Loader"
ICD_DIR="$DEV_DIR/OpenCL-ICD-Loader"
clone_or_update \
    https://github.com/KhronosGroup/OpenCL-ICD-Loader \
    "$ICD_DIR" \
    "$OPENCL_ICD_REV"

cmake_build_install "$ICD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$OPENCL_PREFIX" \
    -DCMAKE_INSTALL_PREFIX="$OPENCL_PREFIX"
ok "ICD loader → $OPENCL_PREFIX"

# ---------------------------------------------------------------------------
# 4. Symlink CL headers to /usr/include
# ---------------------------------------------------------------------------
log "Symlinking CL headers to /usr/include/CL"
sudo rm -f /usr/include/CL
sudo ln -s "$OPENCL_PREFIX/include/CL" /usr/include/CL
ok "/usr/include/CL → $OPENCL_PREFIX/include/CL"

# ---------------------------------------------------------------------------
# 5. llama.cpp (OpenCL backend)
# ---------------------------------------------------------------------------
log "Building llama.cpp"
LLAMA_DIR="$DEV_DIR/llama.cpp"
clone_or_update \
    https://github.com/ggml-org/llama.cpp \
    "$LLAMA_DIR" \
    "$LLAMA_CPP_REV"

mkdir -p "$LLAMA_DIR/build"
cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_OPENCL=ON 2>&1 | grep -v "^--"
ninja -C "$LLAMA_DIR/build" -j"$JOBS"
ok "llama.cpp built"

# ---------------------------------------------------------------------------
# 6. Install binaries to $INSTALL_PREFIX
# ---------------------------------------------------------------------------
log "Installing binaries to $INSTALL_PREFIX/bin"
BIN_SRC="$LLAMA_DIR/build/bin"
if [[ ! -d "$BIN_SRC" ]]; then
    die "Expected bin dir not found: $BIN_SRC"
fi

sudo mkdir -p "$INSTALL_PREFIX/bin"
sudo cp "$BIN_SRC"/llama-* "$INSTALL_PREFIX/bin/"
ok "Copied $(ls "$BIN_SRC"/llama-* | wc -l) binaries → $INSTALL_PREFIX/bin"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo -e "${GREEN}${BOLD}All done.${RESET}"
echo -e "  Binaries in: ${CYAN}$INSTALL_PREFIX/bin/${RESET}"
if [[ ":$PATH:" != *":$INSTALL_PREFIX/bin:"* ]]; then
    echo -e "  ${YELLOW}Note:${RESET} Add to PATH: export PATH=\"$INSTALL_PREFIX/bin:\$PATH\""
fi
