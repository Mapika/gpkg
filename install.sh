#!/bin/sh
# gpkg installer
# curl -LsSf https://raw.githubusercontent.com/Mapika/gpkg/main/install.sh | sh
set -eu

REPO="https://github.com/Mapika/gpkg"

info() { printf '\033[1;32m%s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m%s\033[0m\n' "$*"; }
fail() { printf '\033[1;31m%s\033[0m\n' "$*"; exit 1; }

if command -v uv >/dev/null 2>&1; then
    info "Installing gpkg via uv tool..."
    uv tool install "gpkg @ git+${REPO}"
elif command -v pipx >/dev/null 2>&1; then
    info "Installing gpkg via pipx..."
    pipx install "git+${REPO}"
elif command -v pip >/dev/null 2>&1; then
    warn "Neither uv nor pipx found, falling back to pip --user"
    pip install --user "git+${REPO}"
else
    fail "No package manager found. Install uv first:
  curl -LsSf https://astral.sh/uv/install.sh | sh"
fi

if command -v gpkg >/dev/null 2>&1; then
    info "Done! Run: gpkg --help"
else
    warn "Installed, but gpkg not on PATH."
    warn "Try:  uv tool run gpkg --help"
    warn "Or add ~/.local/bin to PATH."
fi
