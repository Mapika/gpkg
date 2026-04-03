#!/bin/sh
# uvforge installer
# curl -LsSf https://raw.githubusercontent.com/Mapika/uvforge/main/install.sh | sh
set -eu

REPO="https://github.com/Mapika/uvforge"

info() { printf '\033[1;32m%s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m%s\033[0m\n' "$*"; }
fail() { printf '\033[1;31m%s\033[0m\n' "$*"; exit 1; }

if command -v uv >/dev/null 2>&1; then
    info "Installing uvforge via uv tool..."
    uv tool install "uvforge @ git+${REPO}"
elif command -v pipx >/dev/null 2>&1; then
    info "Installing uvforge via pipx..."
    pipx install "git+${REPO}"
elif command -v pip >/dev/null 2>&1; then
    warn "Neither uv nor pipx found, falling back to pip --user"
    pip install --user "git+${REPO}"
else
    fail "No package manager found. Install uv first:
  curl -LsSf https://astral.sh/uv/install.sh | sh"
fi

if command -v uvforge >/dev/null 2>&1; then
    info "Done! Run: uvforge --help"
else
    warn "Installed, but uvforge not on PATH."
    warn "Try:  uv tool run uvforge --help"
    warn "Or add ~/.local/bin to PATH."
fi
