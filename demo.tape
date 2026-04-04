# gpkg demo — record with: vhs demo.tape
# Install vhs: https://github.com/charmbracelet/vhs

Output demo.gif

Set Shell "bash"
Set FontSize 16
Set Width 1100
Set Height 550
Set Theme "Catppuccin Mocha"
Set Padding 30
Set TypingSpeed 40ms
Set CursorBlink false

# Warm up deps and set GitHub token off-screen
Hide
Type "export GITHUB_TOKEN=$(gh auth token)"
Enter
Sleep 1s
Type "uv run gpkg --help > /dev/null 2>&1"
Enter
Sleep 5s
Type "clear"
Enter
Sleep 500ms
Show

Sleep 500ms

Type "# What torch/cuda combos are available?"
Enter
Sleep 1.5s

Type "uv run gpkg --available flash-attn causal-conv1d"
Sleep 500ms
Enter
Sleep 6s

Sleep 2s

Type "# Resolve wheels — build missing ones from source"
Enter
Sleep 1.5s

Type "uv run gpkg --torch 2.11 --cuda 128 flash-attn causal-conv1d mamba-ssm --build-missing -o /tmp/pyproject.toml"
Sleep 500ms
Enter
Sleep 8s

Sleep 2s

Type "# Lock for reproducible installs"
Enter
Sleep 1s

Type "uv run gpkg --torch 2.11 --cuda 128 flash-attn causal-conv1d mamba-ssm --build-missing --lock -o /dev/null"
Sleep 500ms
Enter
Sleep 5s

Sleep 2s

Type "# Install from lockfile — no network needed"
Enter
Sleep 1s

Type "uv run gpkg install -o /tmp/pyproject.toml"
Sleep 500ms
Enter
Sleep 3s

Sleep 2s
