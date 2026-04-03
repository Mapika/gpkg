# uvforge demo — record with: vhs demo.tape
# Install vhs: https://github.com/charmbracelet/vhs

Output demo.gif

Set Shell "bash"
Set FontSize 14
Set Width 900
Set Height 500
Set Theme "Catppuccin Mocha"
Set Padding 20

Type "# Find prebuilt CUDA wheels in seconds"
Enter
Sleep 1s

Type "uvforge --torch 2.11 --cuda 130 flash-attn flash-attn-3 causal-conv1d mamba-ssm"
Enter
Sleep 5s

Type "# See what's available for a package"
Enter
Sleep 500ms

Type "uvforge --available causal-conv1d natten"
Enter
Sleep 4s

Type "# Generate a ready-to-use pyproject.toml"
Enter
Sleep 500ms

Type "uvforge --torch 2.11 --cuda 130 flash-attn causal-conv1d -o pyproject.toml"
Enter
Sleep 4s

Type "cat pyproject.toml"
Enter
Sleep 3s

Type "uv sync  # done!"
Enter
Sleep 2s
