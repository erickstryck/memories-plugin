#!/usr/bin/env bash
# The wizard's front door, and the only piece that can run before `qctx` is on PATH.
#
#     bash ~/.hermes/plugins/memories/scripts/install.sh        # installed by hermes
#     bash ~/.claude/plugins/cache/…/<SHA>/scripts/install.sh   # installed by claude
#     ./scripts/install.sh                                      # cloned
#
# It decides NOTHING. Everything past the `exec` is Python, where the offline suite
# reaches it; a decision taken in bash here would be a decision no test could see.
set -euo pipefail

target="${BASH_SOURCE[0]}"
while [ -L "$target" ]; do
  dest="$(readlink "$target")"
  case "$dest" in
    /*) target="$dest" ;;
    *)  target="$(cd -P "$(dirname "$target")" && pwd)/$dest" ;;
  esac
done
root="$(cd -P "$(dirname "$target")/.." && pwd)"

if ! command -v python3 >/dev/null 2>&1; then
  printf 'install: python3 is required and was not found on PATH\n' >&2
  exit 1
fi

if [ ! -f "$root/cli/qctx.py" ]; then
  printf 'install: %s does not look like the plugin tree (no cli/qctx.py)\n' "$root" >&2
  exit 1
fi

exec python3 "$root/cli/qctx.py" install "$@"