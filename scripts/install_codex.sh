#!/usr/bin/env bash
# Install the Codex CLI as a PACKAGE, not a bare binary.
#
#   scripts/install_codex.sh [VERSION]
#
# The Codex scaffold (instrumental_evasion.runner.scaffolds.codex) requires the
# packaged layout, and preflight refuses a bare binary. GitHub ships two Linux
# artifacts whose names differ by one word: `codex-<target>.tar.gz` is the
# executable alone, while `codex-package-<target>.tar.gz` is the executable PLUS
# the sidecars it resolves relative to itself (bundled ripgrep, the code-mode
# host). With the bare binary Codex still starts and answers `--version`, but its
# file search silently falls back to a system `rg` and Code Mode fails closed --
# a degradation that looks like model behaviour in a transcript. The scaffold
# finds the package by the `codex-package.json` that sits one directory above the
# binary (bin/codex -> ../codex-package.json), which this layout provides.
#
# The layout is a versioned directory plus one symlink on PATH, so several
# versions can sit side by side and a rollback is a re-pointed symlink.
set -euo pipefail

# Codex CLI pinned to the version used in the paper.
VERSION="${1:-0.153.3}"
TARGET="${CODEX_TARGET:-x86_64-unknown-linux-musl}"
PREFIX="${CODEX_PREFIX:-$HOME/.local/share/codex/versions}"
BINDIR="${CODEX_BINDIR:-$HOME/.local/bin}"

DEST="$PREFIX/$VERSION"
URL="https://github.com/openai/codex/releases/download/rust-v${VERSION}/codex-package-${TARGET}.tar.gz"

if [ -x "$DEST/bin/codex" ] && [ -f "$DEST/codex-package.json" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "codex $VERSION already installed at $DEST (FORCE=1 to reinstall)"
else
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    echo "fetching $URL"
    curl -fsSL -o "$tmp/codex-package.tar.gz" "$URL"
    mkdir -p "$DEST"
    tar xzf "$tmp/codex-package.tar.gz" -C "$DEST"
    test -f "$DEST/codex-package.json" || { echo "not a package tarball: $URL" >&2; exit 1; }
    test -x "$DEST/codex-path/rg" || echo "warning: bundled rg missing from the package" >&2
fi

mkdir -p "$BINDIR"
ln -sfn "$DEST/bin/codex" "$BINDIR/codex"
echo "linked $BINDIR/codex -> $DEST/bin/codex"

# The scaffold binds the package ROOT into the container (CODEX_BIN points at the
# symlink; the scaffold resolves it to the package). Print both so a caller can
# export CODEX_BIN and confirm the packaged install.
echo
echo "CODEX_BIN=$BINDIR/codex"
echo "package  =$DEST"
echo
"$BINDIR/codex" --version || true
echo
echo "verify with: codex doctor   (expect 'search ... bundled' and 0 warn)"
