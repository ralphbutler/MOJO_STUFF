#!/bin/bash
# ── macOS local preflight for Mojo ──
# Warns if the project sits in a TCC-protected folder (Desktop/Documents/Downloads),
# where the Mojo REPL gets SIGKILL'd on its first expression with a
# "Library not loaded: libMSupportGlobals.dylib / file system sandbox blocked" wall.
# See MAX_VS_MOJO_GETTING_STARTED.md → "macOS gotcha".
# Run from the project dir:  bash mac_local_check.sh

if [ "$(uname -s)" != "Darwin" ]; then
    echo "Not macOS — TCC folder check not needed. OK."
    exit 0
fi

case "$PWD/" in
    "$HOME"/Desktop/*|"$HOME"/Documents/*|"$HOME"/Downloads/*)
        echo "⚠️  WARNING: this project is under a macOS privacy-protected folder:"
        echo "      $PWD"
        echo "    The Mojo REPL will be KILLED on its first expression here (this is macOS TCC,"
        echo "    not a broken install). Fix with ONE of:"
        echo "      1) Grant iTerm/Terminal Full Disk Access (System Settings > Privacy & Security),"
        echo "         then fully quit (Cmd-Q) and relaunch the terminal."
        echo "      2) Move the project outside Desktop/Documents/Downloads (e.g. ~/Dev/)."
        echo "      3) Skip the REPL; run files with 'mojo file.mojo' (unaffected)."
        exit 1
        ;;
    *)
        echo "Project location OK for the Mojo REPL: $PWD"
        ;;
esac
