#!/usr/bin/env bash
# shellcheck disable=SC2029  # $REMOTE_DIR is intentionally expanded client-side
# Ship the committed source tree to the remote instance.
#
# Uses `git archive`, so only tracked files are sent — git-ignored paths like
# data/, experiments/, .env and __pycache__ never leave the local machine.
#
# Env vars (set by the Makefile or the shell):
#   REMOTE      user@host of the instance          (required)
#   PORT        ssh port                            (default 22)
#   REMOTE_DIR  destination directory on the remote (default /workspace/server)
#   REV         git revision to archive             (default HEAD)
#   SSH_KEY     optional identity file
set -euo pipefail

REMOTE="${REMOTE:?set REMOTE=user@host}"
PORT="${PORT:-22}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/server}"
REV="${REV:-HEAD}"
SSH_KEY="${SSH_KEY:-}"

ssh_opts=(-p "$PORT" -o StrictHostKeyChecking=accept-new)
[[ -n "$SSH_KEY" ]] && ssh_opts+=(-i "$SSH_KEY")

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

echo "code: $(git rev-parse --short "$REV") -> $REMOTE:$REMOTE_DIR"
git archive "$REV" | ssh "${ssh_opts[@]}" "$REMOTE" \
    "mkdir -p '$REMOTE_DIR' && tar -x -C '$REMOTE_DIR'"
echo "code deployed"
