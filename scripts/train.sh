#!/usr/bin/env bash
# shellcheck disable=SC2029  # local vars are intentionally expanded client-side
# Start training on the remote instance, optionally in a detached tmux session.
#
# Env vars (set by the Makefile or the shell):
#   REMOTE      user@host of the instance             (required)
#   PORT        ssh port                               (default 22)
#   REMOTE_DIR  directory containing the deployed code  (default /workspace/server)
#   CONFIG      config file name sent by deploy-config  (basename is used)
#   SESSION     tmux session name                      (default train)
#   DETACH      1/true/yes/on -> detached (tmux -d);   (default 1)
#               anything else -> run in the foreground (needs a TTY)
#   SSH_KEY     optional identity file
set -euo pipefail

REMOTE="${REMOTE:?set REMOTE=user@host}"
PORT="${PORT:-22}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/server}"
CONFIG="${CONFIG:-train_config.toml}"
SESSION="${SESSION:-train}"
DETACH="${DETACH:-1}"
SSH_KEY="${SSH_KEY:-}"

ssh_opts=(-p "$PORT" -o StrictHostKeyChecking=accept-new)
[[ -n "$SSH_KEY" ]] && ssh_opts+=(-i "$SSH_KEY")

config_name="$(basename "$CONFIG")"
train_cmd="uv run --env-file .env python -m cs336_basics.train -c $config_name"

remote_setup="set -e
cd '$REMOTE_DIR'
command -v uv >/dev/null || { echo 'error: uv not found on the instance' >&2; exit 1; }
command -v tmux >/dev/null || { echo 'error: tmux not found on the instance' >&2; exit 1; }
uv sync --frozen"

detach_lc="$(printf '%s' "$DETACH" | tr '[:upper:]' '[:lower:]')"
case "$detach_lc" in
    1 | true | yes | on)
        echo "training (detached): $REMOTE:$REMOTE_DIR ($config_name) -> tmux '$SESSION'"
        ssh "${ssh_opts[@]}" "$REMOTE" "$remote_setup
tmux new-session -d -s '$SESSION' '$train_cmd'"
        echo "started; attach with: ssh -p $PORT $REMOTE -t 'tmux attach -t $SESSION'"
        ;;
    *)
        echo "training (foreground): $REMOTE:$REMOTE_DIR ($config_name) -> tmux '$SESSION'"
        ssh "${ssh_opts[@]}" -t "$REMOTE" "$remote_setup
tmux new-session -s '$SESSION' '$train_cmd'"
        ;;
esac
