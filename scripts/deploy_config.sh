#!/usr/bin/env bash
# shellcheck disable=SC2029  # $REMOTE_DIR is intentionally expanded client-side
# Ship the training config and the .env secrets to the remote instance.
#
# Env vars (set by the Makefile or the shell):
#   REMOTE      user@host of the instance           (required)
#   PORT        ssh port                             (default 22)
#   REMOTE_DIR  destination directory on the remote  (default /workspace/server)
#   CONFIG      local config file to send            (default train_config.toml)
#   ENV_FILE    local .env file to send              (default .env)
#   SSH_KEY     optional identity file
set -euo pipefail

REMOTE="${REMOTE:?set REMOTE=user@host}"
PORT="${PORT:-22}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/server}"
CONFIG="${CONFIG:-train_config.toml}"
ENV_FILE="${ENV_FILE:-.env}"
SSH_KEY="${SSH_KEY:-}"

ssh_opts=(-p "$PORT" -o StrictHostKeyChecking=accept-new)
scp_opts=(-P "$PORT" -o StrictHostKeyChecking=accept-new)
if [[ -n "$SSH_KEY" ]]; then
    ssh_opts+=(-i "$SSH_KEY")
    scp_opts+=(-i "$SSH_KEY")
fi

[[ -f "$CONFIG" ]] || { echo "error: config file not found: $CONFIG" >&2; exit 1; }

ssh "${ssh_opts[@]}" "$REMOTE" "mkdir -p '$REMOTE_DIR'"

config_name="$(basename "$CONFIG")"
scp "${scp_opts[@]}" "$CONFIG" "$REMOTE:$REMOTE_DIR/$config_name"
echo "config: $CONFIG -> $REMOTE:$REMOTE_DIR/$config_name"

# Secrets: stream over stdin (never in argv/shell history) and lock down perms.
if [[ -f "$ENV_FILE" ]]; then
    ssh "${ssh_opts[@]}" "$REMOTE" "umask 077; cat > '$REMOTE_DIR/.env'" < "$ENV_FILE"
    echo ".env:   $ENV_FILE -> $REMOTE:$REMOTE_DIR/.env"
else
    echo "warning: $ENV_FILE not found; skipping (wandb/s3 credentials will be missing)" >&2
fi
