SHELL := bash

# Remote target. Override on the command line, e.g.:
#   make deploy REMOTE=root@1.2.3.4 PORT=40022 CONFIG=experiments/lm-v1/train_config.toml
REMOTE     ?= root@HOST
PORT       ?= 22
REMOTE_DIR ?= /workspace/lm-experiments
CONFIG     ?= train_config.toml
ENV_FILE   ?= .env
REV        ?= HEAD
SESSION    ?= train
DETACH     ?= 1
SSH_KEY    ?=

export REMOTE PORT REMOTE_DIR CONFIG ENV_FILE REV SESSION DETACH SSH_KEY

.PHONY: help deploy deploy-code deploy-config train

help:
	@printf '%s\n' \
		'targets:' \
		'  deploy         deploy code + config/.env' \
		"  deploy-code    git archive HEAD -> remote:$$REMOTE_DIR" \
		"  deploy-config  config + .env -> remote:$$REMOTE_DIR" \
		"  train          uv sync --frozen && start training (DETACH=1 detached)"

deploy: deploy-code deploy-config

deploy-code:
	scripts/deploy_code.sh

deploy-config:
	scripts/deploy_config.sh

train:
	scripts/train.sh
