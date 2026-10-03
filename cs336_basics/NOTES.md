## Training pipeline
- dataset, valid_dataset, tokenizer can be s3 (or s3-compatible) URL strings
- load the scripts via git archive HEAD
- load the config for the experiment via scp/rsync
- credentials for wandb and r2 will be provided in a .env file, transferred over scp/rsync
- config should support notes (multi line markdown string) and tags (list) for wandb
- if a run id is given in the config, use it. else create one and use it across the training script
- only take config as an input, provision a work directory (\_work\_<run-id>) in tmp to publish training artifacts
- if URL for an upload bucket is given, upload training artifacts and checkpoints to it
