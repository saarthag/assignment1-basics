"""S3-compatible and local access for training inputs and artifacts.

Remote URIs use the ``s3://bucket/key`` form.  Credentials, endpoint and region
are read from the environment so they can be injected via a ``.env`` file:

    AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY   (or a boto3 credential chain)
    AWS_ENDPOINT_URL (or S3_ENDPOINT_URL)       e.g. https://<acct>.r2.cloudflarestorage.com
    AWS_DEFAULT_REGION (or S3_REGION)           defaults to "auto" (Cloudflare R2)

``boto3`` is imported lazily so local-only runs do not require it.
"""

import os
from pathlib import Path
from urllib.parse import urlparse

_NOT_FOUND_CODES = {"404", "NoSuchKey", "NotFound"}
_CLIENT = None


def split_s3(uri: str) -> tuple[str, str]:
    """Split an ``s3://bucket/key`` URI into ``(bucket, key)``."""
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise ValueError(f"not an s3 URI: {uri}")
    if not parsed.netloc:
        raise ValueError(f"s3 URI is missing a bucket: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def get_s3_client():
    """Return a cached boto3 S3 client configured for an S3-compatible endpoint."""
    global _CLIENT
    if _CLIENT is None:
        import boto3
        from botocore.config import Config

        endpoint_url = None
        for key in ("S3_ENDPOINT_URL", "AWS_ENDPOINT_URL_S3", "AWS_ENDPOINT_URL"):
            if value := os.environ.get(key):
                endpoint_url = value
                break
        region = "auto"
        for key in ("S3_REGION", "AWS_DEFAULT_REGION", "AWS_REGION"):
            if value := os.environ.get(key):
                region = value
                break
        config = Config(retries={"max_attempts": 10, "mode": "standard"}, signature_version="s3v4")
        _CLIENT = boto3.client("s3", endpoint_url=endpoint_url, region_name=region, config=config)
    return _CLIENT


def _download_object(client, bucket: str, key: str, dest: Path, size: int | None = None) -> None:
    """Download ``bucket/key`` to ``dest``, skipping when an identical file exists."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if size is None:
        size = client.head_object(Bucket=bucket, Key=key)["ContentLength"]
    if dest.is_file() and dest.stat().st_size == size:
        return
    tmp = dest.with_name(dest.name + ".part")
    client.download_file(bucket, key, str(tmp))
    tmp.replace(dest)


def _under(root: Path, key: str) -> Path:
    """Join ``key`` onto ``root``, refusing paths that escape ``root``."""
    root = root.resolve()
    dest = (root / key.lstrip("/")).resolve()
    if not dest.is_relative_to(root):
        raise ValueError(f"object key escapes the destination root: {key!r}")
    return dest


def ensure_local(uri: str, dest_root: Path, base_dir: Path | None = None) -> Path:
    """Ensure ``uri`` exists locally and return its path, downloading if needed.

    - Local paths are resolved against ``base_dir`` (when relative) and returned
      unchanged.
    - For ``s3://bucket/key`` the object — or every object under the ``key``
      prefix — is downloaded beneath ``dest_root`` while preserving the object
      key layout, so ``s3://b/data/x.txt`` becomes ``<dest_root>/data/x.txt`` and
      ``s3://b/tokenizer`` becomes ``<dest_root>/tokenizer/...``.
    """
    if not isinstance(uri, str) or not uri.startswith("s3://"):
        path = Path(uri)
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        return path.expanduser().resolve()

    bucket, key = split_s3(uri)
    client = get_s3_client()

    try:
        head = client.head_object(Bucket=bucket, Key=key)
    except client.exceptions.ClientError as exc:
        response = getattr(exc, "response", {}) or {}
        code = response.get("Error", {}).get("Code", "")
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code not in _NOT_FOUND_CODES and status != 404:
            raise
    else:
        dest = _under(dest_root, key)
        _download_object(client, bucket, key, dest, size=head["ContentLength"])
        return dest

    prefix = key.rstrip("/")
    paginator = client.get_paginator("list_objects_v2")
    found = False
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/" if prefix else ""):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith("/"):
                continue
            found = True
            _download_object(client, bucket, obj["Key"], _under(dest_root, obj["Key"]), size=obj["Size"])
    if not found:
        raise FileNotFoundError(f"no objects found at {uri}")
    return _under(dest_root, prefix)
