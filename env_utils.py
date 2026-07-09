"""Lightweight .env file loader with indirect variable expansion.

Handles the common patterns:
    KEY=value
    KEY="value with spaces"
    KEY=$OTHER_VAR          # resolved from os.environ at load time
    KEY=${OTHER_VAR}        # same, brace form
    KEY=keyring:SERVICE,USER  # resolved from OS keyring at load time
    # comment line

Existing environment variables are never overwritten — the .env file
only fills in values that are not already set. This means shell exports
always take precedence over the file.
"""
import os
import re
from importlib import import_module

_VAR_REF_RE = re.compile(r'\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?')
_KEYRING_REF_RE = re.compile(r'^keyring:([^,]+),(.+)$')


def _resolve_keyring_reference(raw_value):
    """Resolve keyring:SERVICE,USER references to a secret value.

    If keyring isn't installed/configured, the reference is invalid, or no
    secret exists for the pair, returns raw_value unchanged.
    """
    m = _KEYRING_REF_RE.match(raw_value)
    if not m:
        return raw_value

    service = m.group(1).strip()
    username = m.group(2).strip()
    if not service or not username:
        return raw_value

    try:
        keyring = import_module("keyring")
    except Exception:
        return raw_value

    try:
        secret = keyring.get_password(service, username)
    except Exception:
        return raw_value

    return secret if secret else raw_value


def load_env_file(path=".env"):
    """Load KEY=VALUE pairs from path into os.environ.

    Returns the set of keys that were actually set (useful for logging).
    Silently does nothing if the file does not exist.
    """
    if not os.path.isfile(path):
        return set()

    set_keys = set()
    with open(path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, _, raw_value = line.partition("=")
            key = key.strip()
            if not key:
                continue

            raw_value = raw_value.strip()

            # Strip inline comment (whitespace + # ...) when value is unquoted
            if raw_value and raw_value[0] not in ('"', "'"):
                raw_value = re.sub(r'\s+#.*$', '', raw_value)

            # Strip matching outer quotes
            if (len(raw_value) >= 2
                    and raw_value[0] == raw_value[-1]
                    and raw_value[0] in ('"', "'")):
                raw_value = raw_value[1:-1]

            # Expand $VAR / ${VAR} against the current environment.
            # Unresolvable references are left as-is rather than blanked.
            value = _VAR_REF_RE.sub(
                lambda m: os.environ.get(m.group(1), m.group(0)),
                raw_value,
            )
            value = _resolve_keyring_reference(value)

            if key not in os.environ:
                os.environ[key] = value
                set_keys.add(key)

    return set_keys
