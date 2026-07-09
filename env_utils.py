"""Lightweight .env file loader with indirect variable expansion.

Handles the common patterns:
    KEY=value
    KEY="value with spaces"
    KEY=$OTHER_VAR          # resolved from os.environ at load time
    KEY=${OTHER_VAR}        # same, brace form
    KEY=keyring:SERVICE,USER  # resolved from OS keyring at load time
    # comment line

Environment variables may be overwritten when the resolved .env value
differs from the current os.environ value. This enables indirect values
like keyring references to be resolved and refreshed at load time.
"""
import os
import re
import subprocess
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
    except Exception as e:
        print(f"Warning: keyring module not available, cannot resolve {raw_value!r}: {e}")
        keyring = None

    secret = None
    if keyring is not None:
        try:
            secret = keyring.get_password(service, username)
        except Exception as e:
            print(f"Warning: failed to get password from keyring for {raw_value!r}: {e}")

    # Linux fallback: query Secret Service keyring directly via secret-tool.
    # This helps when python-keyring backend selection fails but the login
    # keyring is available through libsecret.
    if not secret and os.name == "posix":
        secret = _lookup_with_secret_tool(service, username)

    return secret if secret else raw_value


def _lookup_with_secret_tool(service, username):
    """Best-effort Linux Secret Service lookup via secret-tool.

    Tries common attribute names used by keyring/libsecret integrations.
    Returns None if secret-tool is unavailable or lookup fails.
    """
    attribute_variants = (
        ("service", service, "username", username),
        ("service", service, "user", username),
        ("service", service, "account", username),
    )

    for attrs in attribute_variants:
        cmd = ["secret-tool", "lookup", *attrs]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except FileNotFoundError:
            return None
        except Exception:
            continue

        if result.returncode == 0:
            candidate = result.stdout.strip()
            if candidate:
                return candidate

    return None


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

            if key not in os.environ or os.environ[key] != value:
                os.environ[key] = value
                set_keys.add(key)

    return set_keys
