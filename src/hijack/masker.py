"""Provenance masking for the restricted guard view (protocol §4.3).

Each argument value of a tool call is either kept (it comes from the trusted user
request or the user's own details) or replaced by a tag that says where it came
from, so the guard judges provenance and structure instead of attacker text.
"""

import re
from dataclasses import dataclass

EMAIL = re.compile(r"^[\w.+-]+@[\w-]+(\.[\w-]+)+$")
URL = re.compile(r"^(https?://|www\.)\S+$", re.I)
IBAN = re.compile(r"^[A-Z]{2}\d{2}[A-Z0-9]{8,30}$")
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?$")


@dataclass
class Observation:
    """One tool output the agent has seen so far."""

    step: int
    tool: str
    text: str


def normalize(text: str) -> str:
    """Collapse whitespace and YAML-escaped newlines so wrapped tool outputs still match."""
    return " ".join(text.replace("\\n", " ").split())


def value_kind(value: str) -> str:
    v = value.strip()
    for name, pattern in (("email", EMAIL), ("url", URL), ("iban", IBAN), ("date", DATE)):
        if pattern.match(v):
            return name
    return "text"


def windows(text: str, size: int):
    """All substrings of `text` with length `size` (or `text` itself if shorter)."""
    if len(text) <= size:
        yield text
        return
    for i in range(len(text) - size + 1):
        yield text[i : i + size]


def find_source(value: str, observations: list[Observation], min_len: int) -> Observation | None:
    """First observation that contains `value`, or any `min_len`-character piece of it."""
    v = normalize(value)
    if not v:
        return None
    for obs in observations:
        text = normalize(obs.text)
        if v in text or (len(v) > min_len and any(w in text for w in windows(v, min_len))):
            return obs
    return None


def mask_value(value, trusted_text: str, trusted_values: set[str], observations, min_len, path, mask_map):
    if isinstance(value, dict):
        return {k: mask_value(v, trusted_text, trusted_values, observations, min_len, f"{path}.{k}", mask_map)
                for k, v in value.items()}
    if isinstance(value, list):
        return [mask_value(v, trusted_text, trusted_values, observations, min_len, f"{path}[{i}]", mask_map)
                for i, v in enumerate(value)]
    if not isinstance(value, str) or not value.strip():
        return value  # numbers, booleans and empty strings carry no free text
    v = normalize(value)
    if v in trusted_values or v in trusted_text:
        return value
    kind = value_kind(value)
    source = find_source(value, observations, min_len)
    if source is not None:
        tag = f"<untrusted:{kind} len={len(value)} from={source.tool}#{source.step}>"
    else:
        tag = f"<generated:{kind} len={len(value)}>"
    mask_map.append({"path": path, "value": value, "tag": tag})
    return tag


def mask_args(args: dict, user_request: str, trusted: list[str], observations: list[Observation],
              min_len: int = 12) -> tuple[dict, list[dict]]:
    """Return masked arguments and the list of replacements made."""
    mask_map: list[dict] = []
    trusted_values = {normalize(t) for t in trusted}
    masked = mask_value(args, normalize(user_request), trusted_values, observations, min_len, "args", mask_map)
    return masked, mask_map


def leaked(secrets: list[str], rendered: str, allowed: str, min_len: int = 12) -> bool:
    """True if any `min_len`-character piece of an attacker string reaches `rendered`.

    Pieces that also occur in `allowed` (user request, trusted values, tool list) are ignored,
    because the guard would see them anyway.
    """
    rendered_n, allowed_n = normalize(rendered), normalize(allowed)
    for secret in secrets:
        s = normalize(secret)
        for w in windows(s, min_len):
            if len(w) >= min_len and w.strip() and w in rendered_n and w not in allowed_n:
                return True
    return False
