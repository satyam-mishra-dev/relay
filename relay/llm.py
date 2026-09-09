import hashlib
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

API_MODELS = {"sonnet": "claude-sonnet-5", "opus": "claude-opus-5"}

CACHE_DIR = Path("data/cache")


def _key(model, system, user):
    return hashlib.sha256("\x00".join((model, system, user)).encode()).hexdigest()[:24]


def _api(system, user, model, max_tokens):
    import anthropic

    message = anthropic.Anthropic().messages.create(
        model=API_MODELS[model],
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in message.content if b.type == "text")


def _cli(system, user, model, max_tokens):
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    out = subprocess.run(
        [
            "claude", "-p",
            "--model", model,
            "--system-prompt", system,
            "--output-format", "json",
            "--tools", "",
            "--no-session-persistence",
            "--strict-mcp-config",
        ],
        input=user, capture_output=True, text=True, env=env, check=True, timeout=300,
    )
    return json.loads(out.stdout)["result"]


def backend():
    return _api if os.environ.get("ANTHROPIC_API_KEY") else _cli


def complete(system: str, user: str, model: str, max_tokens: int = 800) -> str:
    path = CACHE_DIR / f"{_key(model, system, user)}.json"
    if path.exists():
        return json.loads(path.read_text())["text"]
    call = backend()
    try:
        text = call(system, user, model, max_tokens)
    except Exception:
        text = call(system, user, model, max_tokens)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": model, "system": system, "user": user, "text": text}))
    return text


def complete_many(jobs: list[dict]) -> list[str]:
    with ThreadPoolExecutor(max_workers=6) as pool:
        return list(pool.map(lambda j: complete(**j), jobs))


def parse_json(text):
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None
