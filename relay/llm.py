import hashlib
import json
import os
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

API_MODELS = {"sonnet": "claude-sonnet-5", "opus": "claude-opus-5"}
PRICES = {"claude-sonnet-5": (2.0, 10.0), "claude-opus-5": (5.0, 25.0)}

CACHE_DIR = Path("data/cache")
SPEND = CACHE_DIR / "spend.json"
_ledger_lock = threading.Lock()


def load_env(path=Path(".env")):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        name, sep, value = line.partition("=")
        if sep and not name.strip().startswith("#"):
            os.environ.setdefault(name.strip(), value.strip().strip("'\""))


load_env()


def ledger():
    return json.loads(SPEND.read_text()) if SPEND.exists() else {"usd": 0.0, "calls": 0}


def record(model, usage):
    price_in, price_out = PRICES[API_MODELS[model]]
    cost = (
        usage.input_tokens * price_in
        + usage.output_tokens * price_out
        + (getattr(usage, "cache_read_input_tokens", 0) or 0) * price_in * 0.1
        + (getattr(usage, "cache_creation_input_tokens", 0) or 0) * price_in * 1.25
    ) / 1e6
    with _ledger_lock:
        spent = ledger()
        SPEND.parent.mkdir(parents=True, exist_ok=True)
        SPEND.write_text(
            json.dumps({"usd": round(spent["usd"] + cost, 6), "calls": spent["calls"] + 1})
        )
    return cost


def client():
    import anthropic

    return anthropic.Anthropic()


def _key(model, system, user):
    return hashlib.sha256("\x00".join((model, system, user)).encode()).hexdigest()[:24]


def _api(system, user, model, max_tokens):
    if ledger()["usd"] >= float(os.environ.get("RELAY_BUDGET_USD", 1.0)):
        raise RuntimeError("api budget exhausted")
    message = client().messages.create(
        model=API_MODELS[model],
        max_tokens=max_tokens,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}],
    )
    record(model, message.usage)
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
    choice = os.environ.get("RELAY_BACKEND") or (
        "api" if os.environ.get("ANTHROPIC_API_KEY") else "cli"
    )
    return _api if choice == "api" else _cli


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


if __name__ == "__main__":
    spent = ledger()
    budget = float(os.environ.get("RELAY_BUDGET_USD", 1.0))
    print(f"api spend ${spent['usd']:.6f} over {spent['calls']} calls, budget ${budget:.2f}")
