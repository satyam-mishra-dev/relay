import json
import random
from collections import defaultdict
from pathlib import Path

from relay.agent import handle
from relay.data import pool
from relay.intents import classify_llm
from relay.retrieve import index

CANDIDATES = Path("data/golden/candidates.jsonl")
AGENT_OUTPUTS = Path("results/agent_outputs.jsonl")
STRATIFIED = 120
UNIFORM = 80


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def round_robin(rows, n, rng):
    by_intent = defaultdict(list)
    for row in rows:
        by_intent[row["prelabel_intent"]].append(row)
    for group in by_intent.values():
        rng.shuffle(group)
    picked = {}
    while len(picked) < n and any(by_intent.values()):
        for intent in sorted(by_intent):
            if by_intent[intent] and len(picked) < n:
                row = by_intent[intent].pop()
                picked[row["id"]] = row
    return picked


def sample(rows, rng):
    picked = {
        i: dict(row, sample_stratum="stratified")
        for i, row in round_robin(rows, STRATIFIED, rng).items()
    }
    rest = [r for r in rows if r["id"] not in picked]
    for row in rng.sample(rest, min(UNIFORM, len(rest))):
        picked[row["id"]] = dict(row, sample_stratum="uniform")
    return list(picked.values())


def main():
    rows = pool()
    labels = classify_llm([r["customer"] for r in rows])
    labelled = [
        {
            "id": r["id"],
            "customer": r["customer"],
            "brand_reply": r["brand_reply"],
            "followups": r["followups"],
            "prelabel_intent": intent,
            "prelabel_confidence": confidence,
        }
        for r, (intent, confidence) in zip(rows, labels, strict=True)
    ]
    candidates = sample(labelled, random.Random(0))
    write_jsonl(CANDIDATES, candidates)
    built = index()
    write_jsonl(
        AGENT_OUTPUTS,
        [dict(handle(c["customer"], built), id=c["id"]) for c in candidates],
    )
    print(len(candidates), "candidates")


if __name__ == "__main__":
    main()
