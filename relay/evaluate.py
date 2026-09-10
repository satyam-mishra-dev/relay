import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, precision_recall_fscore_support

from relay.agent import MIN_EVIDENCE_SCORE, SENSITIVE
from relay.data import corpus, is_english
from relay.golden import AGENT_OUTPUTS, read_jsonl, write_jsonl
from relay.intents import CLASSIFIERS, weak_training_set
from relay.judge import agreed, compare_details, evidence_for, score_many
from relay.retrieve import index, similar

GOLDEN = Path("data/golden/golden.jsonl")
RATINGS = Path("data/golden/reply_ratings.jsonl")
RATINGS_KEY = Path("data/golden/reply_ratings_key.jsonl")
HUMAN_RATINGS = Path("data/golden/reply_ratings_human.jsonl")
SELF_CONSISTENCY = 30
METRICS = Path("results/metrics.json")
ERRORS = Path("results/errors.jsonl")
RESAMPLES = 1000
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def accuracy(truth, pred):
    return float(np.mean([t == p for t, p in zip(truth, pred, strict=True)]))


def macro_f1(truth, pred):
    return float(f1_score(truth, pred, average="macro", zero_division=0))


def bootstrap(truth, pred, metric):
    rng = np.random.default_rng(0)
    n = len(truth)
    draws = [
        metric([truth[i] for i in idx], [pred[i] for i in idx])
        for idx in rng.integers(0, n, size=(RESAMPLES, n))
    ]
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def intent_block(truth, predictions):
    labels = sorted(set(truth))
    out = {}
    for name, pred in predictions.items():
        per_class = f1_score(truth, pred, average=None, labels=labels, zero_division=0)
        out[name] = {
            "accuracy": accuracy(truth, pred),
            "accuracy_ci": bootstrap(truth, pred, accuracy),
            "macro_f1": macro_f1(truth, pred),
            "macro_f1_ci": bootstrap(truth, pred, macro_f1),
            "per_class_f1": dict(zip(labels, map(float, per_class), strict=True)),
        }
    return out


def triage_metrics(should_escalate, escalated):
    precision, recall, f1, _ = precision_recall_fscore_support(
        should_escalate, escalated, average="binary", zero_division=0
    )
    risky = [s for s, e in zip(should_escalate, escalated, strict=True) if s]
    missed = [1 for s, e in zip(should_escalate, escalated, strict=True) if s and not e]
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "unsafe_auto_rate": len(missed) / len(risky) if risky else 0.0,
        "auto_coverage": float(np.mean([not e for e in escalated])),
    }


def rules_only(golden, built):
    return [
        bool(SENSITIVE.search(g["customer"]))
        or not is_english(g["customer"])
        or similar(g["customer"], 1, built)[0]["score"] < MIN_EVIDENCE_SCORE
        for g in golden
    ]


def reply_block(golden, evidence, replies):
    items = [
        (g["customer"], evidence[g["id"]], replies[name][g["id"]])
        for name in sorted(replies)
        for g in golden
    ]
    scored = score_many(items)
    out = {}
    for i, name in enumerate(sorted(replies)):
        chunk = [s for s in scored[i * len(golden) : (i + 1) * len(golden)] if s]
        out[name] = {
            dim: float(np.mean([s[dim] for s in chunk]))
            for dim in ("grounded", "resolves", "tone", "safe", "overall")
        } | {"scored": len(chunk)}
    return out, scored


def coverage_risk(golden, outputs):
    curve = []
    for threshold in THRESHOLDS:
        auto = [
            outputs[g["id"]]["action"] == "auto" and outputs[g["id"]]["confidence"] >= threshold
            for g in golden
        ]
        risky = [g for g in golden if g["should_escalate"]]
        unsafe = [g for g, a in zip(golden, auto, strict=True) if g["should_escalate"] and a]
        curve.append(
            {
                "threshold": threshold,
                "auto_coverage": float(np.mean(auto)),
                "unsafe_auto_rate": len(unsafe) / len(risky) if risky else 0.0,
            }
        )
    return curve


def average_ranks(values):
    values = np.asarray(values, dtype=float)
    ranks = np.empty(len(values), dtype=float)
    ranks[values.argsort(kind="stable")] = np.arange(len(values), dtype=float)
    for value in np.unique(values):
        ranks[values == value] = ranks[values == value].mean()
    return ranks


def spearman(a, b):
    return float(np.corrcoef(average_ranks(a), average_ranks(b))[0, 1])


def kappa(a, b):
    return float(cohen_kappa_score(a, b, weights="quadratic"))


def judge_validation(built, flip_rate):
    sheet = sorted(read_jsonl(RATINGS), key=lambda r: r["id"])
    agent_side = {r["id"]: r["agent_is"] for r in read_jsonl(RATINGS_KEY)}
    human_rows = {r["id"]: r for r in read_jsonl(HUMAN_RATINGS)}
    items, human, systems = [], [], []
    for row in sheet:
        evidence = evidence_for(row["customer"], built)
        for side in ("a", "b"):
            items.append((row["customer"], evidence, row[f"reply_{side}"]))
            human.append(int(round(human_rows[row["id"]][f"score_{side}"])))
            systems.append("agent" if agent_side[row["id"]] == side else "nearest_reply")
    scored = score_many(items)
    kept = [(h, s["overall"], sys) for h, s, sys in zip(human, scored, systems, strict=True) if s]
    human_kept = [h for h, _, _ in kept]
    judge_kept = [j for _, j, _ in kept]
    rounded = [int(round(j)) for j in judge_kept]
    strict = score_many(items[:SELF_CONSISTENCY], suffix="\n\nBe strict.")
    pairs = [
        (int(round(a["overall"])), int(round(b["overall"])))
        for a, b in zip(scored[:SELF_CONSISTENCY], strict, strict=True)
        if a and b
    ]
    return {
        "rated_rows": len(sheet),
        "scored_replies": len(kept),
        "spearman": spearman(human_kept, judge_kept),
        "quadratic_kappa": kappa(human_kept, rounded),
        "exact_agreement": float(
            np.mean([h == j for h, j in zip(human_kept, rounded, strict=True)])
        ),
        "within_one_agreement": float(
            np.mean([abs(h - j) <= 1 for h, j in zip(human_kept, rounded, strict=True)])
        ),
        "human_mean": {
            system: float(np.mean([h for h, _, s in kept if s == system]))
            for system in ("agent", "nearest_reply")
        },
        "judge_mean": {
            system: float(np.mean([j for _, j, s in kept if s == system]))
            for system in ("agent", "nearest_reply")
        },
        "self_consistency_kappa": kappa([a for a, _ in pairs], [b for _, b in pairs]),
        "self_consistency_rows": len(pairs),
        "pairwise_order_flip_rate": flip_rate,
    }


def round_floats(value, places=4):
    if isinstance(value, float):
        return round(value, places)
    if isinstance(value, dict):
        return {k: round_floats(v, places) for k, v in value.items()}
    if isinstance(value, list):
        return [round_floats(v, places) for v in value]
    return value


def table(title, rows, columns):
    lines = [f"\n### {title}", "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def main():
    golden = read_jsonl(GOLDEN)
    outputs = {row["id"]: row for row in read_jsonl(AGENT_OUTPUTS)}
    built = index()
    texts = [g["customer"] for g in golden]
    truth = [g["intent"] for g in golden]
    train = weak_training_set()

    predictions = {name: fn(train, texts) for name, fn in sorted(CLASSIFIERS.items())}
    predictions["agent"] = [outputs[g["id"]]["intent"] for g in golden]

    should_escalate = [bool(g["should_escalate"]) for g in golden]
    triage = {
        "escalate_all": triage_metrics(should_escalate, [True] * len(golden)),
        "auto_all": triage_metrics(should_escalate, [False] * len(golden)),
        "rules_only": triage_metrics(should_escalate, rules_only(golden, built)),
        "agent": triage_metrics(
            should_escalate, [outputs[g["id"]]["action"] == "escalate" for g in golden]
        ),
    }

    canned = Counter(r["brand_reply"] for r in corpus()).most_common(1)[0][0]
    evidence = {g["id"]: evidence_for(g["customer"], built) for g in golden}
    nearest = {g["id"]: similar(g["customer"], 1, built)[0]["brand_reply"] for g in golden}
    replies = {
        "agent": {g["id"]: outputs[g["id"]]["reply"] for g in golden},
        "canned": {g["id"]: canned for g in golden},
        "nearest_reply": nearest,
    }
    reply, scored = reply_block(golden, evidence, replies)
    details = compare_details(
        [
            (g["customer"], evidence[g["id"]], replies["agent"][g["id"]], nearest[g["id"]])
            for g in golden
        ]
    )
    verdicts = [agreed(first, second) for first, second in details]
    flip_rate = float(np.mean([agreed(f, s) == "TIE" and f != "TIE" for f, s in details]))
    reply["pairwise_agent_vs_nearest_reply"] = {
        "win": verdicts.count("A"),
        "loss": verdicts.count("B"),
        "tie": verdicts.count("TIE"),
    }

    confusion = Counter((t, p) for t, p in zip(truth, predictions["agent"], strict=True) if t != p)
    metrics = {
        "brand": "hulu_support",
        "golden_rows": len(golden),
        "canned_reply": canned,
        "intent": intent_block(truth, predictions),
        "triage": triage,
        "reply": reply,
        "coverage_risk": coverage_risk(golden, outputs),
        "judge": judge_validation(built, flip_rate),
        "top_confusions": [
            {"truth": t, "predicted": p, "count": n}
            for (t, p), n in sorted(confusion.items(), key=lambda kv: (-kv[1], kv[0]))
        ],
    }
    metrics = round_floats(metrics)
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    agent_scores = scored[sorted(replies).index("agent") * len(golden) :][: len(golden)]
    write_jsonl(
        ERRORS,
        [
            {
                "id": g["id"],
                "customer": g["customer"],
                "true_intent": g["intent"],
                "predicted": {name: predictions[name][i] for name in sorted(predictions)},
                "should_escalate": g["should_escalate"],
                "escalate_reason": g["escalate_reason"],
                "agent_action": outputs[g["id"]]["action"],
                "agent_reason": outputs[g["id"]]["reason"],
                "agent_reply": outputs[g["id"]]["reply"],
                "judge": agent_scores[i],
            }
            for i, g in enumerate(golden)
            if predictions["agent"][i] != g["intent"]
            or (outputs[g["id"]]["action"] == "escalate") != g["should_escalate"]
        ],
    )
    print(report(metrics))


def report(m):
    intent_rows = [
        (
            name,
            f"{v['accuracy']:.3f}",
            f"[{v['accuracy_ci'][0]:.3f}, {v['accuracy_ci'][1]:.3f}]",
            f"{v['macro_f1']:.3f}",
            f"[{v['macro_f1_ci'][0]:.3f}, {v['macro_f1_ci'][1]:.3f}]",
        )
        for name, v in sorted(m["intent"].items())
    ]
    triage_rows = [
        (
            name,
            f"{v['precision']:.3f}",
            f"{v['recall']:.3f}",
            f"{v['f1']:.3f}",
            f"{v['unsafe_auto_rate']:.3f}",
            f"{v['auto_coverage']:.3f}",
        )
        for name, v in sorted(m["triage"].items())
    ]
    reply_rows = [
        (name, v["overall"], v["grounded"], v["resolves"], v["tone"], v["safe"])
        for name, v in sorted(m["reply"].items())
        if isinstance(v, dict) and "overall" in v
    ]
    curve_rows = [
        (c["threshold"], c["auto_coverage"], c["unsafe_auto_rate"]) for c in m["coverage_risk"]
    ]
    pair = m["reply"]["pairwise_agent_vs_nearest_reply"]
    return "\n".join(
        [
            table("Intent", intent_rows, ["classifier", "acc", "acc 95% CI", "macro-F1", "F1 CI"]),
            table(
                "Triage (escalate)",
                triage_rows,
                ["system", "P", "R", "F1", "unsafe_auto_rate", "auto_coverage"],
            ),
            table(
                "Reply (judge 1-5)",
                reply_rows,
                ["system", "overall", "grounded", "resolves", "tone", "safe"],
            ),
            table(
                "Coverage vs risk",
                curve_rows,
                ["confidence threshold", "auto_coverage", "unsafe_auto_rate"],
            ),
            f"\nPairwise agent vs nearest_reply: {pair['win']} win / {pair['tie']} tie "
            f"/ {pair['loss']} loss",
        ]
    )


if __name__ == "__main__":
    main()
