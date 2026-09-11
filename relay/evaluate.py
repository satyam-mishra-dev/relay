import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, precision_recall_fscore_support

from relay.agent import MIN_EVIDENCE_SCORE, hard_reason
from relay.data import corpus, is_english
from relay.golden import AGENT_OUTPUTS, read_jsonl, write_jsonl
from relay.intents import CLASSIFIERS, weak_training_set
from relay.judge import agreed, compare_details, evidence_for, score_many
from relay.retrieve import index, similar

GOLDEN = Path("data/golden/golden.jsonl")
RATINGS = Path("data/golden/reply_ratings.jsonl")
RATINGS_KEY = Path("data/golden/reply_ratings_key.jsonl")
HUMAN_RATINGS = Path("data/golden/reply_ratings_human.jsonl")
METRICS = Path("results/metrics.json")
ERRORS = Path("results/errors.jsonl")
ITERATIONS = Path("results/iterations.json")
CALIBRATION = 30
RESAMPLES = 1000
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SPLITS = {"all": None, "dev": "stratified", "test": "uniform"}
SYSTEMS = ("agent", "canned", "nearest_reply")


def accuracy(truth, pred):
    return float(np.mean([t == p for t, p in zip(truth, pred, strict=True)]))


def macro_f1(truth, pred, labels=None):
    labels = labels if labels is not None else sorted(set(truth))
    return float(f1_score(truth, pred, average="macro", labels=labels, zero_division=0))


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
    for name, pred in sorted(predictions.items()):
        per_class = f1_score(truth, pred, average=None, labels=labels, zero_division=0)
        out[name] = {
            "accuracy": accuracy(truth, pred),
            "accuracy_ci": bootstrap(truth, pred, accuracy),
            "macro_f1": macro_f1(truth, pred, labels),
            "macro_f1_ci": bootstrap(truth, pred, lambda t, p: macro_f1(t, p, labels)),
            "per_class_f1": dict(zip(labels, map(float, per_class), strict=True)),
        }
    return out


def triage_metrics(should_escalate, escalated):
    precision, recall, f1, _ = precision_recall_fscore_support(
        should_escalate, escalated, average="binary", zero_division=0
    )
    risky = [s for s in should_escalate if s]
    missed = [s for s, e in zip(should_escalate, escalated, strict=True) if s and not e]
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "unsafe_auto_rate": len(missed) / len(risky) if risky else 0.0,
        "auto_coverage": float(np.mean([not e for e in escalated])),
    }


def rules_only(rows, built):
    return [
        hard_reason(r["customer"], "") is not None
        or not is_english(r["customer"])
        or similar(r["customer"], 1, built)[0]["score"] < MIN_EVIDENCE_SCORE
        for r in rows
    ]


def coverage_risk(rows, outputs):
    curve = []
    for threshold in THRESHOLDS:
        auto = [
            outputs[r["id"]]["action"] == "auto" and outputs[r["id"]]["confidence"] >= threshold
            for r in rows
        ]
        risky = [r for r in rows if r["should_escalate"]]
        unsafe = [r for r, a in zip(rows, auto, strict=True) if r["should_escalate"] and a]
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
    if len(set(a)) < 2 and len(set(b)) < 2:
        return 1.0 if list(a) == list(b) else 0.0
    return float(cohen_kappa_score(a, b, weights="quadratic"))


def agreement(human, judge):
    rounded = [int(round(j)) for j in judge]
    return {
        "n": len(human),
        "spearman": spearman(human, judge),
        "quadratic_kappa": kappa(human, rounded),
        "exact_agreement": float(np.mean([h == j for h, j in zip(human, rounded, strict=True)])),
        "within_one_agreement": float(
            np.mean([abs(h - j) <= 1 for h, j in zip(human, rounded, strict=True)])
        ),
    }


def rating_scores(built):
    sheet = read_jsonl(RATINGS)
    agent_side = {r["id"]: r["agent_is"] for r in read_jsonl(RATINGS_KEY)}
    human_rows = {r["id"]: r for r in read_jsonl(HUMAN_RATINGS)}
    items, rated = [], []
    for position, row in enumerate(sheet):
        evidence = evidence_for(row["customer"], built)
        for side in ("a", "b"):
            items.append((row["customer"], evidence, row[f"reply_{side}"]))
            rated.append(
                {
                    "id": row["id"],
                    "position": position,
                    "system": "agent" if agent_side[row["id"]] == side else "nearest_reply",
                    "human": int(round(human_rows[row["id"]][f"score_{side}"])),
                }
            )
    scored = score_many(items)
    strict = score_many(items[:CALIBRATION], suffix="\n\nBe strict.")
    for entry, score in zip(rated, scored, strict=True):
        entry["judge"] = score["overall"] if score else None
    consistency = [
        (int(round(a["overall"])), int(round(b["overall"])))
        for a, b in zip(scored[:CALIBRATION], strict, strict=True)
        if a and b
    ]
    return rated, consistency


def judge_block(rated, consistency, flip_rate):
    kept = [r for r in rated if r["judge"] is not None]
    if not kept:
        return {"scored_replies": 0, "pairwise_order_flip_rate": flip_rate}
    human = [r["human"] for r in kept]
    judge = [r["judge"] for r in kept]
    holdout = [r for r in kept if r["position"] >= CALIBRATION]
    return agreement(human, judge) | {
        "scored_replies": len(kept),
        "holdout": agreement([r["human"] for r in holdout], [r["judge"] for r in holdout]),
        "human_mean": {
            system: float(np.mean([r["human"] for r in kept if r["system"] == system]))
            for system in ("agent", "nearest_reply")
        },
        "judge_mean": {
            system: float(np.mean([r["judge"] for r in kept if r["system"] == system]))
            for system in ("agent", "nearest_reply")
        },
        "perturbation_kappa": kappa([a for a, _ in consistency], [b for _, b in consistency]),
        "perturbation_rows": len(consistency),
        "pairwise_order_flip_rate": flip_rate,
    }


def split_block(rows, outputs, predictions, scored, details, rated, consistency, built):
    ids = [r["id"] for r in rows]
    index_of = {r["id"]: i for i, r in enumerate(rows)}
    truth = [r["intent"] for r in rows]
    should_escalate = [bool(r["should_escalate"]) for r in rows]
    verdicts = [agreed(*details[i]) for i in ids]
    flip_rate = float(
        np.mean([agreed(*details[i]) == "TIE" and details[i][0] != "TIE" for i in ids])
    )
    kept = [r for r in rated if r["id"] in index_of]
    confusion = Counter(
        (t, p)
        for t, p in zip(truth, [predictions["agent"][r["id"]] for r in rows], strict=True)
        if t != p
    )
    return {
        "rows": len(rows),
        "intent": intent_block(truth, {k: [v[i] for i in ids] for k, v in predictions.items()}),
        "triage": {
            "escalate_all": triage_metrics(should_escalate, [True] * len(rows)),
            "auto_all": triage_metrics(should_escalate, [False] * len(rows)),
            "rules_only": triage_metrics(should_escalate, rules_only(rows, built)),
            "agent": triage_metrics(
                should_escalate, [outputs[i]["action"] == "escalate" for i in ids]
            ),
        },
        "reply": {
            system: {
                dim: float(np.mean([scored[system][i][dim] for i in ids if scored[system][i]]))
                for dim in ("grounded", "resolves", "tone", "safe", "overall")
            }
            | {"scored": sum(1 for i in ids if scored[system][i])}
            for system in SYSTEMS
        }
        | {
            "pairwise_agent_vs_nearest_reply": {
                "win": verdicts.count("A"),
                "loss": verdicts.count("B"),
                "tie": verdicts.count("TIE"),
            }
        },
        "coverage_risk": coverage_risk(rows, outputs),
        "judge": judge_block(kept, consistency, flip_rate),
        "top_confusions": [
            {"truth": t, "predicted": p, "count": n}
            for (t, p), n in sorted(confusion.items(), key=lambda kv: (-kv[1], kv[0]))
        ],
    }


def round_floats(value, places=4):
    if isinstance(value, float):
        return round(value, places)
    if isinstance(value, dict):
        return {k: round_floats(v, places) for k, v in value.items()}
    if isinstance(value, list):
        return [round_floats(v, places) for v in value]
    return value


def main():
    golden = read_jsonl(GOLDEN)
    outputs = {row["id"]: row for row in read_jsonl(AGENT_OUTPUTS)}
    built = index()
    train = weak_training_set()
    texts = [g["customer"] for g in golden]
    ids = [g["id"] for g in golden]

    predictions = {
        name: dict(zip(ids, fn(train, texts), strict=True))
        for name, fn in sorted(CLASSIFIERS.items())
    }
    predictions["agent"] = {i: outputs[i]["intent"] for i in ids}

    canned = Counter(r["brand_reply"] for r in corpus()).most_common(1)[0][0]
    evidence = {i: evidence_for(g["customer"], built) for i, g in zip(ids, golden, strict=True)}
    nearest = {
        i: similar(g["customer"], 1, built)[0]["brand_reply"]
        for i, g in zip(ids, golden, strict=True)
    }
    replies = {
        "agent": {i: outputs[i]["reply"] for i in ids},
        "canned": {i: canned for i in ids},
        "nearest_reply": nearest,
    }
    flat = score_many(
        [
            (g["customer"], evidence[i], replies[system][i])
            for system in SYSTEMS
            for i, g in zip(ids, golden, strict=True)
        ]
    )
    scored = {
        system: dict(zip(ids, flat[k * len(ids) : (k + 1) * len(ids)], strict=True))
        for k, system in enumerate(SYSTEMS)
    }
    details = dict(
        zip(
            ids,
            compare_details(
                [
                    (g["customer"], evidence[i], replies["agent"][i], nearest[i])
                    for i, g in zip(ids, golden, strict=True)
                ]
            ),
            strict=True,
        )
    )
    rated, consistency = rating_scores(built)

    metrics = {"brand": "hulu_support", "canned_reply": canned, "golden_rows": len(golden)}
    for name, stratum in SPLITS.items():
        rows = [g for g in golden if stratum is None or g["sample_stratum"] == stratum]
        metrics[name] = split_block(
            rows, outputs, predictions, scored, details, rated, consistency, built
        )
    metrics = round_floats(metrics)
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    write_jsonl(
        ERRORS,
        [
            {
                "id": g["id"],
                "split": "dev" if g["sample_stratum"] == "stratified" else "test",
                "customer": g["customer"],
                "true_intent": g["intent"],
                "predicted": {name: predictions[name][g["id"]] for name in sorted(predictions)},
                "should_escalate": g["should_escalate"],
                "escalate_reason": g["escalate_reason"],
                "agent_action": outputs[g["id"]]["action"],
                "agent_reason": outputs[g["id"]]["reason"],
                "agent_reply": outputs[g["id"]]["reply"],
                "judge": scored["agent"][g["id"]],
            }
            for g in golden
            if predictions["agent"][g["id"]] != g["intent"]
            or (outputs[g["id"]]["action"] == "escalate") != bool(g["should_escalate"])
        ],
    )
    print(report(metrics))


def record(version, changes):
    metrics = json.loads(METRICS.read_text())
    entry = {"version": version, "changes": changes}
    for split in SPLITS:
        block = metrics[split]
        entry[split] = round_floats(
            {
                "intent_acc": block["intent"]["agent"]["accuracy"],
                "agent_macro_f1": block["intent"]["agent"]["macro_f1"],
                "llm_intent_acc": block["intent"]["llm"]["accuracy"],
                "unsafe_auto_rate": block["triage"]["agent"]["unsafe_auto_rate"],
                "auto_coverage": block["triage"]["agent"]["auto_coverage"],
                "judge_overall_agent": block["reply"]["agent"]["overall"],
                "judge_overall_nearest": block["reply"]["nearest_reply"]["overall"],
                "judge_kappa": block["judge"]["quadratic_kappa"],
            }
        )
    history = json.loads(ITERATIONS.read_text()) if ITERATIONS.exists() else []
    history = [h for h in history if h["version"] != version] + [entry]
    ITERATIONS.write_text(json.dumps(history, indent=2) + "\n")
    print(json.dumps(entry, indent=2))


def table(title, rows, columns):
    lines = [f"\n### {title}", "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def report(metrics):
    out = []
    for split in SPLITS:
        m = metrics[split]
        intent_rows = [
            (name, v["accuracy"], f"[{v['accuracy_ci'][0]}, {v['accuracy_ci'][1]}]", v["macro_f1"])
            for name, v in sorted(m["intent"].items())
        ]
        triage_rows = [
            (name, v["precision"], v["recall"], v["f1"], v["unsafe_auto_rate"], v["auto_coverage"])
            for name, v in sorted(m["triage"].items())
        ]
        reply_rows = [
            (name, v["overall"], v["grounded"], v["resolves"], v["tone"], v["safe"])
            for name, v in sorted(m["reply"].items())
            if isinstance(v, dict) and "overall" in v
        ]
        pair = m["reply"]["pairwise_agent_vs_nearest_reply"]
        out += [
            f"\n## {split} ({m['rows']} rows)",
            table("Intent", intent_rows, ["classifier", "acc", "acc 95% CI", "macro-F1"]),
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
            f"\nPairwise agent vs nearest_reply: {pair['win']} win / {pair['tie']} tie "
            f"/ {pair['loss']} loss",
            f"Judge vs human: spearman {m['judge'].get('spearman')} kappa "
            f"{m['judge'].get('quadratic_kappa')} within1 {m['judge'].get('within_one_agreement')}",
        ]
    return "\n".join(out)


if __name__ == "__main__":
    main()
