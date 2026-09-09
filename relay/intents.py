import json
import random
import re
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from relay.llm import complete_many, parse_json

WEAK_LABELS = Path("data/intents/weak_labels.jsonl")
WEAK_LABEL_SAMPLE = 1500

TAXONOMY = {
    "playback_error": (
        "The stream fails to play: buffering, freezing, blank screen or an error code. "
        'Cues: "error code 300", "keeps buffering".'
    ),
    "device_app_issue": (
        "The app itself is broken, crashing or missing features on a named device or platform. "
        'Cues: "app crashes on my Roku", "not working on my smart TV".'
    ),
    "login_access": (
        "Cannot sign in or reach the account: password, email, verification or lockout. "
        'Cues: "can\'t log in", "password reset link never arrives".'
    ),
    "billing_subscription": (
        "Money or plan: charges, refunds, trials, upgrades, downgrades, cancellation. "
        'Cues: "charged twice", "how do I cancel".'
    ),
    "ads_complaint": (
        "Ads or commercials: too many, repeated, or shown on a plan the customer pays to avoid. "
        'Cues: "I pay for no commercials", "same ad five times an episode".'
    ),
    "content_availability": (
        "A title already in the catalogue is missing, removed, out of order or blocked locally. "
        'Cues: "episodes disappeared", "not available in my area".'
    ),
    "content_request": (
        "Asks the service to add a show, season, channel or sports feed it does not carry. "
        'Cues: "please add season 3", "when will you get NBC".'
    ),
    "product_feedback": (
        "An opinion about the interface, design or product direction rather than a fault to fix. "
        'Cues: "the new interface is terrible", "bring back the old app".'
    ),
    "feature_question": (
        "Asks how something works or whether it is possible: plans, devices, settings, live TV. "
        'Cues: "how do I turn off autoplay", "does it work on Chromecast".'
    ),
    "praise_or_thanks": (
        "Compliments, thanks, or friendly chatter with no request for support. "
        'Cues: "thanks for the quick help", "love the new lineup".'
    ),
    "other": (
        "Anything else: unclear, off-topic, empty or not about this service. "
        'Cues: "hey guys", "<link>".'
    ),
}

EXAMPLES = {
    "playback_error": [
        "getting error code 300. What is this & how can we get it fixed??",
        "been trying to watch and it keeps buffering... tried everything.",
    ],
    "device_app_issue": [
        "app is still not working on my smart tv. I've reset multiple times and nothing",
        "why does new iPhone app show black screen when connected to Apple TV?",
    ],
    "login_access": [
        "I have been trying to cancel my HBO trial for over a week but can't access my account!",
        "password reset email never shows up, I am locked out",
    ],
    "billing_subscription": [
        "I canceled hulu but was charged. How do I get that back?",
        "Just upgraded to no commercials, how do I downgrade again?",
    ],
    "ads_complaint": [
        "at which time should there be ads in past programming when you pay for no ads?",
        "Why do I have to pay for Hulu and have to watch commercials?",
    ],
    "content_availability": [
        "why are my favorite shows all saying not available right now??",
        "Alias series 1-5 was removed last night while I was half way through",
    ],
    "content_request": [
        "when is season 3 of rick and morty coming",
        "Hulu Team please add Bein sports to live TV line up.",
    ],
    "product_feedback": [
        "Your slick new interface makes browsing and discovery more difficult.",
        "the new iPad platform is terrible!!",
    ],
    "feature_question": [
        "is there a way to enable 5.1 sound on one",
        "do you have a free trial so I can compare Hulu Live with my current subscription?",
    ],
    "praise_or_thanks": ["thanks so much, that fixed it!", "It's so good, loving the new lineup"],
    "other": ["hey guys", "<link>"],
}

KEYWORD_RULES = [
    ("ads_complaint", r"\bads?\b|commercial"),
    ("billing_subscription", r"charg|refund|bill|subscri|cancel|trial|price|payment|plan\b"),
    ("login_access", r"log ?in|sign ?in|password|locked out|can'?t access my account|verification"),
    (
        "playback_error",
        r"buffer|error|freez|won'?t (play|load)|keeps? (stopping|crashing)|glitch|lag",
    ),
    (
        "device_app_issue",
        r"roku|apple tv|chromecast|xbox|playstation|smart ?tv|fire ?stick"
        r"|app (is|isn'?t|not|crash)",
    ),
    (
        "content_request",
        r"please add|can you add|when (will|is|does).*(come|add|available)|bring back",
    ),
    ("content_availability", r"not available|unavailable|missing|removed|disappear|out of order"),
    (
        "product_feedback",
        r"(new|old) (interface|app|design|layout)|terrible|worst|hate the",
    ),
    ("praise_or_thanks", r"thank|thx|love (you|hulu|the)|appreciate|you rock|best"),
    ("feature_question", r"^(how|does|do you|is there|can i|what)\b"),
]


def prompt(text):
    lines = [f"{name}: {definition}" for name, definition in TAXONOMY.items()]
    shots = [f"{name} <- {ex}" for name, examples in EXAMPLES.items() for ex in examples]
    system = (
        "You label an inbound customer tweet sent to a streaming service's support handle "
        "with exactly one intent from this taxonomy.\n\n"
        + "\n".join(lines)
        + "\n\nExamples:\n"
        + "\n".join(shots)
        + '\n\nAnswer with JSON only: {"intent": "<name>", "confidence": <0-1>}'
    )
    return {"system": system, "user": text, "model": "sonnet", "max_tokens": 60}


def classify_llm(texts):
    out = []
    for raw in complete_many([prompt(t) for t in texts]):
        parsed = parse_json(raw) or {}
        intent = parsed.get("intent")
        out.append(
            (
                intent if intent in TAXONOMY else "other",
                float(parsed.get("confidence", 0.0)) if intent in TAXONOMY else 0.0,
            )
        )
    return out


def majority(train, texts):
    top = Counter(label for _, label in train).most_common(1)[0][0]
    return [top] * len(texts)


def keywords(train, texts):
    out = []
    for text in texts:
        lowered = text.lower()
        hit = next((name for name, rule in KEYWORD_RULES if re.search(rule, lowered)), "other")
        out.append(hit)
    return out


def tfidf_lr(train, texts):
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    model.fit([t for t, _ in train], [label for _, label in train])
    return list(model.predict(texts))


def llm(train, texts):
    return [intent for intent, _ in classify_llm(texts)]


CLASSIFIERS = {"majority": majority, "keywords": keywords, "tfidf_lr": tfidf_lr, "llm": llm}


def weak_training_set(path=WEAK_LABELS):
    return [
        (row["customer"], row["intent"]) for row in map(json.loads, path.read_text().splitlines())
    ]


def build_weak_labels(rows, n=WEAK_LABEL_SAMPLE, path=WEAK_LABELS):
    sample = random.Random(0).sample(rows, min(n, len(rows)))
    labels = classify_llm([r["customer"] for r in sample])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps({"id": r["id"], "customer": r["customer"], "intent": i, "confidence": c})
            + "\n"
            for r, (i, c) in zip(sample, labels, strict=True)
        )
    )
    return Counter(i for i, _ in labels)


def main():
    from relay.data import corpus

    print(build_weak_labels(corpus()).most_common())


if __name__ == "__main__":
    main()
