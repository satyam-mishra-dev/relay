from relay.llm import complete_many, parse_json
from relay.retrieve import similar

DIMENSIONS = ("grounded", "resolves", "tone", "safe")

POINTWISE = """You are grading one customer-support reply from the Twitter support account for a \
streaming service (hulu_support). Score strictly from the evidence given, not from general \
knowledge of the brand.

You will receive: the customer tweet, three historical brand replies retrieved as evidence for \
similar past issues, and the candidate reply to grade.

Score four dimensions 1-5 each. Length is not quality: a short reply that correctly asks one \
clarifying question can score 5 on resolves if that is what the evidence shows the brand doing \
for this kind of issue.

grounded - does the candidate's claim/step match what the evidence replies actually did?
1: states a fix, policy, or fact no evidence reply supports, or contradicts the evidence.
3: on-topic and plausible but not traceable to any specific evidence reply's approach.
5: the step, link, or question in the candidate matches an approach used in at least one \
evidence reply.

resolves - does it move the issue to its correct next step?
1: non-answer - generic apology, deflection, or ignores what was asked.
3: addresses the issue but the next step is vague or incomplete.
5: gives the exact next action (right fix step, right clarifying question, or right <link>) that \
matches the evidence pattern for this issue.

tone - brand voice: friendly, asks clarifying device/error questions, points to help via <link>.
1: cold, scolding, or off-brand (corporate boilerplate, curt, or over-apologetic).
3: polite but generic, not distinctly this brand's voice.
5: sounds like this brand: friendly, concise, right clarifying question or <link>.

safe - no invented policy, no promised outcome, no public request for sensitive data.
1: promises a refund/credit/specific outcome, invents a policy, or asks for password/account/\
payment info publicly.
3: no promises or invented policy, but asks for borderline personal info (e.g. email) without \
directing to DM.
5: no promises, no invented policy, any sensitive-data ask is directed to DM/private channel.

Output strict JSON only, no prose outside it:
{"grounded": <1-5>, "resolves": <1-5>, "tone": <1-5>, "safe": <1-5>, \
"rationale": "<one sentence>"}"""

PAIRWISE = """You are comparing two candidate replies (A and B) to the same customer tweet from \
the Twitter support account for a streaming service (hulu_support), given the same three \
historical brand replies as evidence.

Judge only on: grounded (matches an evidence reply's approach), resolves (correct next step - a \
well-targeted clarifying question can beat a longer non-answer), tone (brand voice: friendly, \
asks clarifying device/error questions, points to help via <link>), safe (no invented policy, no \
promised outcome, no public request for sensitive data).

Ignore the order A/B was presented in and ignore length - a shorter correct reply beats a longer \
vague one. Ties are allowed and preferred over a forced pick when the two are equivalent on all \
four dimensions.

Output strict JSON only, no prose outside it:
{"winner": "A"|"B"|"tie", "reason": "<one sentence>"}"""


def evidence_for(customer, built):
    return [hit["brand_reply"] for hit in similar(customer, 3, built)]


def shown(customer, evidence):
    lines = "\n".join(f"{i}. {e}" for i, e in enumerate(evidence, 1))
    return f"Customer tweet:\n{customer}\n\nEvidence replies:\n{lines}"


def pointwise_job(customer, evidence, reply, suffix=""):
    return {
        "system": POINTWISE + suffix,
        "user": f"{shown(customer, evidence)}\n\nCandidate reply:\n{reply}",
        "model": "opus",
        "max_tokens": 400,
    }


def pairwise_job(customer, evidence, a, b):
    return {
        "system": PAIRWISE,
        "user": f"{shown(customer, evidence)}\n\nReply A:\n{a}\n\nReply B:\n{b}",
        "model": "opus",
        "max_tokens": 400,
    }


def parse_score(raw):
    parsed = parse_json(raw)
    if not parsed or not all(d in parsed for d in DIMENSIONS):
        return None
    dims = {d: int(parsed[d]) for d in DIMENSIONS}
    return dims | {
        "overall": sum(dims.values()) / len(DIMENSIONS),
        "rationale": str(parsed.get("rationale", "")),
    }


def score_many(items, suffix=""):
    jobs = [pointwise_job(c, e, r, suffix) for c, e, r in items]
    return [parse_score(raw) for raw in complete_many(jobs)]


def score(customer, evidence, reply, suffix=""):
    return score_many([(customer, evidence, reply)], suffix)[0]


def parse_winner(raw):
    parsed = parse_json(raw) or {}
    winner = str(parsed.get("winner", "tie")).strip().upper()
    return winner if winner in ("A", "B") else "TIE"


def compare_details(items):
    jobs = []
    for customer, evidence, a, b in items:
        jobs.append(pairwise_job(customer, evidence, a, b))
        jobs.append(pairwise_job(customer, evidence, b, a))
    raw = [parse_winner(r) for r in complete_many(jobs)]
    return list(zip(raw[0::2], raw[1::2], strict=True))


def agreed(first, second):
    return first if first == {"A": "B", "B": "A", "TIE": "TIE"}[second] else "TIE"


def compare_many(items):
    verdicts = [agreed(first, second) for first, second in compare_details(items)]
    return [v if v in ("A", "B") else "tie" for v in verdicts]


def compare(customer, evidence, a, b):
    return compare_many([(customer, evidence, a, b)])[0]
