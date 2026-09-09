import json
import re
import sys

from relay.data import is_english
from relay.intents import TAXONOMY
from relay.llm import complete, parse_json
from relay.retrieve import similar

MIN_CONFIDENCE = 0.6
MIN_EVIDENCE_SCORE = 0.15
SENSITIVE = re.compile(
    r"\b(lawyer|legal|sue|suing|lawsuit|attorney|press|journalist|reporter|fraud|hacked|"
    r"stolen|identity theft|charged twice|double charged|unauthori[sz]ed charge|chargeback|"
    r"dispute|kill myself|suicide|self.?harm|die|threat|abuse|harass|racist|discriminat)\b",
    re.I,
)

VOICE = """Voice rules distilled from 11,679 real replies by this handle:
- One tweet, at most 280 characters; the median real reply is 109 and 90% are under 126.
- Half of all replies open with brief empathy: "Oh no!", "Sorry for the trouble", "Uh oh!",
  "Apologies for the mix-up". Use the customer's first name only if they gave it.
- 58% point the customer at a help article, written as the literal token <link>. Use <link>
  when you are sending them to steps you cannot type out; never invent a URL.
- 23% ask exactly one clarifying question (device, app version, browser, which show). Ask one
  only when the evidence shows the cause cannot be known yet.
- Fewer than 1 in 500 mention refunds, credits or compensation. Never promise a refund, a
  credit, a fix date, a feature or a content licence.
- At most one emoji, and only for light or positive messages.
- No agent signature, no hashtags, no "DM us" unless the evidence replies do it for this issue.
"""

POLICY = """Escalate instead of auto-handling when: the issue needs account access or a human
decision, the customer is angry enough to churn or threaten, money is disputed, the evidence
does not cover the issue, or you are unsure of the intent. Otherwise auto-handle."""

SCHEMA = """Answer with JSON only:
{"intent": "<taxonomy name>", "confidence": <0-1>, "reply": "<the tweet>",
 "action": "auto" | "escalate", "reason": "<short phrase>"}"""


def system_prompt():
    taxonomy = "\n".join(f"- {name}: {definition}" for name, definition in TAXONOMY.items())
    return (
        "You are the support agent for the hulu_support Twitter handle. You draft one public "
        "reply to an inbound customer tweet, grounded only in how this handle has answered "
        f"similar tweets before.\n\nIntent taxonomy:\n{taxonomy}\n\n{VOICE}\n{POLICY}\n\n{SCHEMA}"
    )


def user_prompt(text, evidence):
    shown = "\n\n".join(
        f"[{e['id']}] customer: {e['customer']}\n     reply: {e['brand_reply']}" for e in evidence
    )
    return f"Past threads from this handle:\n\n{shown}\n\nNew customer tweet:\n{text}"


def handle(text, built=None):
    evidence = similar(text, 5, built)
    raw = complete(system_prompt(), user_prompt(text, evidence), "sonnet", max_tokens=600)
    parsed = parse_json(raw)
    ids = [e["id"] for e in evidence]
    if not parsed or not parsed.get("reply"):
        return escalated("other", 0.0, "", "unparseable", ids)
    intent = parsed.get("intent") if parsed.get("intent") in TAXONOMY else "other"
    confidence = float(parsed.get("confidence") or 0.0)
    reply = str(parsed["reply"]).strip()
    reason = override(text, intent, confidence, parsed.get("action"), evidence)
    if reason:
        return escalated(intent, confidence, reply, reason, ids)
    return {
        "intent": intent,
        "confidence": confidence,
        "reply": reply,
        "action": "auto",
        "reason": "grounded_and_routine",
        "evidence": ids,
    }


def override(text, intent, confidence, action, evidence):
    if SENSITIVE.search(text):
        return "sensitive_topic"
    if not is_english(text):
        return "non_english"
    if evidence[0]["score"] < MIN_EVIDENCE_SCORE:
        return "no_grounding"
    if confidence < MIN_CONFIDENCE:
        return "low_confidence"
    if action != "auto":
        return "llm_escalated"
    return None


def escalated(intent, confidence, reply, reason, ids):
    return {
        "intent": intent,
        "confidence": confidence,
        "reply": reply,
        "action": "escalate",
        "reason": reason,
        "evidence": ids,
    }


if __name__ == "__main__":
    print(json.dumps(handle(sys.argv[1]), indent=2))
