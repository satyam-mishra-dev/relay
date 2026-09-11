import json
import re
import sys

from relay.data import is_english
from relay.intents import TAXONOMY
from relay.llm import complete, parse_json
from relay.retrieve import similar

MIN_EVIDENCE_SCORE = 0.15
REASONS = (
    "account_action",
    "billing_dispute",
    "security_or_legal",
    "abusive",
    "repeat_unresolved",
    "ambiguous",
)
SECURITY = re.compile(
    r"\b(phishing|legit|scam|lawyer|legal|sue|suing|lawsuit|attorney|fraud|hacked|stolen|"
    r"identity theft|police|threat(en)?|kill myself|suicide|self.?harm|harass|unsafe)\b",
    re.I,
)
BILLING_DISPUTE = re.compile(
    r"\b(charged (me )?twice|double charged|billed twice|unauthori[sz]ed charge|chargeback|"
    r"charge I did ?n.t|dispute)\b",
    re.I,
)
ABUSE = re.compile(
    r"\b(fuck|screw) (you|u|off)\b|\byou('re| are)? ?(all )?(idiots?|morons?|clowns?|stupid|"
    r"useless|incompetent|liars?)\b|\b(idiots?|morons?|clowns?) (at|running|who)\b",
    re.I,
)
REPEAT_CONTACT = re.compile(
    r"\b(third|fourth|3rd|4th) time\b|\bcalled (you )?twice\b|\bno one (is )?(responding|"
    r"answering|helping)\b|\bno help\b|\breached out (before|already|twice)\b|"
    r"\bstill (no|waiting for a) (response|reply|answer)\b",
    re.I,
)
HUMAN_CHANNEL = re.compile(
    r"\bphone\b|\bchat\b|\bcall us\b|\bgive us a call\b|\breach out here\b|\bDM us\b", re.I
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

POLICY = """Escalate when the first correct action needs account access or a human channel: a \
specific charge, refund, trial or promo that must be looked up or changed; cancellation problems; \
entitlements such as DVR storage; password-reset emails not arriving; too-many-devices or \
home-location resets. Escalate on any security, legal or safety angle (possible phishing, \
threats), on abuse aimed at staff, on a repeat contact that says earlier help failed, and when \
there is no discernible request. Everything else is auto, even when the customer is angry: \
troubleshooting steps, clarifying questions, outage status, catalogue and rights answers, \
explaining a published plan, price or policy. If the correct reply is to send the customer to \
phone, chat or DM for an account lookup, the action is escalate with reason account_action."""

SCHEMA = """Answer with JSON only:
{"intent": "<taxonomy name>", "confidence": <0-1>, "reply": "<the tweet>",
 "action": "auto" | "escalate",
 "reason": "account_action" | "billing_dispute" | "security_or_legal" | "abusive" |
            "repeat_unresolved" | "ambiguous"}"""


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
    if not parsed:
        return escalated("other", 0.0, "", "unparseable", ids)
    intent = parsed.get("intent") if parsed.get("intent") in TAXONOMY else "other"
    confidence = float(parsed.get("confidence") or 0.0)
    reply = str(parsed.get("reply") or "").strip()
    reason = override(text, reply, parsed.get("action"), parsed.get("reason"), evidence)
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


def hard_reason(text, reply):
    if ABUSE.search(text):
        return "abusive"
    if SECURITY.search(text):
        return "security_or_legal"
    if BILLING_DISPUTE.search(text):
        return "billing_dispute"
    if REPEAT_CONTACT.search(text):
        return "repeat_unresolved"
    if HUMAN_CHANNEL.search(reply):
        return "account_action"
    return None


def override(text, reply, action, reason, evidence):
    if not reply:
        return "ambiguous"
    if not is_english(text):
        return "ambiguous"
    if evidence[0]["score"] < MIN_EVIDENCE_SCORE:
        return "no_grounding"
    hard = hard_reason(text, reply)
    if hard:
        return hard
    if action != "auto":
        return reason if reason in REASONS else "ambiguous"
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
