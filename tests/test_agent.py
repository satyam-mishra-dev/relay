import json

import relay.agent as agent

EVIDENCE = [{"id": 7, "customer": "buffering", "brand_reply": "Try this", "score": 0.5}]


def stub(reply_json):
    def fake(system, user, model, max_tokens=800):
        return reply_json

    return fake


def run(text, monkeypatch, reply_json, evidence=EVIDENCE):
    monkeypatch.setattr(agent, "similar", lambda t, k, built: evidence)
    monkeypatch.setattr(agent, "complete", stub(reply_json))
    return agent.handle(text)


GOOD = json.dumps(
    {"intent": "playback_error", "confidence": 0.9, "reply": "Try this", "action": "auto"}
)


def test_auto_when_grounded_and_confident(monkeypatch):
    out = run("my show keeps buffering on the app", monkeypatch, GOOD)
    assert out["action"] == "auto"
    assert out["evidence"] == [7]


def test_hard_rules_override_the_model(monkeypatch):
    sensitive = run("is this email from you legit or a scam", monkeypatch, GOOD)
    assert (sensitive["action"], sensitive["reason"]) == ("escalate", "security_or_legal")

    routed = json.dumps(
        {"intent": "billing_subscription", "confidence": 0.9, "action": "auto",
         "reply": "Sorry! Please give us a call so we can look at the account."}
    )
    assert run("why was I billed early", monkeypatch, routed)["reason"] == "account_action"

    repeat = run("this is the third time I am asking, no one is responding", monkeypatch, GOOD)
    assert repeat["reason"] == "repeat_unresolved"

    weak = [{"id": 7, "customer": "x", "brand_reply": "y", "score": 0.01}]
    ungrounded = run("my show keeps buffering", monkeypatch, GOOD, weak)
    assert ungrounded["reason"] == "no_grounding"

    unsure = json.dumps({"intent": "playback_error", "confidence": 0.2, "reply": "x",
                         "action": "auto"})
    assert run("my show keeps buffering", monkeypatch, unsure)["action"] == "auto"

    broken = run("my show keeps buffering", monkeypatch, "not json at all")
    assert (broken["action"], broken["reason"]) == ("escalate", "unparseable")

    empty = json.dumps({"intent": "other", "confidence": 0.9, "reply": "", "action": "auto"})
    assert run("go team", monkeypatch, empty)["reason"] == "ambiguous"
