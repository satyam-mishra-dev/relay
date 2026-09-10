import json

import relay.judge as judge


def test_parse_score_rejects_incomplete_json():
    assert judge.parse_score("no json here") is None
    assert judge.parse_score('{"grounded": 4, "resolves": 4}') is None
    scored = judge.parse_score('{"grounded":5,"resolves":3,"tone":4,"safe":4,"rationale":"ok"}')
    assert scored["overall"] == 4.0


def test_pairwise_keeps_agreeing_verdicts_and_ties_on_flip(monkeypatch):
    verdicts = [
        {"winner": "A"},
        {"winner": "B"},
        {"winner": "A"},
        {"winner": "A"},
        {"winner": "tie"},
        {"winner": "tie"},
    ]
    monkeypatch.setattr(judge, "complete_many", lambda jobs: [json.dumps(v) for v in verdicts])
    items = [("c", ["e"], "first", "second")] * 3
    assert judge.compare_many(items) == ["A", "tie", "tie"]
