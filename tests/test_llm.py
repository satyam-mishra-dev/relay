import relay.llm as llm


def test_cache_avoids_second_backend_call(tmp_path, monkeypatch):
    calls = []

    def fake(system, user, model, max_tokens):
        calls.append(user)
        return "ok"

    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(llm, "backend", lambda: fake)
    assert llm.complete("s", "u", "sonnet") == "ok"
    assert llm.complete("s", "u", "sonnet") == "ok"
    assert calls == ["u"]
    assert llm.complete_many([{"system": "s", "user": "u", "model": "sonnet"}]) == ["ok"]
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_retry_once_then_raises(tmp_path, monkeypatch):
    attempts = []

    def flaky(system, user, model, max_tokens):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError
        return "second"

    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(llm, "backend", lambda: flaky)
    assert llm.complete("s", "u", "opus") == "second"


class FakeUsage:
    input_tokens = 1000
    output_tokens = 500
    cache_read_input_tokens = 2000
    cache_creation_input_tokens = 0


class FakeBlock:
    type = "text"
    text = "hi"


class FakeClient:
    class messages:
        @staticmethod
        def create(**kwargs):
            return type("M", (), {"content": [FakeBlock()], "usage": FakeUsage()})()


def test_api_ledger_accumulates_and_stops_at_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(llm, "SPEND", tmp_path / "spend.json")
    monkeypatch.setattr(llm, "client", FakeClient)
    monkeypatch.setenv("RELAY_BACKEND", "api")
    monkeypatch.setenv("RELAY_BUDGET_USD", "1.0")
    assert llm.complete("s", "u", "sonnet") == "hi"
    spent = llm.ledger()
    assert spent["calls"] == 1
    assert abs(spent["usd"] - (1000 * 2 + 500 * 10 + 2000 * 2 * 0.1) / 1e6) < 1e-9

    monkeypatch.setenv("RELAY_BUDGET_USD", "0.000001")
    try:
        llm.complete("s", "other", "sonnet")
        raise AssertionError("budget not enforced")
    except RuntimeError as exc:
        assert "budget" in str(exc)
