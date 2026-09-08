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
