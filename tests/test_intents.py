from relay.intents import TAXONOMY, keywords, majority, tfidf_lr


def test_taxonomy_is_named_and_defined():
    assert 9 <= len(TAXONOMY) <= 11
    assert "other" in TAXONOMY
    assert all(len(d.split()) > 5 and "Cues:" in d for d in TAXONOMY.values())


def test_keyword_and_trivial_classifiers():
    texts = [
        "I was charged twice for my subscription",
        "the stream keeps buffering and then an error code appears",
        "why am I seeing commercials when I pay for no ads",
        "zzzz",
    ]
    assert keywords(None, texts) == [
        "billing_subscription",
        "playback_error",
        "ads_complaint",
        "other",
    ]
    train = [("a", "playback_error"), ("b", "playback_error"), ("c", "ads_complaint")]
    assert majority(train, texts) == ["playback_error"] * 4
    assert all(label in TAXONOMY for label in keywords(None, texts))


def test_tfidf_lr_learns_training_labels():
    train = [
        ("charged twice for my plan", "billing_subscription"),
        ("my card was billed again", "billing_subscription"),
        ("buffering error on every show", "playback_error"),
        ("error code keeps showing", "playback_error"),
    ]
    assert tfidf_lr(train, ["billed twice again"]) == ["billing_subscription"]
