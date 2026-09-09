from relay.retrieve import index, similar

ROWS = [
    {"id": 1, "customer": "my stream keeps buffering on roku", "brand_reply": "Try these steps"},
    {"id": 2, "customer": "how do I cancel my subscription", "brand_reply": "Here is how"},
    {"id": 3, "customer": "please add season 3 of that show", "brand_reply": "We will pass it on"},
    {"id": 4, "customer": "cancel my subscription on roku please", "brand_reply": "Steps here"},
    {"id": 5, "customer": "the stream keeps buffering every episode", "brand_reply": "Sorry!"},
]


def test_similar_ranks_the_matching_thread_first():
    built = index(ROWS)
    hits = similar("cancel my subscription on roku", 3, built)
    assert hits[0]["id"] == 4
    assert hits[0]["score"] > hits[1]["score"]
    assert set(hits[0]) == {"id", "customer", "brand_reply", "score"}
    assert similar("stream is buffering", 1, built)[0]["id"] in {1, 5}
