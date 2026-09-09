import gzip
import html
import json
import re
from pathlib import Path

import pandas as pd

RAW = Path("data/raw/twcs.csv")
OUT = Path("data/brand/threads.jsonl.gz")
SPLIT = Path("data/brand/split.json")
BRAND = "hulu_support"
CAP = 20000
POOL_SHARE = 0.15

MENTION = re.compile(r"@\w+")
URL = re.compile(r"https?://\S+|www\.\S+")
WHITESPACE = re.compile(r"\s+")
DM_REQUEST = re.compile(r"\b(dm|direct message|private message|pm us)\b", re.I)
STOPWORDS = {
    "the", "to", "you", "and", "for", "is", "it", "my", "in", "on", "of", "a", "i",
    "we", "this", "that", "with", "not", "can", "your", "me", "have", "but", "are",
}


def clean(text):
    text = URL.sub("<link>", MENTION.sub(" ", html.unescape(str(text))))
    return WHITESPACE.sub(" ", text).strip()


def is_english(text):
    chars = [c for c in text if not c.isspace()]
    if len(chars) < 5:
        return False
    ascii_share = sum(c.isascii() for c in chars) / len(chars)
    return ascii_share > 0.9 and bool(set(re.findall(r"[a-z']+", text.lower())) & STOPWORDS)


def is_substantive(reply):
    return not (DM_REQUEST.search(reply) and len(reply) <= 120)


def load(path=RAW):
    df = pd.read_csv(path, dtype={"author_id": "string", "text": "string"})
    df["tweet_id"] = pd.to_numeric(df["tweet_id"], errors="coerce")
    df["in_response_to_tweet_id"] = pd.to_numeric(df["in_response_to_tweet_id"], errors="coerce")
    return df.dropna(subset=["tweet_id", "text"]).sort_values("tweet_id")


def threads(df):
    outbound = df[~df["inbound"]].drop_duplicates("in_response_to_tweet_id")
    any_reply = df.dropna(subset=["in_response_to_tweet_id"]).drop_duplicates(
        "in_response_to_tweet_id"
    )
    roots = df[df["inbound"] & df["in_response_to_tweet_id"].isna()]
    cols = ["tweet_id", "text", "author_id"]
    t = roots[cols + ["created_at"]].merge(
        outbound[cols + ["in_response_to_tweet_id"]],
        left_on="tweet_id",
        right_on="in_response_to_tweet_id",
        suffixes=("", "_b"),
    )
    for step in ("_f1", "_f2"):
        prev = "tweet_id_b" if step == "_f1" else "tweet_id_f1"
        t = t.merge(
            any_reply[["in_response_to_tweet_id", "tweet_id", "text"]].rename(
                columns={
                    "in_response_to_tweet_id": "parent" + step,
                    "tweet_id": "tweet_id" + step,
                    "text": "text" + step,
                }
            ),
            left_on=prev,
            right_on="parent" + step,
            how="left",
        )
    return pd.DataFrame(
        {
            "id": t["tweet_id"].astype("int64"),
            "brand": t["author_id_b"],
            "customer": t["text"].map(clean),
            "brand_reply": t["text_b"].map(clean),
            "followup_1": t["text_f1"].map(clean),
            "followup_2": t["text_f2"].map(clean),
            "created_at": pd.to_datetime(t["created_at"], format="%a %b %d %H:%M:%S %z %Y"),
        }
    )


def brand_stats(t):
    t = t.assign(
        substantive=t["brand_reply"].map(is_substantive), length=t["brand_reply"].str.len()
    )
    stats = t.groupby("brand").agg(
        threads=("id", "size"),
        substantive_share=("substantive", "mean"),
        median_reply_len=("length", "median"),
    )
    return stats.sort_values("threads", ascending=False)


def english_threads(t, brand=BRAND):
    t = t[(t["brand"] == brand) & t["customer"].map(is_english)]
    return t.sort_values("created_at").tail(CAP)


def write(t, out=OUT, split=SPLIT):
    out.parent.mkdir(parents=True, exist_ok=True)
    cutoff = t["created_at"].quantile(1 - POOL_SHARE)
    with gzip.open(out, "wt") as f:
        for row in t.itertuples():
            f.write(
                json.dumps(
                    {
                        "id": int(row.id),
                        "customer": row.customer,
                        "brand_reply": row.brand_reply,
                        "followups": [x for x in (row.followup_1, row.followup_2) if x],
                        "created_at": row.created_at.isoformat(),
                    }
                )
                + "\n"
            )
    split.write_text(
        json.dumps(
            {
                "brand": BRAND,
                "pool_starts_at": cutoff.isoformat(),
                "corpus": int((t["created_at"] < cutoff).sum()),
                "pool": int((t["created_at"] >= cutoff).sum()),
            },
            indent=2,
        )
    )


def read(path=OUT):
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


def main():
    t = threads(load())
    print(brand_stats(t).head(20).to_string())
    chosen = english_threads(t)
    write(chosen)
    print(SPLIT.read_text())


if __name__ == "__main__":
    main()
