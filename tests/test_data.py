import io

import pandas as pd

from relay.data import brand_stats, clean, is_english, threads

CSV = """tweet_id,author_id,inbound,created_at,text,response_tweet_id,in_response_to_tweet_id
1,900,True,Tue Oct 31 22:10:47 +0000 2017,@acme my app crashes &amp; I can't log in http://x.co,2,
2,acme,False,Tue Oct 31 22:12:47 +0000 2017,@900 Sorry! Please reinstall the app and try again.,3,1
3,900,True,Tue Oct 31 22:20:47 +0000 2017,@acme still broken,,2
4,901,True,Wed Nov 01 09:00:00 +0000 2017,@acme donde esta mi pedido por favor senor,5,
5,acme,False,Wed Nov 01 09:01:00 +0000 2017,@901 DM us,,4
6,902,True,Wed Nov 01 10:00:00 +0000 2017,@acme this is a tweet with no reply at all,,
"""


def test_threads_and_stats():
    t = threads(pd.read_csv(io.StringIO(CSV)))
    assert list(t["id"]) == [1, 4]
    row = t[t["id"] == 1].iloc[0]
    assert row["customer"] == "my app crashes & I can't log in"
    assert row["brand_reply"] == "Sorry! Please reinstall the app and try again."
    assert row["followup_1"] == "still broken"
    stats = brand_stats(t)
    assert stats.loc["acme", "threads"] == 2
    assert stats.loc["acme", "substantive_share"] == 0.5


def test_english_filter():
    assert is_english("my app crashes and I cannot log in")
    assert not is_english(clean("@acme donde esta mi pedido por favor senor"))
