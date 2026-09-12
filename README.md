# Relay

Relay is a support agent for one brand's Twitter support handle. It reads an inbound customer tweet, classifies the intent, drafts a reply grounded in how the brand actually resolved similar tweets in the past, and decides whether to send the draft automatically or hand the case to a human, with a stated reason. The evaluation harness is the larger half of the project: a hand-labelled golden set, two baselines per task, an LLM judge whose agreement with a human rater is measured, an iteration log with a dev/test split, and a failure analysis.

Brand: `hulu_support`, chosen from the Kaggle Customer Support on Twitter dataset.

| Headline (200 golden rows) | Relay | Best baseline |
|---|---|---|
| Intent accuracy / macro-F1 | 0.82 / 0.81 | TF-IDF + logistic regression 0.65 / 0.61 |
| Missed escalations (should escalate, was auto-handled) | 0 of 19 | rules-only 17 of 19 |
| Auto-handle coverage | 78% | escalate-all 0% |
| Reply quality, LLM judge 1 to 5 | 4.35 | nearest historical reply 3.00 |
| Pairwise vs nearest historical reply | 181 wins, 14 ties, 5 losses | |
| Judge vs human agreement, weighted kappa | 0.61 on the held-out half | |

Every number above is regenerated offline by `make reproduce` from committed model outputs. Read the section "What is misleading about the headline number" before quoting any of them.

## Quickstart

Reproduce the results without any API key. Needs Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```
git clone <this repo> relay && cd relay
uv sync --frozen
make reproduce
```

`make reproduce` recomputes every metric from the committed LLM cache (`data/cache`, 6,605 prompt/response pairs) and fails loudly if any prompt is missing. It takes about 15 seconds after the dependency sync and writes `results/metrics.json`, `results/errors.jsonl`, and a table to stdout. CI runs the same command and fails if the metrics file drifts.

To handle a new message live:

```
export ANTHROPIC_API_KEY=...
uv run python -m relay.agent "my hulu app on roku freezes every 5 minutes, already reinstalled it"
```

```json
{"intent": "device_app_issue", "confidence": 0.88,
 "reply": "Sorry for the trouble! Since reinstalling didn't help, try these steps for Roku freezing: <link>",
 "action": "auto", "reason": "grounded_and_routine", "evidence": [1025914, 2311736, 2978047, 1454896, 214609]}
```

```
uv run python -m relay.agent "I cancelled last month but you charged me again today, I want my money back"
```

```json
{"intent": "billing_subscription", "confidence": 0.95,
 "reply": "Sorry for any unexpected charges! Please call us/chat us so we can take a look at our options: <link>",
 "action": "escalate", "reason": "account_action", "evidence": [2644865, 3169, 59941, 1144536, 1527398]}
```

`make eval` re-runs the whole harness live, using the cache for anything already seen. Two LLM backends exist: the Anthropic API when `ANTHROPIC_API_KEY` is set, and the Claude Code CLI (`claude -p`) otherwise, selectable with `RELAY_BACKEND=api|cli`. The API path meters every response against the price table and refuses to run once cumulative spend in `data/cache/spend.json` reaches `RELAY_BUDGET_USD` (default $1.00). Stable system prompts are marked for prompt caching. Every response is cached on disk by a hash of model, system prompt and user prompt, so re-running anything already evaluated costs nothing.

Rebuilding the data from scratch needs the 516 MB raw file from Kaggle (or its Hugging Face mirror) at `data/raw/twcs.csv`, then `make data`. That step is not needed for reproduction: the brand subsample is committed.

## Problem framing

### Why hulu_support

The dataset has 733,498 reconstructable threads across dozens of brands. The brand had to have enough volume to learn from, replies that contain actual resolutions rather than channel hand-offs, and a bounded domain so that a small intent taxonomy is honest. Per-brand statistics on the full file:

| brand | threads | substantive reply share | hand-off share | troubleshooting language |
|---|---|---|---|---|
| AmazonHelp | 76,770 | 0.99 | 0.09 | 0.20 |
| AppleSupport | 74,569 | 0.62 | 0.55 | 0.56 |
| Uber_Support | 39,344 | 0.75 | 0.78 | 0.05 |
| SpotifyCares | 26,048 | 0.71 | 0.40 | 0.33 |
| Delta | 24,535 | 0.84 | 0.21 | 0.09 |
| **hulu_support** | **14,047** | **1.00** | **0.06** | **0.39** |
| Tesco | 15,295 | 0.89 | 0.33 | 0.25 |

Substantive means the reply is not just a request to move to DM. Hand-off share is the fraction of replies that push the customer to another channel. SpotifyCares was the initial candidate and lost on the data: its modal reply is "Can you DM us your account's email address?", and an agent grounded on that corpus learns to deflect. hulu_support replies contain device questions, troubleshooting steps and rights explanations, in a bounded domain (playback, devices, catalogue, billing, ads). AmazonHelp has volume but 19% non-English traffic and an intent space spanning orders, delivery, devices and marketplace sellers.

### What good means for this brand

Reading a few hundred real hulu_support threads gives a clear picture of what the humans do. They answer in one tweet, median 109 characters. Half open with brief empathy. 58% point to a help article. 23% ask exactly one clarifying question (which device, which show). Fewer than 1 in 500 mention refunds or credits. Anything touching the account moves to phone or chat. So "good" here is: the right next step for this kind of issue, in that voice, without inventing anything, and knowing when the right next step is a human.

The decision that matters most is the third one. A wrong intent label costs a little. A wrong auto-reply to a billing dispute costs a customer. The metric the project optimises is therefore not F1 but the share of should-escalate cases the agent auto-handles (`unsafe_auto_rate`), traded against how much traffic it can take (`auto_coverage`).

### What was deliberately not built

- Multi-turn handling. Relay answers the first customer tweet only, although 99.7% of hulu threads have a follow-up.
- Non-English support. The pipeline drops non-English tweets with a cheap heuristic; the brand handles them by hand.
- A learned escalation model. Escalation is an explicit policy plus hard rules, because a policy can be read, argued with and changed, and there are only 19 positive examples to learn from.
- Embedding retrieval. TF-IDF retrieval keeps the project dependency-free and reproducible in seconds; the ablation is listed under next steps.
- A UI, a queue, a send path. Nothing here posts to Twitter.

## How it works

```
tweet ──> clean ──> TF-IDF retrieval over 11,679 past threads ──> top-5 (customer, brand reply)
                                                                        │
              taxonomy + voice rules + escalation policy + evidence ────┤
                                                                        ▼
                                                          one Claude Sonnet 5 call (JSON)
                                                                        │
                        hard rules override the model's action ◄────────┘
                        (human-channel reply, abuse, repeat contact, phishing, no grounding, unparseable)
                                                                        │
                                                                        ▼
                     {intent, confidence, reply, action, reason, evidence ids}
```

- `relay/data.py` reconstructs threads from `in_response_to_tweet_id` / `response_tweet_id`, keeps threads whose root is a customer tweet with a brand reply, strips handles, replaces URLs with a literal `<link>` token, and drops non-English rows. The last 15% of hulu threads by date (from 2017-11-24) are a held-out pool; everything before is the retrieval corpus. Nothing from the pool is ever retrieved, used as a few-shot example, or used to train a baseline.
- `relay/retrieve.py` is a single `TfidfVectorizer` over the corpus's customer texts. `similar(text, k)` returns the k nearest past threads with their brand replies.
- `relay/agent.py` builds one prompt: role, taxonomy, voice rules distilled from corpus statistics, the escalation policy, a JSON schema, then the five retrieved threads as evidence and the tweet. After the call, code applies hard rules that can only move `auto` to `escalate`, never the reverse. A reply that routes the customer to phone, chat or DM is by definition an escalation with reason `account_action`.
- `relay/llm.py` is the only place a model is called. Disk cache, backend selection, spend ledger, prompt caching.
- `relay/judge.py`, `relay/evaluate.py`: the harness, described below.

## Intent taxonomy

Derived from the data, not chosen up front. KMeans with 20 clusters over TF-IDF vectors of the corpus's customer texts; each cluster's top terms and sample tweets were read and the clusters collapsed into ten intents plus `other`. Corpus share comes from 1,500 corpus tweets labelled by the LLM classifier.

| intent | share | definition |
|---|---|---|
| playback_error | 23% | The stream fails to play: buffering, freezing, blank screen, error code |
| content_availability | 17% | A title, season or episode the service normally carries is missing, removed, mislabelled or wrong |
| content_request | 11% | Asks to add a title, season, channel or league the service does not carry |
| device_app_issue | 10% | The app is broken, crashing or missing features on a named device or platform |
| feature_question | 8% | Asks whether, when, where or how something works or is available: device, region, plan, setting, release timing |
| product_feedback | 8% | Opinion or request about the product itself, including feature requests such as shuffle or skip-intro |
| billing_subscription | 7% | Charges, refunds, trials, upgrades, downgrades, cancellation |
| ads_complaint | 6% | Ads: too many, repeated, or shown on a plan bought to avoid them |
| other | 5% | Unclear, off-topic, empty or not about this service |
| login_access | 4% | Cannot sign in or reach the account: password, email, verification, lockout |
| praise_or_thanks | 1% | Compliments, thanks or chatter with no request |

Banking77 was not used. Its intents are banking-specific and the point of the exercise was to derive intents from this brand's traffic.

## Golden set

`data/golden/golden.jsonl`, 200 rows from the held-out pool, so none of them is in the retrieval corpus or the weak-label training set.

Sampling. The pool's 2,062 tweets were pre-labelled by the LLM classifier. 120 rows were drawn round-robin across the eleven predicted intents so that rare intents such as login_access and praise_or_thanks have enough rows to measure, and 80 rows were drawn uniformly at random so that the real intent mix is also represented. The `sample_stratum` column records which. The stratified 120 doubled as the development split and the uniform 80 as the test split for the one improvement iteration.

Labelling. Every row was read with the brand's real reply visible, which is the ground truth for what the brand actually did, and labelled with intent, `should_escalate`, an escalation reason, and free-text notes on borderline rows. The Sonnet pre-label was a starting point and was overturned on 34 of 200 rows (17%). The escalation policy used for labelling is the same text that the agent is given:

> Escalate when the first correct action needs account access or a human channel: a specific charge, refund, trial or promo that must be looked up or changed; cancellation problems; entitlements such as DVR storage; password-reset emails not arriving; too-many-devices or home-location resets. Escalate on any security, legal or safety angle, on abuse aimed at staff, on a repeat contact that says earlier help failed, and when there is no discernible request. Everything else is auto, even when the customer is angry: troubleshooting steps, clarifying questions, outage status, catalogue and rights answers, explaining a published plan, price or policy.

Result: 19 of 200 should escalate (account_action 10, billing_dispute 5, ambiguous, security_or_legal, abusive and repeat_unresolved 1 each).

Reply ratings. 60 of the 200 rows (stratified over intent) were rated blind: the agent's draft and the nearest historical reply appear as A and B in random order (`reply_ratings.jsonl`, key in `reply_ratings_key.jsonl`), each scored 1 to 5 overall on the same anchors the judge uses (`reply_ratings_human.jsonl`). Unblinded afterwards: agent mean 4.08, nearest reply 2.10; the agent scored higher on 52 rows, tied on 7, lower on 1.

The main weakness of the golden set is that it has one annotator. There is no inter-annotator agreement figure, and the person who wrote the escalation policy also applied it.

## Results

All figures on the 200 golden rows unless stated. Bootstrap confidence intervals use 1,000 resamples with a fixed seed.

### Intent

| classifier | accuracy | 95% CI | macro-F1 | 95% CI |
|---|---|---|---|---|
| majority (trivial) | 0.180 | 0.130 to 0.235 | 0.028 | 0.021 to 0.035 |
| keywords (trivial) | 0.375 | 0.315 to 0.450 | 0.378 | 0.310 to 0.437 |
| tfidf_lr (simple) | 0.650 | 0.590 to 0.715 | 0.611 | 0.544 to 0.673 |
| llm classifier alone | 0.855 | 0.805 to 0.905 | 0.850 | 0.793 to 0.896 |
| **Relay agent** | **0.820** | 0.765 to 0.870 | **0.815** | 0.744 to 0.862 |

The `tfidf_lr` baseline is trained on 1,500 corpus tweets weak-labelled by the LLM classifier, never on golden rows. The agent's intent comes from the same call that drafts the reply, which is why it trails the dedicated classifier by three points: the prompt is doing three jobs. Per-class F1 for the agent: ads_complaint 0.97, billing_subscription 0.94, login_access 0.88, playback_error 0.87, product_feedback 0.84, praise_or_thanks 0.80, content_request 0.78, device_app_issue 0.77, content_availability 0.74, other 0.70, feature_question 0.67.

### Triage

| system | precision | recall | F1 | unsafe_auto_rate | auto_coverage |
|---|---|---|---|---|---|
| escalate-all (trivial) | 0.095 | 1.000 | 0.173 | 0.000 | 0.000 |
| auto-all (trivial) | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| rules-only (simple) | 0.667 | 0.105 | 0.182 | 0.895 | 0.985 |
| **Relay agent** | 0.432 | **1.000** | 0.603 | **0.000** | **0.780** |

`unsafe_auto_rate` is the share of the 19 should-escalate rows that were auto-handled. `auto_coverage` is the share of all rows the system would answer without a human. The agent escalates 44 rows, of which 19 are right and 25 are cautious. Sweeping the confidence threshold from 0.3 to 0.9 moves coverage from 0.78 down to 0.30 without changing the unsafe rate, which is already zero; confidence is not the signal that carries safety here, the policy and hard rules are.

### Reply quality

Judged by Claude Opus 5 on four dimensions, 1 to 5, against retrieved replies 2 to 4 as evidence (see judge validation for why not 1 to 3).

| system | overall | grounded | resolves | tone | safe |
|---|---|---|---|---|---|
| canned (trivial): the corpus's most common reply | 2.03 | 1.17 | 1.24 | 2.60 | 3.11 |
| nearest_reply (simple): brand reply of the most similar past tweet | 3.00 | 2.28 | 2.15 | 3.18 | 4.39 |
| **Relay agent** | **4.35** | **4.13** | **4.11** | **4.42** | **4.74** |

Pairwise, agent vs nearest_reply, each pair judged in both orders with a disagreement counted as a tie: 181 wins, 14 ties, 5 losses. The canned reply is the mode of the corpus, "Sorry! There was an issue with Live TV last night, but it's now fixed: <link> Thanks for bearing with us!", which is a strawman; the nearest-reply baseline is the real comparison, and it is not weak. It is a genuine human reply to a genuinely similar tweet.

### Dev/test check

Intent accuracy and triage for the agent on the two strata:

| split | rows | intent accuracy | macro-F1 | unsafe_auto_rate | auto_coverage |
|---|---|---|---|---|---|
| dev (stratified) | 120 | 0.783 | 0.760 | 0.000 | 0.750 |
| test (uniform) | 80 | 0.875 | 0.849 | 0.000 | 0.825 |

## Judge validation

The judge is a different model from the generator (Opus 5 judging Sonnet 5 drafts) because judges rate their own family's outputs higher. Each pairwise comparison runs in both orders. Against the 60 blind human ratings:

| | n replies | Spearman | quadratic-weighted kappa | within 1 point | exact |
|---|---|---|---|---|---|
| all 60 rated rows | 120 | 0.746 | 0.657 | 0.875 | 0.350 |
| held-out rows 30 to 59 | 60 | 0.737 | **0.607** | 0.833 | 0.317 |

The rubric was tuned once, on rows 0 to 29 only, so the held-out row is the number to trust: substantial agreement by the usual bands, with the judge and the human agreeing within one point five times out of six. Two more checks: re-scoring 30 replies with "Be strict." appended to the rubric gives kappa 0.88 against the default run (prompt-perturbation stability, not sampling noise, since responses are cached), and 1.5% of pairwise verdicts flip when A and B are swapped.

Where the judge still disagrees with the human: it is kinder to the nearest-reply baseline (judge mean 2.93, human 2.10). A human penalises a fluent reply addressed to the wrong customer or about the wrong show more than the judge does, even after the rubric was told to.

## The iteration

The first full run (v1) produced the numbers on the left. Its error file was read, four causes were found, and one round of changes was made. `results/iterations.json` records both versions.

| | v1 | v2 | what changed |
|---|---|---|---|
| unsafe_auto_rate | 0.526 | 0.000 | in 7 of 10 misses the reply already sent the customer to phone/chat for an account lookup, but the action said auto; the policy now defines that as escalation and a hard rule enforces it |
| auto_coverage | 0.840 | 0.780 | cost of the above, plus dropping the confidence gate that was escalating ambiguous tweets where a clarifying question is the right auto reply |
| triage precision | 0.281 | 0.432 | |
| agent intent accuracy | 0.790 | 0.820 | sharper definitions for feature_question, content_availability, content_request and product_feedback (feature requests had been going to content_request) |
| judge: agent vs nearest_reply | 4.30 vs 3.86 | 4.35 vs 3.00 | the judge had been shown the nearest reply as evidence item 1, so the nearest-reply baseline was grading itself as grounded; evidence is now retrieved replies 2 to 4 for every system |
| judge kappa vs human | 0.455 | 0.657 (0.607 held out) | rubric: wrong customer, title, device or problem scores 1 on grounded and resolves |

On the dev split the intent change was slightly negative (0.800 to 0.783) and on the test split strongly positive (0.775 to 0.875). The confidence intervals overlap almost entirely, so the honest reading is that the intent change is roughly neutral and the gap between splits is sampling noise on 80 rows. The triage and judge changes moved every split the same way. No further tuning was done.

## Failure analysis

Real rows from `results/errors.jsonl` (v2) and the blind rating sheet.

1. **Learned deflection.** 19 of the 25 false escalations are replies that route to phone or chat when a published answer exists. "The fact i see the exact same commercial on every commercial break" gets "please give us a call/chat so that we can take look", while the brand's own reply explains ad preferences and links to them. Hypothesis: 58% of corpus replies link out and a meaningful share route to phone or chat, so the model has learned that deflection is on-brand; the human-channel hard rule then correctly turns each of these into an escalation, converting a quality problem into a coverage problem. The lever is the evidence, not the rule: prefer substantive replies when retrieving, and have the judge penalise hand-offs where the evidence shows a direct answer.

2. **The catalogue boundary.** 9 of the 36 intent errors are content_availability vs content_request vs feature_question. "Hey you're 1 season behind on #fixerupper" is availability (the brand carries the show) but was called a request; "how long does the latest episode take to show up on your service" is a feature question but was called availability. Hypothesis: the distinction depends on whether the service carries the title, which is knowledge about the catalogue, not about the tweet. The labeller had the brand's reply to settle it; the model does not. A catalogue lookup, or collapsing the two content intents, would remove most of these.

3. **Invented specifics.** In the v1 rating sheet the agent answered "why don't we get the new [episode]... we have to wait a week" with "The show has been on a 2-week hiatus. It'll be back on 10/26", a date that exists nowhere in the evidence, and answered a no-commercials complaint with "Our team is aware of this and looking into whether or not this is intended". Hypothesis: evidence replies contain concrete dates and status claims about other shows, and the model transplants them. The judge's grounded dimension catches some of this (agent grounded 4.13, not 5), which is why the rubric anchors grounded on traceability to a specific evidence reply.

4. **Sarcasm and venting read literally.** "Thanks for adding that 'wake up the kids' noise at app start up. Works perfectly!" is a complaint the brand recognised as one; the agent labelled it praise_or_thanks. "I hope we get more #FutureMan... he makes it THAT MUCH BETTER!" is praise labelled as a content request. Hypothesis: the model reads the tweet in isolation, and the tone cues (quotation marks, exclamation after a complaint) are weak signals against strong lexical ones. Rare in the golden set (3 rows) but each one produces a tone-deaf auto reply.

5. **Stated prior effort ignored.** "Been tryin the last two days... Done all troubleshooting suggested and still not working" got "Let's try: <link>" in v1. "I've called twice now" got a generic apology. Hypothesis: retrieval returns the standard troubleshooting reply as evidence and the model follows evidence over the tweet. The v2 `repeat_unresolved` rule now escalates the explicit cases, at the cost of 4 false escalations where "no one was there" describes chat wait time, not a failed resolution.

The baseline's own dominant failure is worth naming because it explains the pairwise result: the nearest reply is often a fluent answer to a different tweet, complete with someone else's first name ("Hey, Antonio!"). Lexical similarity finds tweets about the same show or the same device, not the same problem.

## What is misleading about the headline number

The headline is "zero missed escalations at 78% coverage". Reasons not to take it at face value:

- **The policy, the prompt and the labels have one author.** The escalation policy was written, then given to the model verbatim, then used to label the golden rows. Agreement between the model and the labels partly measures how well the model follows a paragraph it was shown, not whether the paragraph is what hulu_support would want. A second annotator applying their own judgement is the missing experiment.
- **19 positives.** Recall of 1.000 on 19 cases has a 95% lower bound around 0.82. One more missed case would read as 0.947. The metric is real but its resolution is coarse.
- **The v2 fixes were chosen after reading v1's errors on these same 200 rows.** The dev/test split limits the damage, but the test split is 80 rows with 6 escalations, and the fixes were policy and rule changes that generalise by construction only if the policy is right.
- **Coverage counts decisions, not resolutions.** An auto reply that asks "which device are you on?" counts as covered. Nothing here measures whether the customer's next tweet was happier; the dataset has those follow-ups and they were not used.
- **Reply quality is an LLM's opinion, validated against one person's opinion.** Kappa 0.61 is substantial, not almost-perfect, and the judge is systematically kinder to the retrieval baseline than the human was.
- **The held-out pool is two weeks of late November 2017.** The same live-sports outage (SEC championship weekend) appears in both the corpus and the pool, so retrieval looks more useful than it would on a genuinely new incident.
- **English only, text only.** The pool was filtered to English, and tweets whose content is a screenshot are reduced to `<link>`. Real traffic is messier.
- **The canned baseline is a strawman.** It exists because a trivial baseline was required; the nearest-reply baseline is the comparison that means something, and the agent's 181 to 5 pairwise result against it is the number to quote.
- **Nothing was sent to a customer.** Every reply was graded, none was delivered.

## What I would do next with one more week

1. A second annotator on all 200 rows and the 60 ratings, and report Cohen's kappa between annotators before quoting any model number. This is the single biggest gap.
2. Use the follow-up turns (present on 99.7% of threads) as an outcome signal: did the customer's next tweet resolve, repeat, or escalate in tone? That turns "resolves" from a judged quantity into a measured one.
3. Retrieval ablation: sentence embeddings and a reranker vs TF-IDF, measured by whether the retrieved brand reply's approach matches the real reply's approach, not by cosine.
4. Filter evidence to substantive replies and add a judge dimension that penalises hand-off when the evidence shows a direct answer. This targets failure mode 1, which now costs 19 points of coverage.
5. Grow the golden set to 500 with the uniform stratum only, so that the real distribution drives the numbers and per-class F1 for rare intents stops resting on ten rows.
6. Replace the flat hard rules with a cost model: expected cost of a wrong auto reply per reason vs cost of human review, and choose the operating point on the coverage-risk curve explicitly, in the spirit of the learning-to-defer literature.
7. Shadow mode: run on live traffic, never send, and compare the drafts with what the human agents actually sent.

## Decision log

- Picked hulu_support over SpotifyCares after computing hand-off share per brand. Spotify's modal reply is a DM request; an agent grounded on it learns to deflect.
- Replaced URLs with a `<link>` token instead of deleting them, so "try these steps: <link>" reads correctly and the model learns that the brand links out, without ever being able to invent a URL.
- Time-based split (last 15% by date) rather than random, so that retrieval cannot see the future and the golden set resembles new traffic.
- TF-IDF retrieval, no embedding model. Zero heavy dependencies, reproducible in seconds, and the failure analysis shows the bottleneck is what the model does with evidence, not evidence recall.
- One LLM call per tweet doing intent, reply and triage together, rather than three. Cheaper and the reply and the decision should be consistent with each other. Cost: three points of intent accuracy vs the dedicated classifier.
- Hard rules only ever move auto to escalate. The model cannot talk its way past a rule.
- Confidence gate removed in v2. The sweep showed it traded coverage for nothing once the policy rules were in place.
- Escalation reasons are a fixed enum of six, so that they can be counted, not free text.
- Judge is Opus 5, generator is Sonnet 5, to avoid self-preference bias. Pairwise judging runs both orders and treats a flip as a tie.
- Judge evidence excludes the top-1 retrieved reply for every system, because that reply is the nearest-reply baseline's output.
- Rubric calibrated on 30 rated rows, reported on the other 30, because 60 rows do not allow a proper split and reporting in-sample agreement would be misleading.
- The "self-consistency" check is reported as prompt-perturbation stability. With a response cache, a literal repeat is trivially identical.
- LLM cache is committed (29 MB) so that reproduction needs no credentials. CI reproduces and diffs the metrics file on every push.
- API spend is metered with a hard $1 cap. Total API spend for this repo: $0.0001 (one verification call). Every bulk run went through the CLI backend.
- Bootstrap CIs pin the label set per resample so that a resample missing a class does not average over fewer classes.
- Banking77 not used; the intents had to come from this brand's traffic.
- No comments in the code. Names and small functions carry the meaning; this README carries the reasoning.

## Borrowed and cited

- Dataset: Customer Support on Twitter, Kaggle, uploaded by thoughtvector. https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter . Loaded from the Hugging Face mirror `SunidhiSriram/twcs`, which is the same `twcs.csv`.
- Thread reconstruction from the `response_tweet_id` / `in_response_to_tweet_id` link structure follows the approach in Feigenblat et al., TWEETSUMM, Findings of EMNLP 2021. https://aclanthology.org/2021.findings-emnlp.24/
- Retrieval vs generation on this corpus: Hardalov, Koychev and Nakov, Towards Automated Customer Support, 2018. https://arxiv.org/abs/1809.00303
- Judge design: Zheng et al., Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena, 2023 (position swap, agreement measured against humans). https://arxiv.org/abs/2306.05685 . Liu et al., G-Eval, 2023 (form-filling numeric rubric). https://arxiv.org/abs/2303.16634
- Judge bias: Ye et al., Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge, 2024. https://arxiv.org/abs/2410.02736 . Self-Preference Bias in LLM-as-a-Judge, 2024. https://arxiv.org/abs/2410.21819
- Agreement metrics: quadratic-weighted Cohen's kappa for ordinal scores, with the usual bands (0.4 to 0.6 moderate, 0.6 to 0.8 substantial). https://arxiv.org/abs/2603.06865
- Escalation as deferral: Mozannar and Sontag, Consistent Estimators for Learning to Defer to an Expert, ICML 2020. https://arxiv.org/abs/2006.01862
- Intent taxonomy references: Casanueva et al., Efficient Intent Detection with Dual Sentence Encoders (Banking77), 2020. https://arxiv.org/abs/2003.04807 . Zhang, Wang and Shang, ClusterLLM, 2023. https://arxiv.org/abs/2305.14871
- Libraries: pandas, scikit-learn, numpy, anthropic SDK, pytest, ruff, uv. Models: Claude Sonnet 5 (agent, classifier, weak labels), Claude Opus 5 (judge).
- Written with an AI coding assistant. Every prompt, rule and threshold in the repo was read and is explained above.

## Layout

```
relay/        data.py  intents.py  retrieve.py  agent.py  llm.py  judge.py  golden.py  evaluate.py
data/brand    committed hulu_support threads (13,741) and the time split
data/golden   candidates, golden.jsonl, blind rating sheet, key, human ratings
data/intents  1,500 weak labels used to train the tfidf_lr baseline
data/cache    every LLM prompt and response, plus the API spend ledger
results/      metrics.json, errors.jsonl, agent_outputs.jsonl, iterations.json
tests/        one small test per module, no network
```
