# Triage foundation model benchmark, 2026-09-26

Purpose: choose the OpenRouter foundation model for the first real task (ManyFails candidate triage, decision 3).
Input: the ManyFails labelled fixture of 30 real headlines (`src/test/fixtures/triage/headlines.json`, 14 true positives with a product name), the production triage prompt, the production JSON schema as a strict `json_schema` response format, temperature 0, 400 output tokens, title and description only (no scraped page text).
Scoring: `is_failure` agreement, `kind` agreement, agreement with the ManyFails acceptance rule (confidence at least 0.6, a product or company named, kind not watch or not_a_failure, incident or is_failure), product-name match on the 14 positives, wall-clock p50/p95 per call from this Mac, and cost from reported usage at OpenRouter list prices. Errors are calls that failed or returned output outside the schema.
Runs 1 and 2 are the first pass; run 3 repeats the four leaders to check stability. The fixture is small, so a one-item difference is noise.

| Run | Model | Errors | is_failure | kind | accept rule | product | p50 s | p95 s | USD per 1,000 calls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `deepseek/deepseek-v3.2` | 0 | 28/30 | 22/30 | 26/30 | 13/14 | 7.0 | 11.59 | 0.255 |
| 1 | `deepseek/deepseek-v4-flash` | 8 | 22/22 | 18/22 | 19/22 | 8/8 | 5.8 | 14.68 | 0.063 |
| 1 | `deepseek/deepseek-v4.1-flash` | 1 | 27/29 | 23/29 | 25/29 | 13/13 | 1.51 | 3.14 | 0.171 |
| 1 | `anthropic/claude-haiku-4.5` | 0 | 29/30 | 26/30 | 29/30 | 14/14 | 1.82 | 3.93 | 1.741 |
| 2 | `openai/gpt-5.4-nano` | 0 | 28/30 | 25/30 | 29/30 | 14/14 | 1.92 | 2.32 | 0.277 |
| 2 | `openai/gpt-6-luna` | 0 | 28/30 | 26/30 | 29/30 | 13/14 | 3.27 | 6.01 | 0.176 |
| 2 | `openai/gpt-5.4-mini` | 0 | 28/30 | 24/30 | 29/30 | 14/14 | 1.39 | 1.77 | 1.03 |
| 2 | `google/gemini-3.5-flash-lite` | 0 | 28/30 | 25/30 | 29/30 | 14/14 | 0.94 | 1.06 | 0.534 |
| 2 | `google/gemini-3.8-flash` | 11 | 19/19 | 18/19 | 19/19 | 6/6 | 4.46 | 5.82 | 1.925 |
| 2 | `anthropic/claude-sonnet-5` | 0 | 26/30 | 22/30 | 27/30 | 14/14 | 2.69 | 4.62 | 4.704 |
| 3 | `anthropic/claude-haiku-4.5` | 0 | 29/30 | 26/30 | 29/30 | 14/14 | 1.73 | 2.68 | 1.741 |
| 3 | `openai/gpt-6-luna` | 0 | 28/30 | 27/30 | 29/30 | 13/14 | 3.4 | 5.38 | 0.179 |
| 3 | `openai/gpt-5.4-nano` | 0 | 28/30 | 23/30 | 27/30 | 12/14 | 1.89 | 2.35 | 0.277 |
| 3 | `google/gemini-3.5-flash-lite` | 0 | 28/30 | 24/30 | 28/30 | 14/14 | 1.0 | 1.21 | 0.532 |

Notes: `openai/gpt-5.4-nano`, `openai/gpt-6-luna`, `openai/gpt-5.4-mini`, the two Gemini models and `anthropic/claude-sonnet-5` refused the first pass only because the request asked OpenRouter to require parameter support (`provider.require_parameters`); run 2 repeats them without that flag. `google/gemini-3.8-flash` and `deepseek/deepseek-v4-flash` produced many outputs outside the schema under this setting and are excluded from consideration. `anthropic/claude-sonnet-5` scored below `anthropic/claude-haiku-4.5` on this fixture at 2.7 times the price.

Choice: `anthropic/claude-haiku-4.5` as the foundation. It had the best and most stable agreement (29/30, 26/30, 29/30, 14/14 in both runs), no schema failures, and a p50 under two seconds. `openai/gpt-6-luna` matched it on kind at about a tenth of the price but was slower and missed one product name; it is the natural cheap comparison arm once the platform routes live traffic.

Reproduction: the script lives outside the repository in the reviewer's session scratchpad (`triage-bench/bench.py`); it needs the ManyFails checkout and an OpenRouter key. Results files: `results-1.jsonl`, `results-2.jsonl`, `results-3.jsonl`.
