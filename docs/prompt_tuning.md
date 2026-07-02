# Prompt Tuning

Infrastructure for iterating the **VLM prompt** against a labelled evaluation set,
closing the loop:

```
prompt ─► run VLM on labelled cases ─► score ─► propose next prompt ─► …
```

The "propose next prompt" step is a pluggable **port**: today a human edits the
prompt in the terminal, or an LLM iterates via a file bridge; an online LLM API
drops in by implementing one method. The reusable, fully-tested core lives in
[`src/moment_to_action/prompt_tuning/`](../src/moment_to_action/prompt_tuning);
the runnable driver is
[`scripts/tune_vlm_prompt.py`](../scripts/tune_vlm_prompt.py) (mirrors the
`benchmark_*.py` scripts — heavy/interactive, coverage-exempt).

## Why this shape (efficiency)

Naive `prompt → result → new prompt` throws away everything except the last
score. Instead the proposer sees the whole **scored trajectory** and the
**failing cases**, so both humans and LLMs hill-climb with memory
([OPRO](https://arxiv.org/abs/2309.03409)-style optimization by prompting):

- **Failure-focused feedback** — the proposer is handed the worst failing cases
  (question, expected answer, what the model actually said), not just an
  aggregate number. This is what makes each iteration targeted.
- **Optimization trajectory** — prior attempts are shown worst-score-first,
  best-last, so the LLM anchors on the strongest example and tries to beat it.
- **Response caching** — VLM inference is slow, so responses are cached by
  `(prompt content hash, case id)`. Re-proposing a prompt already tried (even
  across generations) is free; only failed cases are re-run.

## Components

| Type | Role |
|------|------|
| `PromptCandidate` | The thing being tuned: `system_prompt` + `task_template` (+ lineage). `compose(question)` folds both into one user-prompt string. |
| `EvalCase` / `EvalDataset` | A labelled example (base64 frames + question + expected label/keywords) and a collection of them. |
| `Scorer` | `(response, case) → [0,1]`. Ships `KeywordRecallScorer` and `LabelMatchScorer`. |
| `ResponseTarget` | "Run the pipeline": `(candidate, case) → response`. `VLMResponseTarget` wraps a GGUF VLM today; a `Pipeline`-backed target can replace it later without touching anything else. |
| `PromptRunner` | Evaluates a candidate over a dataset into an `EvalReport` (per-case results + aggregates), with the response cache. |
| `PromptProposer` | **The port.** `TuningState → next PromptCandidate`. `HumanProposer` and `LLMProposer` ship. |
| `ChatClient` | The LLM sub-port used by `LLMProposer`: `complete(system, user) → reply`. |
| `PromptTuner` | The loop: seed → evaluate → propose → repeat, until `max_iterations`, `target_score`, or `StopTuning`. |
| `TrajectoryStore` | Persists `trajectory.jsonl`, `best.json`, `best_prompt.txt` under a run dir. |

## Data flow

```
seed PromptCandidate
      │
      ▼
PromptTuner.run ──► PromptRunner.evaluate ──► ResponseTarget.generate ──► VLM
      │                    │                         (compose prompt, run)
      │                    └──► Scorer.score ──► EvalReport (+ cache)
      ▼
TuningState (scored trajectory) ──► PromptProposer.propose ──► next PromptCandidate
      │                                   (human edits │ LLM via ChatClient)
      └──► TrajectoryStore (jsonl + best) ; on_report callback (live table)
```

Note the system prompt is composed **into the per-request user prompt**, and the
VLM is constructed with an empty system prompt. Swapping candidates therefore
never reloads weights or restarts `llama-server`.

## Running it

```bash
# Human-in-the-loop (edit the prompt each round in the terminal):
uv run python scripts/tune_vlm_prompt.py --mode human --max-iterations 8

# LLM-in-the-loop with the offline file bridge (no API needed):
uv run python scripts/tune_vlm_prompt.py --mode llm --bridge-dir ./bridge
#   → writes bridge/request_NN.md; paste into any chat model;
#     save the JSON reply to bridge/response_NN.txt; press Enter.

# Tune only one application, score by exact label match, stop early at 0.95:
uv run python scripts/tune_vlm_prompt.py --apps violence_detection \
    --scorer label --target-score 0.95
```

The eval set is the same scene set as `benchmark_vlms.py`, so a tuned prompt can
be compared directly against the generic benchmark prompt. Real clips are used
when `--video-dir <dir>/<scene>.mp4` exists, otherwise synthetic frames are
rendered. Requires `llama_server_path` in the M2A config (or `--server-path`).

Output (`prompt_runs/<timestamp>/` by default):

- `trajectory.jsonl` — one scored candidate per line (prompt + per-case results).
- `best.json` — the winning candidate and its full report.
- `best_prompt.txt` — the winning prompt, ready to copy into the registry.

## Manual (human) tuning walkthrough

This is the step-by-step for `--mode human` (the default). No LLM or bridge
files are involved — you are the proposer, deciding the next prompt from the
evidence the tool shows you each round.

1. **Launch it.**

   ```bash
   uv run python scripts/tune_vlm_prompt.py --mode human --model moondream2 \
       --max-iterations 5 --apps violence_detection
   ```

   - `--model` picks a VLM already resolvable in the model cache (downloads
     it via `ModelManager` if not present).
   - `--apps` narrows the eval set to one application while you're getting a
     feel for the loop; omit it to tune across all five.
   - Leaving `--video-dir` unset renders **synthetic frames** (solid-color
     PIL scenes matching each label) instead of requiring real clips — this
     is the fast path for iterating on prompt *mechanics* (format compliance,
     keyword usage, refusal to hedge) rather than raw visual recognition.

2. **Read the seed report.** The tool evaluates your `--seed-system` /
   `--seed-template` first (defaults are a generic "analyze the scene"
   prompt) and prints a table: mean score, pass rate, per-app breakdown, plus
   up to 5 of the worst-scoring cases with the model's actual response (or
   its error). This table is the entire signal loop — read the failing
   responses, not just the score.

3. **Diagnose the failure mode**, not the symptom. Typical patterns to look
   for in the printed responses:
   - **Hedging / no verdict** — model rambles about the scene but never
     commits to YES/NO → tighten the system prompt to demand the verdict
     word first.
   - **Right answer, wrong vocabulary** — model says "there is fighting"
     but the scorer looks for the literal keyword "yes" → either adjust the
     `recall_keywords` (the dataset, not the prompt) or force an
     answer-format contract in the prompt (e.g. `Answer: YES/NO`).
   - **Systematically wrong on one app only** — check `per_app` scores; a
     single low app means the *task framing* doesn't transfer, not that the
     prompt is bad overall.
   - **Empty / errored responses** — a runtime problem (model, server,
     image encoding), not a prompt problem; fix that before tuning further.

4. **Answer the three prompts** the tool asks for, one per line, finished
   with a lone `.`:
   - `New system_prompt` — blank reuses the current best; type a full
     replacement otherwise (don't diff against the old one, just restate it).
   - `New task_template` — must eventually place the question; either use
     the literal placeholder `{question}` (see `QUESTION_PLACEHOLDER`) or
     leave blank to keep the current template and only change the system
     prompt.
   - `Rationale` — one line, free text, purely for your own trajectory log
     (`trajectory.jsonl`); not sent to the model.

   Enter `/stop` at any prompt to end the run early and keep the best
   candidate found so far.

5. **Repeat.** Each round's report shows the new candidate's score plus its
   own failing cases — keep addressing the single worst failure mode rather
   than rewriting the whole prompt every round; large single-round rewrites
   make it hard to tell which change caused which score delta.

6. **Stop when either**: `--max-iterations` is reached, you hit the score you
   want and type `/stop`, or blank answers on all three fields ("no change")
   — the tool treats "nothing to try" as a stop signal too.

7. **Collect the result** from the output directory
   (`prompt_runs/<timestamp>/` by default): `best_prompt.txt` has the winning
   `system_prompt` + `task_template` ready to paste into the model's real
   `system_prompt=` construction; `trajectory.jsonl` has every round's full
   report for a post-hoc writeup (e.g. "tuned vs. generic prompt" comparison).

## Wiring an online LLM (the port)

`LLMProposer` depends only on the `ChatClient` protocol:

```python
class ChatClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...
```

Implement it for your endpoint and return it from `_build_chat_client` in the
driver (or construct `LLMProposer(client=...)` directly):

```python
@attrs.define
class MyHttpChatClient:
    base_url: str
    api_key: str
    def complete(self, system: str, user: str) -> str:
        resp = httpx.post(self.base_url, json={...}, headers={...})
        return resp.json()["choices"][0]["message"]["content"]
```

`NotConfiguredChatClient` is the placeholder that errors clearly until a real
client is supplied; `FileBridgeChatClient` is the manual/offline bridge.

## Extending

- **New scorer** (e.g. LLM-as-judge): implement `Scorer` (`name` + `score`).
- **New target** (e.g. run the full `Pipeline` instead of the raw VLM):
  implement `ResponseTarget.generate`; nothing else changes.
- **New proposer** (e.g. a local mutation heuristic): implement
  `PromptProposer.propose`; raise `StopTuning` to end the loop.
