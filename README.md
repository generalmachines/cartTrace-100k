# CartTrace-100k

101,416 journeys of an AI shopping agent (Qwen3.8-27B) buying from 1,044 stand-in stores built from open
product catalogs, end to end through checkout and payment, over the agentic commerce protocols: UCP (MCP and REST), ACP,
AP2, MPP, Shop Pay and WebMCP. Payments use test tokens only; nothing was bought on a real store.

This repository lets you do two things with the dataset:

1. **Replay a trace** in the store it was recorded in and check it ends exactly where the dataset says.
2. **Put your own model in the agent's seat** on the same store, task and shopper, and compare.

## Setup

You need [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
git clone https://github.com/generalmachines/cartTrace-100k && cd cartTrace-100k
uv sync --extra dataset
```

Download the dataset from [Hugging Face](https://huggingface.co/datasets/generalmachines/CartTrace-100k) into
`dataset/` (the `data/` tables) and `worlds/` (the stores):

```bash
uvx --from huggingface_hub hf download generalmachines/CartTrace-100k --repo-type dataset --include "data/*" --local-dir dataset
uvx --from huggingface_hub hf download generalmachines/CartTrace-100k --repo-type dataset --include "worlds/*" --local-dir .
```

## Look at a trace

The dataset is plain parquet: `messages` (one row per journey), `turns`, `wire`, `reference` and `stores`, joined on
`episode_id` (`stores` on `world_id`).

```python
import pyarrow.dataset as ds

turns = ds.dataset("dataset/data/turns", format="parquet")
rows = turns.to_table(filter=ds.field("episode_id") == "ep-00079e641081").to_pylist()
for t in sorted(rows, key=lambda r: r["seq"]):
    print(t["actor"], t["kind"], (t.get("text") or t.get("tool") or "")[:100])
```

## Replay a trace

```bash
uv run carttrace replay ep-00079e641081 --dataset dataset --worlds worlds
```

```
episode   ep-00079e641081
ended     cart_ready (trace: cart_ready)
calls     7 of 7 recorded model calls replayed
events    same chain head
state     same hash
REPRODUCED
```

No model, GPU or network needed; it takes seconds. The agent's model is replaced by the trace's own recorded
answers, and everything else is the real code: the store, the shopper, the wallet and the payment simulator. The
store has to reach the same final state hash and event chain as the trace.

WebMCP traces drive the storefront in a browser: `uv sync --extra dataset --extra browser` and
`uv run playwright install chromium` first.

## Run your own agent

```bash
uv run carttrace rerun ep-00079e641081 --dataset dataset --worlds worlds \
    --base-url http://127.0.0.1:8000/v1 --model your-model
```

Any OpenAI-compatible chat endpoint with tool calling works (vLLM, llama.cpp, Ollama, or a hosted API with
`--api-key-env`). The row keeps its store, task, shopper, protocol, payment method, faults and seed; your agent gets
the same prompt and tools. The new trace is written under `var/replays/reruns/` in the same format.

## Good to know

- The dataset hides each shopper's email and phone behind placeholders; a replay puts the persona's own back
  (personas are fictional). When an agent wrote a contact detail in a way the replay cannot guess, it reports
  NOT REPRODUCED.
- Each world names the real store it stood in for as `withheld`; it still replays under its published id.
- The dataset shows each store under a `ct-<serial>.example` domain, and its world keeps the domain it was served
  under; a replay from the dataset spells the first as the second wherever the agent copied it into a call.
- Code comments cite internal design decisions by number (D13, D102, ...) and name commands of the pipeline that
  collected the dataset. Neither that decision log nor those commands ship here; this kit's commands are `replay`
  and `rerun`.
- `uv run python tools/fetch_specs.py` downloads the Google Pay and Shop Pay handler specs, which we may not
  redistribute. Replays do not need them.

## Licence

The code is Apache 2.0. Vendored protocol specifications keep their own licences; see NOTICE.
The dataset is CC BY 4.0; the product data in it keeps the licences of its source catalogs (see the dataset card).

## Citation

The dataset and its paper, *CartTrace-100k: How Shopping Agents Fare Across Stores, Categories and Commerce
Protocols*, are described on the [dataset card](https://huggingface.co/datasets/generalmachines/CartTrace-100k).

```bibtex
@misc{carttrace100k,
  title     = {CartTrace-100k: How Shopping Agents Fare Across Stores, Categories and Commerce Protocols},
  author    = {Pratham Bhatnagar and Deepanshu Pal},
  year      = {2026},
  publisher = {General Machines},
  url       = {https://huggingface.co/datasets/generalmachines/CartTrace-100k}
}
```
