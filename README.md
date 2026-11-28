# potato-not-potato

Benchmark structured receipt extraction on a locally served LLM (via `llama.cpp` `llama-server`) using the OpenAI Chat Completions API.

## Usage

Start `llama-server` with `inkling-2-mini` (defaults to `http://localhost:8080`), then run:

```bash
uv run potato
```

### Options

```bash
uv run potato \
  --base-url http://localhost:8080/v1 \
  --model inkling-2-mini \
  --receipt data/img20260903_15395748.txt \
  --ground-truth data/ground-truth/img20260903_15395748.json \
  --output results
```

- **Performance metrics**: Extracts server-side prefill (`prompt_per_second`) and generation (`predicted_per_second`) token rates from `llama.cpp`'s `timings` payload, with prompt KV caching disabled by default (`--cache-prompt` to re-enable) so full prefill throughput is measured consistently.
- **Accuracy metrics**: Order-preserving alignment + exact-match comparison across item fields (`name`, `quantity`, `unit_price`, `price`) and receipt totals (`subtotal`, `tax`, `total`).
- **Output files**: Passing `--output results` (or `--output results.json` — any file extension is ignored) appends a local timestamp to the stem and writes both JSON and Markdown reports with the same base name (e.g., `results_<YYYY-MM-DD_HH:MM.SS>.json` and `results_<YYYY-MM-DD_HH:MM.SS>.md`).
