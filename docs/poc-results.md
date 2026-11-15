# CPU Inference Proof of Concept: Results

**Date:** 2026-11-15
**Script:** [`scripts/poc_inference.py`](../scripts/poc_inference.py)
**Raw output:** [`results/poc_llama_desktop.txt`](../results/poc_llama_desktop.txt), [`results/poc_phi_desktop.txt`](../results/poc_phi_desktop.txt), [`results/poc_qwen_desktop.txt`](../results/poc_qwen_desktop.txt)

## Goal

Test the project's core assumption: a quantized small language model can run through Ollama on a consumer CPU, with no GPU, fast enough to be usable.

## Test setup

| Item | Value |
|---|---|
| Machine | Desktop, Windows 10 |
| CPU | AMD64 Family 23 Model 113 (AMD Zen 2), 12 logical cores |
| RAM | 15.9 GB |
| GPU | Not used (checked by the script, see below) |
| Ollama | 0.40.16 |
| Python | 3.11.4 |
| Prompt | "Explain in three sentences what retrieval-augmented generation is." |
| Settings | `num_ctx=4096`, `max_tokens=256`, 5 runs per model |

Run 1 is a cold start: the model is unloaded first, so it includes load time. Runs 2 to 5 are warm and show steady-state speed.

Commands:

```powershell
python scripts/poc_inference.py --model llama3.2:3b --runs 5 | Tee-Object results\poc_llama_desktop.txt
python scripts/poc_inference.py --model phi4-mini --runs 5 | Tee-Object results\poc_phi_desktop.txt
python scripts/poc_inference.py --model qwen3.5:4b --think off --runs 5 | Tee-Object results\poc_qwen_desktop.txt
```

## Summary

| Model | Thinking | Memory | CPU-only | Cold load (s) | Cold total (s) | Warm TTFT (s) | Warm prefill (tok/s) | Warm gen (tok/s) | Warm total (s) | Answer tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| llama3.2:3b | model default | 2.39 GB | PASS | 10.94 | 20.94 | **2.12** | **530** | **14.5** | **9.43** | 106 |
| phi4-mini | model default | 2.89 GB | PASS | 9.16 | 18.96 | 2.15 | 171 | 12.1 | 10.10 | 96 |
| qwen3.5:4b | off | 2.93 GB | PASS | 24.30 | 41.67 | 2.21 | 153 | 9.0 | 13.91 | 105 |

The warm TTFT and gen tok/s columns are the script's own averages of runs 2 to 5. I calculated the warm prefill and warm total columns as the mean of the same runs. TTFT means time to first token.

## Per-run data

### llama3.2:3b

| Run | Load (s) | TTFT (s) | Prefill (tok/s) | Gen (tok/s) | Tokens | Total (s) |
|---|---|---|---|---|---|---|
| 1 (cold) | 10.94 | 13.63 | 62.8 | 14.5 | 106 | 20.94 |
| 2 | 0.01 | 2.13 | 511.9 | 14.6 | 106 | 9.38 |
| 3 | 0.00 | 2.13 | 531.7 | 14.6 | 106 | 9.38 |
| 4 | 0.00 | 2.11 | 537.0 | 14.6 | 106 | 9.36 |
| 5 | 0.00 | 2.11 | 538.8 | 14.2 | 106 | 9.60 |

### phi4-mini

| Run | Load (s) | TTFT (s) | Prefill (tok/s) | Gen (tok/s) | Tokens | Total (s) |
|---|---|---|---|---|---|---|
| 1 (cold) | 9.16 | 11.53 | 45.8 | 12.3 | 91 | 18.96 |
| 2 | 0.01 | 2.17 | 178.6 | 12.0 | 96 | 10.18 |
| 3 | 0.00 | 2.16 | 172.6 | 11.9 | 96 | 10.21 |
| 4 | 0.00 | 2.14 | 168.4 | 12.3 | 96 | 9.96 |
| 5 | 0.00 | 2.14 | 164.5 | 12.2 | 96 | 10.04 |

### qwen3.5:4b (thinking off)

| Run | Load (s) | TTFT (s) | Prefill (tok/s) | Gen (tok/s) | Tokens | Total (s) |
|---|---|---|---|---|---|---|
| 1 (cold) | 24.30 | 29.18 | 10.8 | 8.4 | 105 | 41.67 |
| 2 | 0.03 | 2.27 | 142.1 | 9.1 | 105 | 13.79 |
| 3 | 0.00 | 2.21 | 153.5 | 9.0 | 105 | 13.93 |
| 4 | 0.01 | 2.20 | 155.6 | 8.9 | 105 | 14.05 |
| 5 | 0.00 | 2.18 | 162.2 | 9.0 | 105 | 13.86 |

## Answer quality

All three models answered correctly in three sentences. These are the answers from run 1.

- **llama3.2:3b:** Correct but generic. It calls RAG "a type of AI model" and talks about a "knowledge graph", which is a little imprecise.
- **phi4-mini:** Clear and accurate. It describes the retrieve-then-generate flow and mentions grounding.
- **qwen3.5:4b:** The most complete and precise answer. It explains that retrieved context is fed into the model as input and names less hallucination as the benefit.

This is one prompt checked by eye, not a quality benchmark.

## Findings

1. **The core assumption holds.** All three models ran fully on the CPU (0% GPU memory) in under 3 GB of RAM. Each warm answer took 9 to 14 seconds.
2. **llama3.2:3b is the fastest.** It generated tokens about 20% faster than phi4-mini and about 60% faster than qwen3.5:4b. Its prefill was about 3 times faster than the others, which will matter once RAG adds retrieved context to the prompt.
3. **Warm time to first token is about the same (2.1 to 2.2 s) for all three.** With a prompt this short, that time is mostly fixed overhead rather than prompt processing.
4. **Cold starts are slow.** A cold start adds 9 to 28 seconds. qwen3.5:4b is the slowest (24 s load, 42 s total). The pipeline should keep the model loaded, for example with Ollama's `keep_alive`, instead of loading it for each request.
5. **Speed and answer quality trade off.** qwen3.5:4b gave the best answer but was the slowest. phi4-mini sits in the middle on both.
6. **Run-to-run variation is low.** Warm runs varied by less than 0.1 s in TTFT and 0.4 tok/s or less in generation speed, so 5 runs per model gives stable numbers.

## Limitations

- One short prompt, a single sentence. Real RAG prompts with retrieved chunks will be much longer, so prefill speed and TTFT will matter more.
- One machine. Results on other hardware will differ.
- Answer quality was judged on one answer per model, by eye.
- All models used Ollama's default quantization for each tag.

## Next steps

- Re-run with a RAG-sized prompt (for example 1,500 to 3,000 tokens of context) to measure prefill under realistic load.
- Run the same benchmark on a second machine (for example a laptop) to see how much the hardware changes the results.
- Start with llama3.2:3b as the default model for speed. Keep qwen3.5:4b as the option for better answers, and decide between them once retrieval quality can be tested end to end.
