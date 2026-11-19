# CPU Inference Proof of Concept: Results

**Date:** 2026-11-15
**Script:** [`scripts/poc_inference.py`](../scripts/poc_inference.py)
**Raw output (desktop):** [`poc_llama_desktop.txt`](../results/poc/poc_llama_desktop.txt), [`poc_phi_desktop.txt`](../results/poc/poc_phi_desktop.txt), [`poc_qwen_desktop.txt`](../results/poc/poc_qwen_desktop.txt)
**Raw output (laptop):** [`poc_llama_laptop.txt`](../results/poc/poc_llama_laptop.txt), [`poc_phi_laptop.txt`](../results/poc/poc_phi_laptop.txt), [`poc_qwen_laptop.txt`](../results/poc/poc_qwen_laptop.txt)

## Goal

Test the project's core assumption: a quantized small language model can run through Ollama on a consumer CPU, with no GPU, fast enough to be usable.

## Test setup

The same script, prompt and settings were run on two machines.

| Item | Desktop | Laptop |
|---|---|---|
| OS | Windows 10 | Windows 11 Home |
| CPU | AMD Ryzen 5 3600 (Zen 2, desktop), 6 cores / 12 threads | AMD Ryzen 3 5425U (Zen 3, mobile), 4 cores / 8 threads |
| RAM (usable, as reported) | 15.9 GiB | 7.3 GiB (8 GB installed; part is reserved for the integrated GPU) |
| GPU | Not used (checked by the script, see below) | Not used (integrated Radeon graphics, checked by the script) |
| Ollama | 0.40.16 | 0.40.16 |
| Python | 3.11.4 | 3.14.2 |
| Conditions | Mains power, Windows power mode "Best performance" (power plan: High performance), background apps closed | Mains power (charging), power plan "HP Recommended", VS Code open during the runs |

Both machines:

| Item | Value |
|---|---|
| Prompt | "Explain in three sentences what retrieval-augmented generation is." |
| Settings | `num_gpu=0` (CPU only), `num_ctx=4096`, `num_predict=256`, `temperature=0`, `seed=42`, 5 runs per model |

Run 1 is a cold start: the model is unloaded first, so it includes load time. Runs 2 to 5 are warm and show steady-state speed.

Commands:

```powershell
python scripts/poc_inference.py --model llama3.2:3b --runs 5 | Tee-Object results\poc\poc_llama_desktop.txt
python scripts/poc_inference.py --model phi4-mini --runs 5 | Tee-Object results\poc\poc_phi_desktop.txt
python scripts/poc_inference.py --model qwen3.5:4b --think off --runs 5 | Tee-Object results\poc\poc_qwen_desktop.txt
```

The laptop used the same commands, writing to `poc_*_laptop.txt`. Windows PowerShell's `Tee-Object` saves files as UTF-16, so the logs were converted to UTF-8 after capture.

## Summary

### Desktop

| Model | Params | Quant | Thinking | Download size (GB) | RAM when loaded (GiB) | CPU-only | Cold load (s) | Cold total (s) | Warm TTFT (s) | Warm prefill (tok/s) | Warm gen (tok/s) | Warm total (s) | Answer tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama3.2:3b | 3.2B | Q4_K_M | n/a | 2.0 | **2.39** | PASS | 10.94 | 20.94 | **2.12** | **530** | **14.5** | **9.43** | 106 |
| phi4-mini | 3.8B | Q4_K_M | n/a | 2.5 | 2.89 | PASS | **9.16** | **18.96** | 2.15 | 171 | 12.1 | 10.10 | 96 |
| qwen3.5:4b | 4.2B | Q4_K_M | off | 3.3 | 2.93 | PASS | 24.30 | 41.67 | 2.21 | 153 | 9.0 | 13.91 | 105 |

### Laptop

| Model | Params | Quant | Thinking | Download size (GB) | RAM when loaded (GiB) | CPU-only | Cold load (s) | Cold total (s) | Warm TTFT (s) | Warm prefill (tok/s) | Warm gen (tok/s) | Warm total (s) | Answer tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama3.2:3b | 3.2B | Q4_K_M | n/a | 2.0 | **2.39** | PASS | **12.89** | **33.55** | **2.27** | **210** | **6.8** | **17.90** | 106 |
| phi4-mini | 3.8B | Q4_K_M | n/a | 2.5 | 2.89 | PASS | 22.48 | 42.14 | 2.35 | 65 | 5.3 | 20.56 | 96 |
| qwen3.5:4b | 4.2B | Q4_K_M | off | 3.3 | 2.93 | PASS | 39.71 | 74.31 | 3.25 | 33 | 4.1 | 28.92 | 105 |

**Bold** = best value in each column. Thinking is n/a for llama3.2:3b and phi4-mini because they have no thinking mode.

### Desktop vs laptop

All figures are warm-run means except the cold total.

| Model | Gen tok/s (desktop → laptop) | Laptop gen speed as % of desktop | Prefill tok/s (desktop → laptop) | Warm total s (desktop → laptop) | Laptop slowdown (warm total) | Cold total s (desktop → laptop) |
|---|---|---|---|---|---|---|
| llama3.2:3b | 14.5 → 6.8 | 47% | 530 → 210 | 9.43 → 17.90 | 1.9× | 20.94 → 33.55 |
| phi4-mini | 12.1 → 5.3 | 44% | 171 → 65 | 10.10 → 20.56 | 2.0× | 18.96 → 42.14 |
| qwen3.5:4b | 9.0 → 4.1 | 46% | 153 → 33 | 13.91 → 28.92 | 2.1× | 41.67 → 74.31 |

Where the model details come from:

- **Params and Quant:** `ollama show <model>`. All three tags are 4-bit `Q4_K_M` builds, so the comparison is like for like.
- **Download size:** the model's size on disk as shown by `ollama list` (decimal GB). The qwen3.5:4b download also includes a vision projector of about 333M parameters, which this test does not use. This likely explains why qwen3.5:4b uses less RAM when loaded than its download size, unlike the other two models (not yet verified).
- **RAM when loaded:** the `size` that Ollama's `/api/ps` endpoint reports while the model is loaded, which the script converts with 1024³ (GiB). It is Ollama's own figure for the loaded model: the weights, plus the KV cache for `num_ctx=4096`, plus working buffers. It does not count the Ollama server process or the Python script, and it is not an OS measurement of process memory. The script's raw output labels this "GB", but the value is in GiB. A larger `num_ctx` would make this figure bigger.
- **CPU-only:** the same `/api/ps` entry reported `size_vram` = 0, meaning 0% of the model was on the GPU.

The warm TTFT and gen tok/s columns are the script's own averages of runs 2 to 5, on each machine. I calculated the warm prefill and warm total columns as the mean of the same runs. TTFT means time to first token.

## Per-run data

### Desktop

#### llama3.2:3b

| Run | Load (s) | TTFT (s) | Prefill (tok/s) | Gen (tok/s) | Tokens | Total (s) |
|---|---|---|---|---|---|---|
| 1 (cold) | 10.94 | 13.63 | 62.8 | 14.5 | 106 | 20.94 |
| 2 | 0.01 | 2.13 | 511.9 | 14.6 | 106 | 9.38 |
| 3 | 0.00 | 2.13 | 531.7 | 14.6 | 106 | 9.38 |
| 4 | 0.00 | 2.11 | 537.0 | 14.6 | 106 | 9.36 |
| 5 | 0.00 | 2.11 | 538.8 | 14.2 | 106 | 9.60 |

#### phi4-mini

| Run | Load (s) | TTFT (s) | Prefill (tok/s) | Gen (tok/s) | Tokens | Total (s) |
|---|---|---|---|---|---|---|
| 1 (cold) | 9.16 | 11.53 | 45.8 | 12.3 | 91 | 18.96 |
| 2 | 0.01 | 2.17 | 178.6 | 12.0 | 96 | 10.18 |
| 3 | 0.00 | 2.16 | 172.6 | 11.9 | 96 | 10.21 |
| 4 | 0.00 | 2.14 | 168.4 | 12.3 | 96 | 9.96 |
| 5 | 0.00 | 2.14 | 164.5 | 12.2 | 96 | 10.04 |

#### qwen3.5:4b (thinking off)

| Run | Load (s) | TTFT (s) | Prefill (tok/s) | Gen (tok/s) | Tokens | Total (s) |
|---|---|---|---|---|---|---|
| 1 (cold) | 24.30 | 29.18 | 10.8 | 8.4 | 105 | 41.67 |
| 2 | 0.03 | 2.27 | 142.1 | 9.1 | 105 | 13.79 |
| 3 | 0.00 | 2.21 | 153.5 | 9.0 | 105 | 13.93 |
| 4 | 0.01 | 2.20 | 155.6 | 8.9 | 105 | 14.05 |
| 5 | 0.00 | 2.18 | 162.2 | 9.0 | 105 | 13.86 |

### Laptop

#### llama3.2:3b

| Run | Load (s) | TTFT (s) | Prefill (tok/s) | Gen (tok/s) | Tokens | Total (s) |
|---|---|---|---|---|---|---|
| 1 (cold) | 12.89 | 16.87 | 21.6 | 6.3 | 106 | 33.55 |
| 2 | 0.04 | 2.31 | 221.8 | 6.9 | 106 | 17.57 |
| 3 | 0.01 | 2.25 | 197.3 | 6.1 | 106 | 19.50 |
| 4 | 0.01 | 2.25 | 218.0 | 7.2 | 106 | 16.99 |
| 5 | 0.01 | 2.25 | 201.5 | 6.9 | 106 | 17.54 |

#### phi4-mini

| Run | Load (s) | TTFT (s) | Prefill (tok/s) | Gen (tok/s) | Tokens | Total (s) |
|---|---|---|---|---|---|---|
| 1 (cold) | 22.48 | 26.47 | 15.9 | 5.6 | 91 | 42.14 |
| 2 | 0.06 | 2.37 | 75.6 | 5.5 | 96 | 19.75 |
| 3 | 0.02 | 2.35 | 55.2 | 5.1 | 96 | 21.14 |
| 4 | 0.02 | 2.41 | 50.2 | 4.7 | 96 | 22.67 |
| 5 | 0.01 | 2.27 | 79.1 | 5.9 | 96 | 18.66 |

#### qwen3.5:4b (thinking off)

| Run | Load (s) | TTFT (s) | Prefill (tok/s) | Gen (tok/s) | Tokens | Total (s) |
|---|---|---|---|---|---|---|
| 1 (cold) | 39.71 | 43.96 | 12.6 | 3.4 | 105 | 74.31 |
| 2 | 0.06 | 3.92 | 14.8 | 4.0 | 105 | 30.04 |
| 3 | 0.02 | 3.95 | 13.6 | 4.1 | 105 | 29.83 |
| 4 | 0.02 | 2.54 | 54.2 | 4.2 | 105 | 27.66 |
| 5 | 0.01 | 2.61 | 48.7 | 4.1 | 105 | 28.13 |

## Answer quality

All three models answered correctly in three sentences. These are the answers from run 1.

- **llama3.2:3b:** Correct but generic. It calls RAG "a type of AI model" and talks about a "knowledge graph", which is a little imprecise.
- **phi4-mini:** Clear and accurate. It describes the retrieve-then-generate flow and mentions grounding.
- **qwen3.5:4b:** The most complete and precise answer. It explains that retrieved context is fed into the model as input and names less hallucination as the benefit.

The laptop gave exactly the same answers, word for word and with the same token counts, for all three models. With `temperature=0` and a fixed seed, the hardware changed only the speed, not the output. (The run 1 answers are compared here; see finding 13 for the cold/warm difference.)

This is one prompt checked by eye, not a quality benchmark.

## Findings

### Desktop

1. **The core assumption holds.** All three 4-bit (Q4_K_M) models ran fully on the CPU (0% GPU memory). Each used 2.39 to 2.93 GiB of RAM when loaded with a 4,096-token context, which is under a fifth of the machine's RAM. Each warm answer took 9 to 14 seconds.
2. **llama3.2:3b is the fastest.** It generated tokens about 20% faster than phi4-mini and about 60% faster than qwen3.5:4b. Its prefill was about 3 times faster than the others, which will matter once RAG adds retrieved context to the prompt. *Prefill figures here come from a very short prompt repeated identically, which Ollama may partly cache; RAG-sized prompts will give the reliable comparison.*
3. **Warm time to first token is about the same (2.1 to 2.2 s) for all three.** With a prompt this short, that time is mostly fixed overhead rather than prompt processing. At about 530 tok/s, llama3.2:3b's prompt should take around 0.04 s to process, so the cause of the remaining ~2 s is not yet known.
4. **Cold starts are slow.** A cold start adds 9 to 28 seconds. qwen3.5:4b is the slowest (24 s load, 42 s total). The pipeline should keep the model loaded, for example with Ollama's `keep_alive`, instead of loading it for each request.
5. **Speed and answer quality trade off.** qwen3.5:4b gave the best answer but was the slowest. phi4-mini sits in the middle on both.
6. **Run-to-run variation is low.** Warm runs varied by less than 0.1 s in TTFT and 0.4 tok/s or less in generation speed, so 5 runs per model gives stable numbers.

### Laptop compared with desktop

7. **The core assumption also holds on a low-end laptop.** All three models ran fully on the CPU (0% GPU memory) on a 4-core mobile CPU with 7.3 GB of RAM. RAM when loaded was the same as on the desktop (2.39 to 2.93 GiB), but on this machine that is a third to two fifths of total RAM, so there is much less room left for the rest of the pipeline (embedding model, vector store, the OS and other apps).
8. **The laptop is about twice as slow.** Generation ran at 44 to 47% of desktop speed for every model, and a warm answer took 18 to 29 seconds instead of 9 to 14. llama3.2:3b at 6.8 tok/s is still readable as a stream. qwen3.5:4b at 4.1 tok/s is slow.
9. **Prefill dropped more than generation.** Prefill fell to about 40% of desktop speed for llama3.2:3b and phi4-mini, and to about 22% for qwen3.5:4b. RAG prompts are long, so prefill is likely where the laptop will hurt most. *As in finding 2, these prefill figures come from a short, repeated prompt and should be confirmed with RAG-sized prompts.*
10. **The model ranking does not change.** llama3.2:3b is still the fastest, with about 28% faster generation than phi4-mini and 66% faster than qwen3.5:4b. Its prefill lead is larger on the laptop (210 vs 65 vs 33 tok/s).
11. **Cold starts are much worse.** Load time rose to 13 to 40 seconds, and a cold qwen3.5:4b answer took 74 seconds. Keeping the model loaded matters even more on weaker hardware.
12. **Laptop runs vary more.** phi4-mini's warm generation ranged from 4.7 to 5.9 tok/s. qwen3.5:4b's first two warm runs had a TTFT of about 3.9 s and a prefill of about 14 tok/s, then settled to about 2.6 s and 50 tok/s. The cause was not measured. Likely factors are the balanced power plan, the mobile CPU's power and heat limits, other open apps, and memory pressure with 7.3 GB of RAM. Treat the laptop's warm TTFT and prefill averages for qwen3.5:4b with caution.

### Both machines

13. **Cold and warm runs can give different answers.** phi4-mini's cold run produced 91 tokens and its warm runs 96, identically on both machines, despite `temperature=0` and a fixed seed. Because the difference repeats exactly, it comes from the model's state (cold vs warm), not hardware noise. The main benchmark will record every answer rather than assume repeats are identical.

## Limitations

- One short prompt, a single sentence. Real RAG prompts with retrieved chunks will be much longer, so prefill speed and TTFT will matter more.
- Two machines, and their test conditions were not the same. The laptop used a balanced power plan ("HP Recommended") rather than High performance, had VS Code open, and ran a newer Python. The Python version should not affect the timings, which come from Ollama, but the power plan and open apps may have slowed the laptop somewhat.
- Answer quality was judged on one answer per model, by eye.
- All models used Ollama's default quantization for each tag (Q4_K_M). Other quantizations, such as Q8_0, were not tested, and they would change both speed and memory use.
- Memory is Ollama's reported figure for the loaded model. Total system memory use during a run was not measured.
- Warm runs repeated the identical prompt, so Ollama may have reused cached prompt tokens. The script does not log the number of prompt tokens processed per run, so this could not be checked.

## Next steps

- Re-run with a RAG-sized prompt (for example 1,500 to 3,000 tokens of context) to measure prefill under realistic load.
- Find the source of the ~2 s fixed TTFT overhead before the RAG benchmarks, since at present it makes up most of the time to first token.
- Log `prompt_eval_count` per run to check whether warm runs reuse cached prompt tokens.
- Re-run the laptop on a high-performance power plan with other apps closed, to separate the hardware gap from the test conditions.
- Measure total system memory during a run on the laptop, since 7.3 GB leaves little room for the full pipeline.
- Start with llama3.2:3b as the default model for speed. On the laptop it is the only one of the three that stays near a comfortable reading speed. Keep qwen3.5:4b as the option for better answers on stronger hardware, and decide between them once retrieval quality can be tested end to end.
