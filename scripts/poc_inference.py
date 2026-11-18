"""Proof of concept: run a quantized SLM through Ollama on the CPU only and measure latency.

Checks the core assumption of the project: that a quantized small language model can run on a
consumer CPU, with no GPU, at a usable speed.

For each run it reports:
  - load time      time Ollama took to load the model into RAM (cold start only)
  - TTFT           time to first token: request sent -> first visible answer token arrives
  - prefill tok/s  how fast the prompt was processed
  - gen tok/s      how fast the answer was generated (tokens per second, TPS)
It then asks Ollama where the model is loaded, to confirm that none of it is on the GPU.

Usage (from the repository root):
    python scripts/poc_inference.py --model llama3.2:3b
    python scripts/poc_inference.py --model phi4-mini --runs 3
    python scripts/poc_inference.py --model qwen3.5:4b --think off
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
import urllib.error
import urllib.request

DEFAULT_PROMPT = "Explain in three sentences what retrieval-augmented generation is."


# --------------------------------------------------------------------------- Ollama HTTP helpers

def ollama_host() -> str:
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    if not host.startswith("http"):
        host = "http://" + host
    return host


def get_json(host: str, path: str) -> dict:
    with urllib.request.urlopen(host + path, timeout=10) as resp:
        return json.load(resp)


def post(host: str, path: str, payload: dict, timeout: int = 900):
    request = urllib.request.Request(
        host + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(request, timeout=timeout)


def with_tag(model: str) -> str:
    """Ollama reports 'llama3.2' as 'llama3.2:latest', so normalise names before comparing."""
    return model if ":" in model else f"{model}:latest"


def unload_model(host: str, model: str) -> None:
    """keep_alive=0 tells Ollama to unload the model, so the next run is a true cold start."""
    with post(host, "/api/generate", {"model": model, "keep_alive": 0}) as resp:
        resp.read()


# --------------------------------------------------------------------------- one measured run

def run_once(host: str, model: str, prompt: str, options: dict, think: bool | None) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "options": options,
    }
    if think is not None:
        payload["think"] = think

    answer_parts: list[str] = []
    thinking_chars = 0
    ttft = None
    final = None

    start = time.perf_counter()
    with post(host, "/api/chat", payload) as resp:
        for line in resp:  # Ollama streams one JSON object per line
            if not line.strip():
                continue
            chunk = json.loads(line)
            if "error" in chunk:
                raise RuntimeError(chunk["error"])

            message = chunk.get("message", {})
            thinking_chars += len(message.get("thinking") or "")
            text = message.get("content") or ""
            if text and ttft is None:
                ttft = time.perf_counter() - start
            answer_parts.append(text)

            if chunk.get("done"):
                final = chunk
    wall = time.perf_counter() - start

    if final is None:
        raise RuntimeError("Stream ended without a final 'done' message from Ollama.")

    answer = "".join(answer_parts)
    ns = 1e9  # Ollama reports durations in nanoseconds
    eval_count = final.get("eval_count", 0)
    eval_s = final.get("eval_duration", 0) / ns
    prompt_count = final.get("prompt_eval_count", 0)
    prompt_s = final.get("prompt_eval_duration", 0) / ns

    return {
        "load_s": final.get("load_duration", 0) / ns,
        "ttft_s": ttft if ttft is not None else float("nan"),
        "prefill_tps": prompt_count / prompt_s if prompt_s else float("nan"),
        "gen_tps": eval_count / eval_s if eval_s else float("nan"),
        "gen_tokens": eval_count,
        "wall_s": wall,
        "thinking_chars": thinking_chars + answer.count("<think>"),
        "answer": answer,
    }


# --------------------------------------------------------------------------- reporting

def machine_summary(host: str) -> str:
    try:
        import psutil  # installed with the project dependencies
        ram = f"{psutil.virtual_memory().total / 1024**3:.1f} GB"
    except ImportError:
        ram = "unknown (psutil not installed)"
    try:
        ollama_version = get_json(host, "/api/version").get("version", "unknown")
    except Exception:
        ollama_version = "unknown"
    return (
        f"OS: {platform.system()} {platform.release()} | CPU: {platform.processor() or 'unknown'} "
        f"| logical cores: {os.cpu_count()} | RAM: {ram} | Ollama: {ollama_version} "
        f"| Python: {platform.python_version()}"
    )


def placement_check(host: str, model: str) -> tuple[bool, str]:
    """Ask Ollama where the model is loaded. size_vram == 0 means it is entirely in system RAM."""
    for entry in get_json(host, "/api/ps").get("models", []):
        if with_tag(entry.get("name", "")) == with_tag(model):
            size, vram = entry.get("size", 0), entry.get("size_vram", 0)
            gpu_pct = 100 * vram / size if size else 0
            detail = f"model memory {size / 1024**3:.2f} GB, on GPU {vram / 1024**3:.2f} GB ({gpu_pct:.0f}%)"
            return vram == 0, detail
    return False, "model not found in 'ollama ps' output"


def print_table(results: list[dict]) -> None:
    header = f"{'run':>3} | {'load s':>7} | {'TTFT s':>7} | {'prefill tok/s':>13} | {'gen tok/s':>9} | {'tokens':>6} | {'total s':>7}"
    print(header)
    print("-" * len(header))
    for i, r in enumerate(results, 1):
        print(
            f"{i:>3} | {r['load_s']:>7.2f} | {r['ttft_s']:>7.2f} | {r['prefill_tps']:>13.1f} "
            f"| {r['gen_tps']:>9.1f} | {r['gen_tokens']:>6} | {r['wall_s']:>7.2f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU-only inference proof of concept for Ollama models.")
    parser.add_argument("--model", default="llama3.2:3b", help="Ollama model tag, e.g. phi4-mini, qwen3.5:4b")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--runs", type=int, default=3, help="number of runs; run 1 is a cold start")
    parser.add_argument("--ctx", type=int, default=4096, help="context window (num_ctx), same for every model")
    parser.add_argument("--max-tokens", type=int, default=256, help="cap on generated tokens (num_predict)")
    parser.add_argument("--think", choices=["on", "off"], help="thinking mode, for models that support it")
    parser.add_argument("--keep-loaded", action="store_true", help="skip unloading, so run 1 is not cold")
    args = parser.parse_args()

    host = ollama_host()
    think = {"on": True, "off": False}.get(args.think)
    options = {
        "num_gpu": 0,          # offload zero layers to the GPU: CPU-only inference
        "num_ctx": args.ctx,
        "num_predict": args.max_tokens,
        "temperature": 0,      # deterministic output so runs are comparable
        "seed": 42,
    }

    print(f"Machine  | {machine_summary(host)}")
    print(f"Model    | {args.model} | num_ctx={args.ctx} | max_tokens={args.max_tokens} | think={args.think or 'model default'}")
    print()

    try:
        if not args.keep_loaded:
            unload_model(host, args.model)
        results = []
        for i in range(args.runs):
            print(f"Run {i + 1}/{args.runs}...", flush=True)
            results.append(run_once(host, args.model, args.prompt, options, think))
        on_cpu, placement = placement_check(host, args.model)
    except urllib.error.URLError as exc:
        if isinstance(exc, urllib.error.HTTPError):
            body = exc.read().decode(errors="replace")
            if exc.code == 404:
                print(f"Model not found. Download it first:  ollama pull {args.model}")
            else:
                print(f"Ollama returned HTTP {exc.code}: {body}")
        else:
            print(f"Could not reach Ollama at {host}. Is the Ollama app (or 'ollama serve') running?")
        return 1
    except RuntimeError as exc:
        print(f"Ollama error: {exc}")
        return 1

    print("\nAnswer (run 1):")
    print(results[0]["answer"].strip())
    print()
    print_table(results)

    warm = results[1:] or results
    print()
    print(f"Mean of warm runs | TTFT {statistics.mean(r['ttft_s'] for r in warm):.2f} s "
          f"| gen {statistics.mean(r['gen_tps'] for r in warm):.1f} tok/s")
    print(f"CPU-only check    | {'PASS' if on_cpu else 'FAIL'} ({placement})")
    if any(r["thinking_chars"] for r in results):
        print("Thinking check    | model produced reasoning text; use --think off for benchmark runs")

    return 0 if on_cpu else 2


if __name__ == "__main__":
    sys.exit(main())