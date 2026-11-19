"""Unit tests for the Ollama embedder, run against a small fake Ollama server (no models needed)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from local_rag_pipeline.embeddings import EmbeddingError, OllamaEmbedder

MAX_CHARS = 50  # the fake server "refuses" longer inputs when truncate is false


class FakeOllama(BaseHTTPRequestHandler):
    requests: list[dict] = []

    def log_message(self, *args):
        pass

    def _reply(self, code: int, body: dict):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.requests.append(body)
        if body["model"] == "missing":
            return self._reply(404, {"error": "model 'missing' not found"})
        texts = body["input"]
        if not body["truncate"] and any(len(t) > MAX_CHARS for t in texts):
            return self._reply(400, {"error": "input length exceeds the context length"})
        vectors = [[float(len(t)), 1.0, 0.0] for t in texts]
        return self._reply(200, {"embeddings": vectors, "prompt_eval_count": 5 * len(texts)})


@pytest.fixture
def host():
    FakeOllama.requests = []
    server = HTTPServer(("127.0.0.1", 0), FakeOllama)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_texts_are_sent_in_batches(host):
    embedder = OllamaEmbedder("all-minilm", host=host, batch_size=2)
    run = embedder.embed_documents(["a", "b", "c", "d", "e"])
    assert [len(r["input"]) for r in FakeOllama.requests] == [2, 2, 1]
    assert len(run.vectors) == 5 and run.requests == 3


def test_vectors_stay_in_input_order(host):
    run = OllamaEmbedder("all-minilm", host=host, batch_size=2).embed_documents(["x", "xxx", "xx"])
    assert [v[0] for v in run.vectors] == [1.0, 3.0, 2.0]


def test_every_request_is_cpu_only(host):
    OllamaEmbedder("all-minilm", host=host).embed_documents(["a", "b"])
    OllamaEmbedder("all-minilm", host=host).embed_query("question")
    assert all(r["options"]["num_gpu"] == 0 for r in FakeOllama.requests)


def test_too_long_chunks_are_found_truncated_and_counted(host):
    texts = ["short", "L" * 80, "also short"]
    run = OllamaEmbedder("all-minilm", host=host, batch_size=3).embed_documents(texts)
    assert run.truncated == [1]
    assert len(run.vectors) == 3
    assert [v[0] for v in run.vectors] == [5.0, 80.0, 10.0]


def test_token_count_is_summed(host):
    run = OllamaEmbedder("all-minilm", host=host, batch_size=2).embed_documents(["a", "b", "c"])
    assert run.tokens == 15


def test_progress_is_reported(host):
    seen = []
    OllamaEmbedder("all-minilm", host=host, batch_size=2).embed_documents(
        ["a", "b", "c"], progress=lambda done, total: seen.append((done, total)))
    assert seen == [(2, 3), (3, 3)]


def test_missing_model_gives_clear_message(host):
    with pytest.raises(EmbeddingError, match="ollama pull missing"):
        OllamaEmbedder("missing", host=host).embed_query("hi")


def test_ollama_not_running_gives_clear_message():
    with pytest.raises(EmbeddingError, match="Is it running"):
        OllamaEmbedder("all-minilm", host="http://127.0.0.1:9", timeout=2).embed_query("hi")


def test_one_long_chunk_is_isolated_with_few_requests(host):
    texts = ["short"] * 32
    texts[20] = "L" * 80
    run = OllamaEmbedder("all-minilm", host=host, batch_size=32).embed_documents(texts)
    assert run.truncated == [20]
    assert len(run.vectors) == 32
    assert len(FakeOllama.requests) <= 14, "halving should need far fewer than 32 retries"