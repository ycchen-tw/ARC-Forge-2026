"""Control two independent TP replicas as one policy; group traffic goes directly to each backend.

The orchestrator uses this endpoint for health, ledger watermarks and weight updates. Each group
keeps its backend for its entire lifetime (including session close), preserving the real ledger
instance in its manifest. Any partial weight update permanently closes the control endpoint until
it is restarted after the replicas have been reconciled. No generation request is proxied here.
"""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class ReplicaError(RuntimeError):
    pass


def request(root: str, method: str, path: str, body: dict | None = None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(root + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as response:
        raw = response.read()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw.decode()


class Replicas:
    IDENTITY = ("model_path", "served_model_name", "weight_version", "weights_source")
    COUNTERS = ("queue", "pending", "written_bytes", "enqueued_events", "written_events", "lossy")

    def __init__(self, backends: list[str], transport=request):
        if not backends or len(set(backends)) != len(backends):   # one replica: the tp1x1 toy (10-25)
            raise ValueError("at least one replica URL, no duplicates, is required")
        self.backends = [url.rstrip("/") for url in backends]
        self.transport = transport
        self.mutation_lock = threading.Lock()
        self.poisoned: str | None = None
        self.instances: tuple[str, ...] | None = None

    def all(self, method, path, body=None):
        """Every backend's reply, in backend order. GETs go out in parallel (a loaded sglang answers /health in ~1 s, so
        six replicas one by one took 6 s, past the launcher's health timeout, 10-15); mutations stay sequential."""
        if method != "GET":
            return [self.transport(root, method, path, body) for root in self.backends]
        with ThreadPoolExecutor(len(self.backends)) as pool:
            return list(pool.map(lambda root: self.transport(root, method, path, body), self.backends))

    @staticmethod
    def same(docs, keys):
        for key in keys:
            values = [doc[key] for doc in docs]
            if any(value != values[0] for value in values[1:]):
                raise ReplicaError(f"replica {key} differs: {values!r}")

    def model_info(self):
        docs = self.all("GET", "/model_info")
        self.same(docs, self.IDENTITY)
        if any(doc["poisoned"] is not False for doc in docs):
            raise ReplicaError("a replica has poisoned weights")
        return {**docs[0], "replicas": docs, "replica_urls": [root + "/v1" for root in self.backends]}

    def status(self):
        docs = self.all("GET", "/session_ledger/status")
        instances = tuple(doc["instance"] for doc in docs)
        if len(set(instances)) != len(instances):
            raise ReplicaError(f"replicas share a ledger instance: {instances}")
        if self.instances is None:
            self.instances = instances
        elif instances != self.instances:
            raise ReplicaError(f"replica restarted: {self.instances} -> {instances}")
        combined = {**docs[0], "replica_statuses": docs}
        combined.update({key: sum(doc[key] for doc in docs) for key in self.COUNTERS})
        errors = [doc["writer_error"] for doc in docs if doc["writer_error"] is not None]
        combined["writer_error"] = None if not errors else repr(errors)
        return combined

    def handle(self, method, path, body=None):
        if self.poisoned is not None:
            raise ReplicaError(self.poisoned)
        if method == "GET":
            if path == "/health":
                self.all(method, path)
                return {"status": "ok", "replicas": len(self.backends)}
            if path == "/model_info":
                return self.model_info()
            if path == "/session_ledger/status":
                return self.status()
            if path == "/v1/models":
                docs = self.all(method, path)
                self.same([{"ids": [m["id"] for m in doc["data"]]} for doc in docs], ("ids",))
                return docs[0]
            if path.startswith("/v1/loads"):
                docs = self.all(method, path)
                loads = [{**row, "replica": i, "dp_rank": i}
                         for i, doc in enumerate(docs) for row in doc["loads"]]
                return {"loads": loads}
            raise ReplicaError(f"unsupported control GET {path}")
        if method != "POST" or path not in (
                "/pause_generation", "/continue_generation", "/update_weights_from_disk", "/flush_cache"):
            raise ReplicaError(f"unsupported control operation {method} {path}")
        with self.mutation_lock:
            if self.poisoned is not None:
                raise ReplicaError(self.poisoned)
            if path == "/update_weights_from_disk":
                if body.get("keep_identity") is not True:
                    raise ReplicaError("replica updates require keep_identity=true")
                before = self.model_info()
            try:
                docs = self.all(method, path, body)
                if path == "/update_weights_from_disk":
                    if any(doc["success"] is not True for doc in docs):
                        raise ReplicaError(f"weight update failed: {docs!r}")
                    after = self.model_info()
                    if (str(after["weight_version"]) != str(body["weight_version"])
                            or after["weights_source"] != body["model_path"]
                            or after["model_path"] != before["model_path"]
                            or after["served_model_name"] != before["served_model_name"]):
                        raise ReplicaError("replicas did not load the requested policy with identity preserved")
                    return {**docs[0], "replicas": docs,
                            "num_paused_requests": sum(doc["num_paused_requests"] for doc in docs)}
                if path in ("/pause_generation", "/continue_generation"):
                    if any(doc["status"] != "ok" for doc in docs):
                        raise ReplicaError(f"{path} failed: {docs!r}")
                    return {"status": "ok", "replicas": docs}
                return {"status": "ok", "replicas": docs}
            except Exception as error:
                # A timeout cannot prove whether a replica changed. Never resume a mixed policy.
                self.poisoned = f"replica control failed during {path}; reconcile replicas before restart: {error}"
                raise ReplicaError(self.poisoned) from error


def handler(replicas):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.respond("GET")

        def do_POST(self):
            self.respond("POST")

        def respond(self, method):
            try:
                size = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(size)) if size else None
                result = replicas.handle(method, self.path, body)
                status = 200
            except (ReplicaError, OSError, ValueError, KeyError) as error:
                result, status = {"error": str(error)}, 503
            encoded = json.dumps(result).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, fmt, *args):
            if args and str(args[1]) != "200":
                super().log_message(fmt, *args)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--backend", action="append", required=True)
    args = parser.parse_args()
    replicas = Replicas(args.backend)
    replicas.model_info()
    replicas.status()
    ThreadingHTTPServer((args.listen, args.port), handler(replicas)).serve_forever()


if __name__ == "__main__":
    main()
