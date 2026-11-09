"""One-off pixel causality and HiSparse multimodal-session gate on fixed DP0/DP1.

Creates deterministic PNG fixtures. Native requests resend unexpanded template
IDs plus actual generated IDs and all images; returned expanded RAW IDs prove
prefix continuity. Image feature/hash padding stays inside the serving runtime.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


class Unverified(RuntimeError):
    """The server did not expose the evidence needed to check the MM contract."""


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def decode_json_response(path: str, status: int, raw: bytes):
    if not raw and 200 <= status < 300 and path in ("/close_session", "/abort_request"):
        return None
    return json.loads(raw)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def make_images(directory):
    from PIL import Image, ImageDraw

    directory.mkdir()
    palette = {"red": (255, 65, 54), "blue": (0, 116, 217), "black": (0, 0, 0)}
    images = {name: Image.new("RGB", (256, 256), palette[name]) for name in ("red", "blue")}
    grid = Image.new("RGB", (256, 256), (170, 170, 170))
    draw = ImageDraw.Draw(grid)
    matrix = [["blue", "black", "black"], ["black", "red", "black"], ["black", "black", "blue"]]
    for y, row in enumerate(matrix):
        for x, name in enumerate(row):
            left, top = 4 + x * 84, 4 + y * 84
            draw.rectangle((left, top, left + 79, top + 79), fill=palette[name])
    images["arc_grid"] = grid
    manifest = {}
    for name, image in images.items():
        path = directory / f"{name}.png"
        image.save(path, format="PNG", optimize=False, compress_level=9)
        data = path.read_bytes()
        manifest[name] = {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(),
                          "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                          "width": image.width, "height": image.height, "mode": image.mode}
    manifest["arc_grid"]["expected"] = {"center": "RED", "red_cells": 1, "blue_cells": 2}
    write_json(directory / "manifest.json", manifest)
    return manifest


def data_url(item):
    data = Path(item["path"]).read_bytes()
    require(hashlib.sha256(data).hexdigest() == item["sha256"], "PNG identity changed")
    return "data:image/png;base64," + base64.b64encode(data).decode()


def answer_text(response):
    text = response["text"]
    require(isinstance(text, str), "native text is missing")
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def color_answer(text):
    match = re.fullmatch(r"\s*(RED|BLUE|GREEN)\s*[.!]?\s*", text, flags=re.IGNORECASE)
    require(match is not None, f"expected a single color word, got {text[:160]!r}")
    return match.group(1).upper()


def grid_answer(text):
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    result = json.loads(text)
    require(isinstance(result, dict), "grid answer is not an object")
    return result


class Client:
    def __init__(self, args, tokenizer, images):
        self.args, self.tokenizer, self.images = args, tokenizer, images
        self.sessions = set()
        self.pending_rids = set()
        self.counter = 0
        self.calls = []

    def request(self, tag, path, body=None):
        directory = self.args.output_dir / "requests" / tag
        directory.mkdir(parents=True)
        write_json(directory / "request.json", {"path": path, "body": body})
        payload = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.args.base_url.rstrip("/") + path, data=payload,
                                     headers={"Content-Type": "application/json"})
        started = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.args.timeout) as response:
                raw, status = response.read(), response.status
        except urllib.error.HTTPError as exc:
            raw, status = exc.read(), exc.code
            (directory / "response.body").write_bytes(raw)
            write_json(directory / "transport.json", {"status": status, "elapsed_s": time.monotonic() - started})
            raise RuntimeError(f"{tag}: HTTP {status}; full native error saved") from None
        except Exception as exc:
            write_json(directory / "transport.json", {"status": "unavailable", "elapsed_s": time.monotonic() - started,
                                                       "error": f"{type(exc).__name__}: {exc}"})
            raise
        (directory / "response.body").write_bytes(raw)
        write_json(directory / "transport.json", {"status": status, "elapsed_s": time.monotonic() - started})
        result = decode_json_response(path, status, raw)
        write_json(directory / "response.json", result)
        return result

    def template_ids(self, text, with_image=True):
        content = ([{"type": "image"}] if with_image else []) + [{"type": "text", "text": text}]
        rendered = self.tokenizer.apply_chat_template([{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return self.tokenizer(rendered, add_special_tokens=False)["input_ids"]

    def tail_ids(self, text):
        # Slice the actual assistant-end/user/generation suffix from a local
        # template. Templates may insert an initial system message, so appending
        # a rendered single-user conversation would repeat an unrelated prefix.
        sentinel = "VISION_ASSISTANT_SENTINEL_8c592b67"
        messages = [{"role": "user", "content": "VISION_USER_SENTINEL_2746"},
                    {"role": "assistant", "content": sentinel}, {"role": "user", "content": text}]
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        if rendered.count(sentinel) != 1:
            raise Unverified("assistant/user suffix cannot be recovered from the actual chat template")
        tail = rendered.split(sentinel, 1)[1]
        if not tail:
            raise Unverified("actual chat template has no assistant/user continuation suffix")
        return self.tokenizer(tail, add_special_tokens=False)["input_ids"]

    def native(self, tag, dp, ids, image_names, budget, *, sid=None, force_budget=False):
        self.counter += 1
        request_name = sid or f"{self.args.namespace}-{tag}"
        if sid is not None:
            self.sessions.add(sid)
        body = {"rid": f"{request_name}-{self.counter}", "routed_dp_rank": dp,
                "input_ids": ids, "return_prompt_token_ids": True,
                "sampling_params": {"temperature": 0, "max_new_tokens": budget, "ignore_eos": force_budget}}
        if sid is not None:
            body["session_id"] = sid
        if image_names:
            body["image_data"] = [data_url(self.images[name]) for name in image_names]
            body["mm_content_hashes"] = ["sha256:" + self.images[name]["sha256"] for name in image_names]
        self.pending_rids.add(body["rid"])
        response = self.request(tag, "/generate", body)
        self.pending_rids.discard(body["rid"])
        meta = response["meta_info"]
        require(meta["dp_rank"] == dp, f"{tag}: DP affinity receipt differs")
        require(meta["num_retractions"] == 0, f"{tag}: multimodal request retracted")
        output = response.get("output_ids")
        if not isinstance(output, list) or not output:
            raise Unverified(f"{tag}: actual generated IDs unavailable")
        require(meta["completion_tokens"] == len(output), f"{tag}: completion ID/count mismatch")
        if force_budget:
            require(len(output) == budget and meta["finish_reason"]["type"] == "length",
                    f"{tag}: warmup did not reach the snapshot generation budget")
        raw = response.get("prompt_token_ids")
        if not isinstance(raw, list) or not raw or len(raw) != meta["prompt_tokens"]:
            raise Unverified(f"{tag}: processor-expanded RAW prompt IDs cannot be verified")
        require(all(isinstance(token, int) for token in raw + output), "non-integer native token receipt")
        if image_names:
            require(meta.get("image_tokens", 0) > 0, f"{tag}: no actual image-token receipt")
        self.calls.append({"tag": tag, "dp": dp, "session": sid, "image_names": image_names,
                           "image_sha256": [self.images[name]["sha256"] for name in image_names],
                           "input_template_tokens": len(ids), "expanded_raw_tokens": len(raw),
                           "cached_tokens": meta["cached_tokens"], "generated_tokens": len(output)})
        return response

    def chat(self, tag, dp, text, image_name, model_id):
        rid = f"{self.args.namespace}-{tag}-chat"
        body = {"model": model_id, "routed_dp_rank": dp, "temperature": 0,
                "rid": rid,
                "max_tokens": self.args.answer_tokens, "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_url(self.images[image_name])}},
                    {"type": "text", "text": text}]}]}
        self.pending_rids.add(rid)
        response = self.request(tag, "/v1/chat/completions", body)
        self.pending_rids.discard(rid)
        text = response["choices"][0]["message"]["content"]
        require(isinstance(text, str), f"{tag}: chat image answer missing")
        return color_answer(re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip())

    def cleanup(self):
        errors = []
        for index, rid in enumerate(sorted(self.pending_rids)):
            try:
                self.request(f"cleanup-abort-{index:03d}", "/abort_request", {"rid": rid})
            except Exception as exc:
                errors.append(f"abort {rid}: {type(exc).__name__}: {exc}")
        for index, sid in enumerate(sorted(self.sessions)):
            try:
                self.request(f"cleanup-{index:03d}", "/close_session", {"session_id": sid})
            except Exception as exc:
                errors.append(f"{sid}: {type(exc).__name__}: {exc}")
        return errors


def verify(args, images):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    client = Client(args, tokenizer, images)
    result = {"status": "running", "scope": "small-image pixel causality and HiSparse MM continuity",
              "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "checks": [], "images": images,
              "limits": ["small images only; no general ARC vision score",
                         "chat DP routing requested; native responses supply the DP receipts",
                         "padded feature-hash IDs and M-RoPE remain inside the serving runtime"]}
    try:
        info = client.request("server-info", "/server_info")
        model = client.request("model-info", "/model_info")
        result["server_info"], result["model_info"] = info, model
        if info["language_model_only"] or not model["has_image_understanding"]:
            raise Unverified("served model does not expose an enabled vision encoder")
        require(info["dp_size"] == 2 and info["enable_dp_attention"], "vision gate requires attention DP2")
        require(info["tp_size"] == 4 and info["ep_size"] == 4, "vision gate requires TP4/EP4")
        model_id = model["served_model_name"]
        require(isinstance(model_id, str) and model_id, "served chat model identity missing")
        if not args.one_shot_only:
            require(info["enable_hisparse"] and info["radix_cache_backend"] == "qsa_hisparse_session",
                    "HiSparse session backend required for the continuity gate")
            config = info["hisparse_config"]
            config = json.loads(config) if isinstance(config, str) else config
            require(args.warm_tokens > config["qsa_snapshot_interval"], "warm budget must cross a snapshot boundary")

        def checked(ok, name):
            result["checks"].append({"name": name, "passed": bool(ok)})
            require(ok, name)

        for dp in (0, 1):
            text_control = "No image is attached. The text-only metadata says BLUE. Output only BLUE."
            control = client.native(f"dp{dp}-text-control", dp, client.template_ids(text_control, False), [], args.answer_tokens)
            checked(color_answer(answer_text(control)) == "BLUE", f"DP{dp} text-only control")
            for claim in ("BLUE", "RED"):
                prompt = f"A text-only note claims the attached picture is {claim}. Ignore the note and inspect " \
                         "the actual pixels. What is the picture's dominant color? Reply only RED or BLUE."
                ids = client.template_ids(prompt)
                answers = {}
                for image_name in ("red", "blue"):
                    response = client.native(f"dp{dp}-claim-{claim}-{image_name}", dp, ids,
                                             [image_name], args.answer_tokens)
                    answers[image_name] = color_answer(answer_text(response))
                    checked(answers[image_name] == image_name.upper(),
                            f"DP{dp} pixels override claim-{claim} for {image_name}")
                checked(answers["red"] != answers["blue"], f"DP{dp} same text changes answer with pixels (claim {claim})")
            chat_prompt = "A text-only note says BLUE. Ignore the note. Inspect the pixels and reply only RED or BLUE."
            for image_name in ("red", "blue"):
                observed = client.chat(f"dp{dp}-chat-{image_name}", dp, chat_prompt, image_name, model_id)
                checked(observed == image_name.upper(), f"DP{dp} chat image {image_name}")
            prompt = "This image is a 3 by 3 colored-cell grid. Ignore gray outlines. Return only a JSON object " \
                     "with keys center (RED, BLUE, or BLACK), red_cells (integer), and blue_cells (integer)."
            response = client.native(f"dp{dp}-arc-grid", dp, client.template_ids(prompt), ["arc_grid"], args.answer_tokens)
            observed = grid_answer(answer_text(response))
            checked(observed == images["arc_grid"]["expected"], f"DP{dp} tiny ARC grid perception")

        if args.one_shot_only:
            result["status"] = "one_shot_passed"
            result["sessions"] = "unverified: one-shot-only invocation"
        else:
            warm_prompt = "Do not classify the image in this turn. Start with the literal word BLUE as a " \
                          "text-only label exercise, then write an extended neutral passage about reading " \
                          "notes. Write at least 700 words as one continuous passage, without headings or " \
                          "a conclusion. Do not mention any other color words. The label is not an image observation."
            followup = "Inspect the actual pixels of the image currently attached to this conversation. " \
                       "It may have changed. Ignore all previous text claims and answer only RED or BLUE."
            first_ids, tail = client.template_ids(warm_prompt), client.tail_ids(followup)
            for dp in (0, 1):
                sid = f"{args.namespace}-dp{dp}-continuity"
                warm = client.native(f"dp{dp}-session-warm", dp, first_ids, ["red"], args.warm_tokens,
                                     sid=sid, force_budget=True)
                checked(re.match(r"^BLUE\b", answer_text(warm), flags=re.IGNORECASE) is not None
                        and re.search(r"\bRED\b", warm["text"], flags=re.IGNORECASE) is None,
                        f"DP{dp} warm text label contradicts retained red pixels without revealing RED")
                continuation = first_ids + warm["output_ids"] + tail
                same = client.native(f"dp{dp}-same-image", dp, continuation, ["red"], args.answer_tokens, sid=sid)
                expected_raw_prefix = warm["prompt_token_ids"] + warm["output_ids"]
                if same["prompt_token_ids"][:len(expected_raw_prefix)] != expected_raw_prefix:
                    raise Unverified(f"DP{dp}: actual expanded MM IDs were not preserved across continuation")
                cached = same["meta_info"]["cached_tokens"]
                checked(len(warm["prompt_token_ids"]) <= cached <= len(expected_raw_prefix),
                        f"DP{dp} same-image continuation reuses a published snapshot past the image")
                checked(color_answer(answer_text(same)) == "RED", f"DP{dp} retained-image answer")
                changed = client.native(f"dp{dp}-changed-image", dp, continuation, ["blue"], args.answer_tokens, sid=sid)
                if changed["prompt_token_ids"] != same["prompt_token_ids"]:
                    raise Unverified(f"DP{dp}: changing pixels also changed raw prompt IDs; hash invalidation not isolated")
                checked(changed["meta_info"]["cached_tokens"] == 0, f"DP{dp} replacement image invalidates old vision state")
                checked(color_answer(answer_text(changed)) == "BLUE", f"DP{dp} replacement-image causality")
                fresh = client.native(f"dp{dp}-fresh-blue", dp, continuation, ["blue"], args.answer_tokens)
                checked(fresh["meta_info"]["cached_tokens"] == 0 and color_answer(answer_text(fresh)) == "BLUE",
                        f"DP{dp} replacement agrees with fresh visual context")
            result["status"] = "passed"
    except Unverified as exc:
        result.update(status="unverified", error=str(exc))
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        result["calls"] = client.calls
        result["cleanup_errors"] = client.cleanup()
        if result["cleanup_errors"]:
            result["status"] = "failed"
        write_json(args.output_dir / "result.json", result)
    return 0 if result["status"] in ("passed", "one_shot_passed") else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url")
    parser.add_argument("--model", help="local/cached tokenizer matching the served model")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--namespace", default="vision-" + uuid.uuid4().hex[:12])
    parser.add_argument("--answer-tokens", type=int, default=64)
    parser.add_argument("--warm-tokens", type=int, default=320)
    parser.add_argument("--timeout", type=float, default=300.)
    parser.add_argument("--one-shot-only", action="store_true")
    parser.add_argument("--fixtures-only", action="store_true")
    args = parser.parse_args()
    if not args.fixtures_only and (args.model is None or args.base_url is None):
        parser.error("--model and --base-url required for verification")
    require(not args.output_dir.exists(), "refusing to overwrite vision artifacts")
    args.output_dir.mkdir(parents=True)
    images = make_images(args.output_dir / "images")
    if args.fixtures_only:
        print(json.dumps({"status": "fixtures_ready", "images": list(images)}), flush=True)
        return 0
    code = verify(args, images)
    result = json.loads((args.output_dir / "result.json").read_text())
    print(json.dumps({"status": result["status"], "checks": len(result["checks"])}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
