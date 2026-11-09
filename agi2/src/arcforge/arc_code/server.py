"""The gateway's HTTP shell: the official 15 routes, stdlib only.

The route table maps one-to-one onto ``reference/polyphony-arc3-prep/vendor/arc/arc_agi/server.py:35-125``.
Not one more -- **any agent-visible endpoint the official server lacks makes practice differ from the real thing**.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .engine import ACTIONS, ArcEngineAdapter
from .errors import APIError, ApiFault, ServerError
from .gateway import Gateway, competition_divergences
from .ledger import Ledger
from .score import short_index
from .tokens import TokenTable

MAX_BODY = 1 << 20
COOKIE_NAME = "arc_session"


class Handler(BaseHTTPRequestHandler):
    gateway: Gateway
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # --- transport -----------------------------------------------------------
    def _send(self, status: int, payload: Any, with_cookie: bool = False) -> None:
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        session = self.gateway.cookie_value() if with_cookie else None
        if session is not None:
            # The official server attaches the cookie to 200s (api.py:44-53). act.py sends it back on every later request.
            self.send_header("Set-Cookie", f"{COOKIE_NAME}={session}; Path=/")
        self.end_headers()
        self.wfile.write(raw)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            raise ApiFault(413, APIError.VALIDATION_ERROR, "body too large")
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return {}

    def _cookie(self) -> str | None:
        raw = self.headers.get("Cookie")
        if not raw:
            return None
        jar = SimpleCookie()
        jar.load(raw)
        morsel = jar.get(COOKIE_NAME)
        return morsel.value if morsel else None

    def _via(self) -> str:
        """Informational only: the agent can forge it, and that is fine -- both paths are legitimate."""
        agent = (self.headers.get("User-Agent") or "").lower()
        return "act" if "arc-code-act" in agent else "direct"

    # --- routes -----------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def _dispatch(self, method: str) -> None:
        api_key: str | None = None
        try:
            gw = self.gateway
            api_key = gw.api_key(self.headers.get("X-API-Key"))
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            body = self._body() if method == "POST" else {}
            self._send(200, self._route(method, path, body, api_key), True)
        except ApiFault as fault:
            self._send(fault.status, fault.body(), True)
        except ServerError as error:
            # This used to be `raise`, meant as "configuration errors must be loud". But raising in
            # the handler thread only **closes the connection**: the agent gets RemoteDisconnected
            # with no idea what happened, and the run carries on degraded. The worst kind of loud.
            # Observed 2026-09-17: when ACTION6's coordinates arrived wrapped as {"data":{x,y}},
            # engine returned None -> ServerError -> connection dropped.
            print(f"ARCSEC SERVER ERROR: {error}", file=sys.stderr, flush=True)
            self._send(500, {"error": "SERVER_ERROR", "message": str(error)}, True)
        except Exception as error:  # bugs here must be visible, not a dropped socket
            import traceback
            traceback.print_exc()
            self._send(500, {"error": "SERVER_ERROR",
                             "message": f"{type(error).__name__}: {error}"}, True)

    def _route(self, method: str, path: str, body: dict[str, Any],
               api_key: str) -> Any:
        gw = self.gateway
        if method == "GET":
            if path == "/api/games":
                return gw.games()
            if path == "/api/healthcheck":
                return gw.healthcheck()
            m = re.fullmatch(r"/api/games/([^/]+)", path)
            if m:
                return gw.game(m.group(1))
            m = re.fullmatch(r"/api/scorecard/([^/]+)", path)
            if m:
                return gw.get_card(m.group(1), api_key)
            m = re.fullmatch(r"/api/scorecard/([^/]+)/([^/]+)", path)
            if m:
                return gw.get_card_game(m.group(1), m.group(2), api_key)
        else:
            if path == "/api/scorecard/open":
                return gw.open_card(api_key)
            if path == "/api/scorecard/close":
                return gw.close_card(body.get("card_id", ""), api_key)
            m = re.fullmatch(r"/api/cmd/([A-Z0-9_]+)", path)
            if m and m.group(1) in ACTIONS:
                return gw.command(m.group(1), body, api_key,
                                  cookie=self._cookie(), via=self._via())
        raise ApiFault(404, APIError.SERVER_ERROR, f"no route {path}")


def build(corpus: Path, tokens_file: Path, ledger_path: Path, *,
          recordings_dir: Path | None = None, competition_mode: bool = True,
          auth: str = "strict") -> Gateway:
    adapter = ArcEngineAdapter(corpus, recordings_dir)
    # The resolution rule lives in score.short_index, the only copy. The gateway, the launcher's
    # roster and finalise's denominator all go through it -- when each of the three had its own copy it went wrong once.
    resolve = short_index(adapter.list_games())
    tokens = TokenTable.load(tokens_file, resolve=lambda k: resolve.get(k, k))
    return Gateway(adapter, tokens, Ledger(ledger_path),
                   competition_mode=competition_mode, auth=auth)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--tokens", type=Path, required=True)
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--recordings-dir", type=Path)
    p.add_argument("--port", type=int, default=8731)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--auth", choices=("strict", "official"), default="strict")
    # On by default. Turning it off is only for deliberate exploratory experiments, and is recorded in run state.
    p.add_argument("--no-competition-mode", dest="competition_mode",
                   action="store_false")
    a = p.parse_args(argv)

    gw = build(a.corpus, a.tokens, a.ledger, recordings_dir=a.recordings_dir,
               competition_mode=a.competition_mode, auth=a.auth)
    live = competition_divergences(a.auth) if a.competition_mode else []
    for row in live:
        print(f"DIVERGENCE agent-visible: {row['what']}", file=sys.stderr)
    print(f"gateway on {a.host}:{a.port} competition_mode={a.competition_mode} "
          f"auth={a.auth} games={len(gw.adapter.list_games())}", file=sys.stderr)

    Handler.gateway = gw
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
