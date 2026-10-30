"""OPTIONAL: NVIDIA Nemotron recovery advisor via Nebius Token Factory.

Uses Token Factory's OpenAI-compatible REST API (https://api.tokenfactory.nebius.com/v1):
  GET  /models             -> {"data": [{"id": ...}, ...]}   (model availability check)
  POST /chat/completions   -> {"choices": [{"message": {"content": ...}}]}
Auth: "Authorization: Bearer $NEBIUS_API_KEY".

Safety contract:
  * The model sees only the whitelisted verification evidence built by
    recovery.build_evidence() plus the list of allowed action names.
  * It must answer with one action name. recovery.RecoveryPolicy validates the
    answer; anything else falls back to the deterministic policy.
  * It never produces joint commands; there is no code path for that.
  * The API key is read from the environment only, never printed, logged or
    written to any file. Error messages are scrubbed of it.

Configuration (environment variables):
  NEBIUS_API_KEY     required
  NEMOTRON_MODEL     required; exact id from `python src/llm_advisor.py --list-models`,
                     e.g. nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B
  NEBIUS_BASE_URL    optional, default https://api.tokenfactory.nebius.com/v1

Verify locally:  python src/llm_advisor.py --check
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1"

SYSTEM_PROMPT = (
    "You are the recovery planner of a simulated robot pick-and-place system. You receive "
    "a JSON object with the failure category, the physical verification evidence produced "
    "by an independent verifier, the attempts used, and the list of allowed recovery "
    "actions. Choose exactly one action from allowed_actions. Reply with ONLY a JSON "
    'object: {"action": "<one of allowed_actions>", "rationale": "<one short sentence>"}. '
    "Never invent actions and never output joint commands or coordinates."
)


# ----------------------------------------------------------------- errors
class AdvisorError(RuntimeError):
    category = "error"


class AdvisorConfigError(AdvisorError):
    category = "config"


class AdvisorAuthError(AdvisorError):
    category = "auth"


class AdvisorModelUnavailable(AdvisorError):
    category = "model_unavailable"


class AdvisorRateLimited(AdvisorError):
    category = "rate_limited"


class AdvisorTimeout(AdvisorError):
    category = "timeout"


class AdvisorResponseError(AdvisorError):
    category = "malformed_response"


class AdvisorHTTPError(AdvisorError):
    category = "http_error"


# ----------------------------------------------------------------- client
class NemotronAdvisor:
    name = "nemotron"

    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 model: str | None = None, timeout: float = 20.0, max_retries: int = 2,
                 max_tokens: int = 1024, verify_model: bool = True, _opener=None):
        self._api_key = api_key or os.environ.get("NEBIUS_API_KEY", "")
        self.base_url = (base_url or os.environ.get("NEBIUS_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.environ.get("NEMOTRON_MODEL", "")
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self._urlopen = _opener or urllib.request.urlopen
        self.last_call: dict | None = None
        self.last_rationale = ""
        self._disabled_reason: str | None = None   # circuit breaker for non-transient errors
        if not self._api_key:
            raise AdvisorConfigError("NEBIUS_API_KEY is not set")
        if not self.model:
            raise AdvisorConfigError("NEMOTRON_MODEL is not set (see `python src/llm_advisor.py --list-models`)")
        if verify_model:
            self.check_model_available()

    def __repr__(self) -> str:  # never show the key
        return f"NemotronAdvisor(model={self.model!r}, base_url={self.base_url!r})"

    # -------------------------------------------------------------- http
    def _scrub(self, text: str) -> str:
        return text.replace(self._api_key, "***") if self._api_key else text

    def _request(self, method: str, path: str, body: dict | None = None) -> tuple[dict, int, int]:
        """Returns (json, http_status, attempts). Retries 429/5xx/timeouts with bounded backoff."""
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Authorization": f"Bearer {self._api_key}", "Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        last_exc: AdvisorError | None = None
        for attempt in range(1, self.max_retries + 2):
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            try:
                with self._urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read().decode("utf-8", "replace")
                    status = getattr(resp, "status", 200)
                try:
                    return json.loads(raw), status, attempt
                except json.JSONDecodeError as exc:
                    raise AdvisorResponseError(f"non-JSON response from {path}: {raw[:200]!r}") from exc
            except urllib.error.HTTPError as exc:
                body_txt = self._scrub(exc.read().decode("utf-8", "replace")[:300]) if exc.fp else ""
                if exc.code in (401, 403):
                    raise AdvisorAuthError(f"HTTP {exc.code}: authentication failed ({body_txt})") from None
                if exc.code == 404 or (exc.code == 400 and "model" in body_txt.lower()):
                    raise AdvisorModelUnavailable(f"HTTP {exc.code}: model {self.model!r} unavailable "
                                                  f"({body_txt})") from None
                if exc.code == 429 or exc.code >= 500:
                    cls = AdvisorRateLimited if exc.code == 429 else AdvisorHTTPError
                    last_exc = cls(f"HTTP {exc.code} after {attempt} attempt(s): {body_txt}")
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    delay = min(float(retry_after), 5.0) if retry_after and retry_after.replace(".", "", 1).isdigit() \
                        else 0.5 * 2 ** (attempt - 1)
                    if attempt <= self.max_retries:
                        time.sleep(delay)
                        continue
                    raise last_exc from None
                raise AdvisorHTTPError(f"HTTP {exc.code}: {body_txt}") from None
            except (socket.timeout, TimeoutError) as exc:
                last_exc = AdvisorTimeout(f"timeout after {self.timeout}s ({type(exc).__name__})")
            except urllib.error.URLError as exc:
                if isinstance(exc.reason, (socket.timeout, TimeoutError)):
                    last_exc = AdvisorTimeout(f"timeout after {self.timeout}s")
                else:
                    raise AdvisorHTTPError(f"network error: {self._scrub(str(exc.reason))}") from None
            if attempt <= self.max_retries:
                time.sleep(0.5 * 2 ** (attempt - 1))
        raise last_exc or AdvisorHTTPError("request failed")

    # -------------------------------------------------------------- API
    def list_models(self) -> list[str]:
        data, _, _ = self._request("GET", "/models")
        try:
            return sorted(m["id"] for m in data["data"])
        except (KeyError, TypeError) as exc:
            raise AdvisorResponseError(f"unexpected /models response shape: {str(data)[:200]}") from exc

    def check_model_available(self) -> None:
        models = self.list_models()
        if self.model not in models:
            nem = [m for m in models if "nemotron" in m.lower()]
            raise AdvisorModelUnavailable(
                f"model {self.model!r} is not in your Token Factory model list. "
                f"Nemotron models available to this key: {nem or 'none'}")

    def suggest(self, failure_report: dict, allowed_actions: list[str]) -> str | None:
        body = {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(failure_report, default=str)},
            ],
        }
        t0 = time.perf_counter()
        call = {"model": self.model, "status": "error", "latency_s": None, "attempts": None,
                "error": None, "raw_action": None, "rationale": None}
        self.last_call = call
        if self._disabled_reason:
            call.update(status="skipped", latency_s=0.0, error=f"disabled: {self._disabled_reason}")
            raise AdvisorError(f"advisor disabled for this run ({self._disabled_reason})")
        try:
            data, http_status, attempts = self._request("POST", "/chat/completions", body)
            call.update(http_status=http_status, attempts=attempts)
            try:
                msg = data["choices"][0]["message"]
            except (KeyError, IndexError, TypeError) as exc:
                raise AdvisorResponseError(f"no choices/message in response: {str(data)[:200]}") from exc
            text = msg.get("content") or ""
            if not text.strip():
                text = msg.get("reasoning_content") or ""   # some reasoning models put output here
            action, rationale = parse_action(text)
            if action is None:
                raise AdvisorResponseError(f"no JSON action in reply: {text[:200]!r}")
            self.last_rationale = rationale or ""
            call.update(status="ok", raw_action=action, rationale=rationale,
                        usage=data.get("usage"))
            return action
        except AdvisorError as exc:
            call.update(error=f"{exc.category}: {exc}")
            if isinstance(exc, (AdvisorAuthError, AdvisorModelUnavailable)):
                self._disabled_reason = exc.category
            raise
        finally:
            call["latency_s"] = round(time.perf_counter() - t0, 3)


_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


def parse_action(text: str) -> tuple[str | None, str | None]:
    """Return (action, rationale) from the LAST JSON object in the reply that has an
    "action" key. Reasoning traces (<think>...</think>) are ignored. The action is NOT
    validated here; RecoveryPolicy validates it against the allowed list."""
    text = _THINK.sub("", text or "")
    dec = json.JSONDecoder()
    found = None
    i = text.find("{")
    while i != -1:
        try:
            obj, end = dec.raw_decode(text, i)
            if isinstance(obj, dict) and "action" in obj:
                found = obj
            i = text.find("{", end)
        except json.JSONDecodeError:
            i = text.find("{", i + 1)
    if found is None:
        return None, None
    act = found.get("action")
    return (str(act) if act is not None else None), (str(found.get("rationale", "")) or None)


# ----------------------------------------------------------------- CLI check
def _cli(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Verify Nebius Token Factory / Nemotron access")
    ap.add_argument("--list-models", action="store_true", help="list model ids available to your key")
    ap.add_argument("--check", action="store_true",
                    help="verify key + NEMOTRON_MODEL and make one real advisory call")
    args = ap.parse_args(argv)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from recovery import RecoveryPolicy  # noqa: E402

    try:
        if args.list_models:
            adv = NemotronAdvisor(model=os.environ.get("NEMOTRON_MODEL") or "-", verify_model=False)
            models = adv.list_models()
            print(f"{len(models)} models available. Nemotron:")
            for m in models:
                if "nemotron" in m.lower():
                    print("  ", m)
            return 0
        adv = NemotronAdvisor()   # verifies the model id against /models
        print(f"OK: key accepted, model {adv.model!r} is available at {adv.base_url}")
        report = {"reason": "No finger contact with cube – gripper closed on empty space",
                  "checks": {"left_pad_contact": False, "right_pad_contact": False,
                             "gap_consistent_with_cube": False, "cube_between_fingers": False},
                  "measurements": {"gap": 0.0005, "cube_width": 0.04}}
        dec = RecoveryPolicy(adv).decide("grasp_missed", report, attempts_used=1)
        print(json.dumps({"decision": dec.action, "source": dec.source, "note": dec.note,
                          "advisor_call": dec.advisor_call}, indent=2, default=str))
        return 0 if dec.source == "nemotron" else 1
    except AdvisorError as exc:
        print(f"FAILED ({exc.category}): {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(_cli())
