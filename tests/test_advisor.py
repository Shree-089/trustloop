"""Nemotron advisor: HTTP handling, response validation, fallback, secrecy. Network is mocked;
these tests do NOT prove the live Nebius API works (see README: `python src/llm_advisor.py --check`)."""
import email.message
import io
import json
import socket
import urllib.error

import pytest

from llm_advisor import (AdvisorAuthError, AdvisorModelUnavailable, NemotronAdvisor,
                         parse_action)
from recovery import MAX_ATTEMPTS, RecoveryPolicy, build_evidence

KEY = "sk-test-SECRET-123"
MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
REPORT = {"reason": "No finger contact", "failure": "grasp_missed", "success": False,
          "checks": {"left_pad_contact": False, "right_pad_contact": False},
          "measurements": {"gap": 0.0005, "cube_width": 0.04, "cube_pos_world": [1, 2, 3],
                           "secret_internal": "qpos"}}


class _Resp:
    def __init__(self, obj, status=200):
        self._b = (obj if isinstance(obj, str) else json.dumps(obj)).encode()
        self.status = status

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _http_error(code, body="", retry_after=None):
    h = email.message.Message()
    if retry_after is not None:
        h["Retry-After"] = str(retry_after)
    return urllib.error.HTTPError("https://x/v1", code, "err", h, io.BytesIO(body.encode()))


class FakeOpener:
    """Serves queued responses; records requests (to check headers / bodies)."""

    def __init__(self, *responses):
        self.queue = list(responses)
        self.requests = []

    def __call__(self, req, timeout=None):
        self.requests.append(req)
        r = self.queue.pop(0)
        if isinstance(r, BaseException):
            raise r
        return _Resp(r)


def chat(content, reasoning=None):
    msg = {"role": "assistant", "content": content}
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    return {"choices": [{"message": msg}], "usage": {"total_tokens": 42}}


MODELS = {"data": [{"id": MODEL}, {"id": "meta-llama/Llama-3.3-70B-Instruct"}]}


def advisor(*responses, verify=True, **kw):
    op = FakeOpener(*((MODELS,) if verify else ()), *responses)
    adv = NemotronAdvisor(api_key=KEY, model=MODEL, base_url="https://api.example/v1",
                          max_retries=1, timeout=1.0, verify_model=verify, _opener=op, **kw)
    return adv, op


def decide(adv, failure="grasp_missed", attempts=1, max_attempts=MAX_ATTEMPTS):
    return RecoveryPolicy(adv).decide(failure, REPORT, attempts, max_attempts)


# ------------------------------------------------------------------ happy path
def test_valid_action_is_used_and_request_is_well_formed():
    adv, op = advisor(chat('{"action": "adjust_approach", "rationale": "fingers missed"}'))
    d = decide(adv)
    assert d.action == "adjust_approach" and d.source == "nemotron"
    assert d.advisor_call["status"] == "ok" and d.advisor_call["latency_s"] is not None
    req = op.requests[-1]
    assert req.full_url == "https://api.example/v1/chat/completions"
    assert req.get_header("Authorization") == f"Bearer {KEY}"
    body = json.loads(req.data)
    assert body["model"] == MODEL and body["temperature"] == 0.0
    sent = json.loads(body["messages"][1]["content"])
    assert sent["allowed_actions"] == ["retry_grasp", "adjust_approach", "abort_task"]


def test_only_whitelisted_evidence_is_sent():
    adv, op = advisor(chat('{"action": "retry_grasp"}'))
    decide(adv)
    sent = json.loads(json.loads(op.requests[-1].data)["messages"][1]["content"])
    meas = sent["evidence"]["measurements"]
    assert meas == {"gap": 0.0005, "cube_width": 0.04}
    assert "cube_pos_world" not in json.dumps(sent) and "secret_internal" not in json.dumps(sent)
    assert build_evidence("x", REPORT)["checks"] == {"left_pad_contact": False, "right_pad_contact": False}


def test_reasoning_trace_and_reasoning_content_are_handled():
    adv, _ = advisor(chat('<think>maybe {"action": "abort_task"}</think>{"action": "retry_grasp"}'))
    assert decide(adv).action == "retry_grasp"
    adv, _ = advisor(chat(None, reasoning='... final: {"action": "adjust_approach"}'))
    assert decide(adv).action == "adjust_approach"


# ------------------------------------------------------------------ validation
@pytest.mark.parametrize("reply", [
    '{"action": "set_joint_angles", "q": [0, 1, 2, 3, 4, 5]}',   # unrestricted command
    '{"action": "repick_and_place"}',                            # real action, not allowed here
    "Retry the grasp please",                                    # no JSON
])
def test_invalid_or_unsupported_suggestion_falls_back(reply):
    adv, _ = advisor(chat(reply))
    d = decide(adv)
    assert d.source == "fallback" and d.action == "retry_grasp"


def test_advisor_cannot_extend_attempt_budget():
    adv, op = advisor()   # must not even be called: only abort_task is allowed
    d = decide(adv, attempts=MAX_ATTEMPTS)
    assert d.allowed_actions == ["abort_task"] and d.action == "abort_task"
    assert d.source == "deterministic" and len(op.requests) == 1   # only the /models check


# ------------------------------------------------------------------ errors
def test_auth_error_falls_back_and_disables_advisor():
    adv, op = advisor(_http_error(401, '{"error": "bad key"}'))
    d = decide(adv)
    assert d.source == "fallback" and "AdvisorAuthError" in d.note
    n = len(op.requests)
    d2 = decide(adv)                                  # circuit breaker: no new HTTP request
    assert d2.source == "fallback" and len(op.requests) == n


def test_auth_error_at_startup():
    op = FakeOpener(_http_error(403, "forbidden"))
    with pytest.raises(AdvisorAuthError):
        NemotronAdvisor(api_key=KEY, model=MODEL, _opener=op)


def test_unavailable_model_detected_at_startup():
    op = FakeOpener({"data": [{"id": "nvidia/Some-Other-Nemotron"}]})
    with pytest.raises(AdvisorModelUnavailable) as e:
        NemotronAdvisor(api_key=KEY, model=MODEL, _opener=op)
    assert "nvidia/Some-Other-Nemotron" in str(e.value)


def test_rate_limit_is_retried_then_succeeds():
    adv, op = advisor(_http_error(429, "slow down", retry_after=0), chat('{"action": "adjust_approach"}'))
    d = decide(adv)
    assert d.source == "nemotron" and d.advisor_call["attempts"] == 2


def test_persistent_rate_limit_falls_back():
    adv, _ = advisor(_http_error(429, "", 0), _http_error(429, "", 0))
    d = decide(adv)
    assert d.source == "fallback" and "AdvisorRateLimited" in d.note


def test_timeout_falls_back():
    adv, _ = advisor(socket.timeout("t"), urllib.error.URLError(socket.timeout("t")))
    d = decide(adv)
    assert d.source == "fallback" and "AdvisorTimeout" in d.note


@pytest.mark.parametrize("payload", ["<html>502</html>", {"unexpected": True}, {"choices": []}])
def test_malformed_response_falls_back(payload):
    adv, _ = advisor(payload)
    d = decide(adv)
    assert d.source == "fallback" and "AdvisorResponseError" in d.note


def test_missing_configuration(monkeypatch):
    from llm_advisor import AdvisorConfigError
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    monkeypatch.delenv("NEMOTRON_MODEL", raising=False)
    with pytest.raises(AdvisorConfigError):
        NemotronAdvisor()


def test_api_key_never_leaks():
    adv, _ = advisor(_http_error(401, f'{{"error": "invalid key {KEY}"}}'))
    d = decide(adv)
    blob = json.dumps(d.to_dict(), default=str) + repr(adv) + str(adv.last_call)
    assert KEY not in blob


def test_parse_action_takes_last_json_object():
    assert parse_action('{"action": "a"} then {"action": "b", "rationale": "r"}') == ("b", "r")
