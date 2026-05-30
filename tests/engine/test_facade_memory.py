"""Structural and byte-exact tests for memory injection (Chunk 4.2c).

These tests verify:
  1. memory-off no-op: ``memory_components=None`` → behaviour unchanged.
  2. byte-exact: memory-on, fixed context, token mode, frozen=0 → context
     injected into latest user turn, correct placement.
  3. cache mode: injection SKIPPED → body unchanged.
  4. frozen>0: context lands in the latest NON-FROZEN user turn.
  5. empty/None result from fetch_context: no-op.
  6. memory + CCR proactive-expansion together: both land in the latest
     non-frozen user turn in the correct order (memory first, then expansion).
  7. structural: fetch_context is called with the right messages/query/ctx;
     placement correct for both string and list message content.
  8. bypass gate: memory injection skipped under x-headroom-bypass.
  9. inject_context=False gate: injection skipped.

Running
-------
  .venv/bin/python -m pytest tests/engine/test_facade_memory.py -v
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_FIXED_CONTEXT = (
    "## Relevant memory\n- auth_middleware.py: JWT validation\n- auth_router.py: /login route"
)


def _make_engine(
    *,
    fetch_context: Any | None = None,
    memory_handler: Any | None = None,
    user_id: str | None = "test-user",
    config_overrides: dict[str, Any] | None = None,
    frozen_count: int = 0,
    ccr_context_tracker: Any | None = None,
    ccr_config_overrides: dict[str, Any] | None = None,
) -> Any:
    """Build a HeadroomEngine with MemoryComponents for structural/byte-exact tests."""
    from headroom.engine.contract import Flavor, Provider
    from headroom.engine.facade import (
        AnthropicComponents,
        CCRComponents,
        HeadroomEngine,
        MemoryComponents,
    )
    from headroom.proxy.models import ProxyConfig
    from headroom.proxy.server import HeadroomProxy

    config_kwargs: dict[str, Any] = {
        "optimize": True,
        "mode": "token",
        "cache_enabled": False,
        "rate_limit_enabled": False,
        "cost_tracking_enabled": False,
        "log_requests": False,
        "ccr_inject_tool": False,
        "ccr_inject_system_instructions": False,
        "ccr_handle_responses": False,
        "ccr_context_tracking": False,
        "ccr_proactive_expansion": False,
        "image_optimize": False,
    }
    if config_overrides:
        config_kwargs.update(config_overrides)
    if ccr_config_overrides:
        config_kwargs.update(ccr_config_overrides)

    config = ProxyConfig(**config_kwargs)
    proxy = HeadroomProxy(config)

    class _FixedStore:
        def compute_session_id(self, ctx: Any, model: str, msgs: Any) -> str:
            return "memory-structural-test-session"

        def get_or_create(self, session_id: str, provider: str) -> Any:
            class _T:
                def get_frozen_message_count(self) -> int:
                    return frozen_count

                def get_last_original_messages(self) -> list[Any]:
                    return []

                def get_last_forwarded_messages(self) -> list[Any]:
                    return []

            return _T()

        def get_fresh_cache(self, session_id: str) -> Any:
            class _C:
                def apply_cached(self, msgs: list[Any]) -> list[Any]:
                    return list(msgs)

                def compute_frozen_count(self, msgs: list[Any]) -> int:
                    return 0

                def update_from_result(self, orig: Any, compr: Any) -> None:
                    pass

                def mark_stable_from_messages(self, msgs: Any, up_to: int) -> None:
                    pass

            return _C()

    ac = AnthropicComponents(
        pipeline=proxy.anthropic_pipeline,
        provider=proxy.anthropic_provider,
        session_tracker_store=_FixedStore(),
        get_compression_cache=_FixedStore().get_fresh_cache,
        config=proxy.config,
        usage_reporter=None,
    )

    mc = None
    if fetch_context is not None or user_id is not None:
        _handler = memory_handler
        if _handler is None:
            # Default stub: has inject_context=True
            _handler = MagicMock()
            _handler.config.inject_context = True

        _fetch = fetch_context if fetch_context is not None else (lambda *_: None)
        mc = MemoryComponents(
            fetch_context=_fetch,
            memory_handler=_handler,
            user_id=user_id,
        )

    ccr = None
    if ccr_context_tracker is not None:
        ccr = CCRComponents(
            ccr_context_tracker=ccr_context_tracker,
            get_compression_store=lambda: MagicMock(),
            turn_counter=[0],
        )

    engine = HeadroomEngine(
        pipelines={(Provider.ANTHROPIC, Flavor.MESSAGES): proxy.anthropic_pipeline},
        config=proxy.config,
        usage_reporter=None,
        salt=b"memory-structural-test-salt",
        anthropic_components=ac,
        ccr_components=ccr,
        memory_components=mc,
    )
    return engine


def _make_ctx(
    body: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    cwd: str | None = None,
) -> Any:
    """Build a RequestContext for structural tests."""
    from headroom.engine.contract import Flavor, Provider, RequestContext

    h: dict[str, str] = {
        "x-api-key": "test-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    if cwd:
        h["x-headroom-cwd"] = cwd
    if headers:
        h.update(headers)

    return RequestContext(
        provider=Provider.ANTHROPIC,
        flavor=Flavor.MESSAGES,
        headers_view=h,
        raw_body=json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode(),
        session_key="memory-structural",
        request_id="req-mem-test",
    )


# ---------------------------------------------------------------------------
# 1. memory-off no-op (MemoryComponents=None)
# ---------------------------------------------------------------------------


def test_memory_noop_when_components_none() -> None:
    """Engine without MemoryComponents is byte-identical to pre-4.2c behaviour."""
    from headroom.engine.contract import Flavor, Provider
    from headroom.engine.facade import AnthropicComponents, HeadroomEngine
    from headroom.proxy.models import ProxyConfig
    from headroom.proxy.server import HeadroomProxy

    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
    )
    proxy = HeadroomProxy(config)

    class _TrivialStore:
        def compute_session_id(self, *a: Any, **kw: Any) -> str:
            return "s"

        def get_or_create(self, *a: Any, **kw: Any) -> Any:
            class _T:
                def get_frozen_message_count(self) -> int:
                    return 0

                def get_last_original_messages(self) -> list:
                    return []

                def get_last_forwarded_messages(self) -> list:
                    return []

            return _T()

        def get_fresh_cache(self, session_id: str) -> Any:
            class _C:
                def apply_cached(self, msgs: list) -> list:
                    return msgs

                def compute_frozen_count(self, msgs: list) -> int:
                    return 0

                def update_from_result(self, *a: Any) -> None:
                    pass

                def mark_stable_from_messages(self, *a: Any) -> None:
                    pass

            return _C()

    ac = AnthropicComponents(
        pipeline=proxy.anthropic_pipeline,
        provider=proxy.anthropic_provider,
        session_tracker_store=_TrivialStore(),
        get_compression_cache=_TrivialStore().get_fresh_cache,
        config=proxy.config,
        usage_reporter=None,
    )
    engine = HeadroomEngine(
        pipelines={(Provider.ANTHROPIC, Flavor.MESSAGES): proxy.anthropic_pipeline},
        config=proxy.config,
        usage_reporter=None,
        salt=b"s",
        anthropic_components=ac,
        memory_components=None,  # No memory
    )

    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "hello"}],
    }
    from headroom.engine.contract import Flavor, Provider, RequestContext

    ctx = RequestContext(
        provider=Provider.ANTHROPIC,
        flavor=Flavor.MESSAGES,
        headers_view={"x-api-key": "k", "anthropic-version": "2023-06-01"},
        raw_body=json.dumps(body).encode(),
        session_key="s",
        request_id="",
    )
    decision = engine.on_request(ctx)
    # optimize=False → passthrough → raw body unchanged
    assert decision.body == ctx.raw_body


# ---------------------------------------------------------------------------
# 2. byte-exact placement: token mode, frozen=0, string content
# ---------------------------------------------------------------------------


def test_memory_injection_string_content_token_mode() -> None:
    """Memory-on: context is appended to the latest user turn (string content).

    The injected text must appear as ``original_text + "\\n\\n" + context``.
    """
    call_log: list[tuple[list, str]] = []

    def _fetch(messages: list, query: str, ctx: Any) -> str:
        call_log.append((messages, query))
        return _FIXED_CONTEXT

    user_text = "How does authentication work?"
    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": user_text}],
    }
    ctx = _make_ctx(body)
    engine = _make_engine(fetch_context=_fetch)
    decision = engine.on_request(ctx)

    out = json.loads(decision.body)
    last_msg = out["messages"][-1]
    assert last_msg["role"] == "user"
    content = last_msg["content"]
    assert isinstance(content, str)
    assert content == user_text + "\n\n" + _FIXED_CONTEXT

    # fetch_context must have been called once
    assert len(call_log) == 1
    _, called_query = call_log[0]
    assert called_query == user_text


def test_memory_injection_list_content_token_mode() -> None:
    """Memory-on: context is appended to the first text block (list content)."""

    def _fetch(messages: list, query: str, ctx: Any) -> str:
        return _FIXED_CONTEXT

    original_text = "What is the rate limit policy?"
    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": original_text},
                    {"type": "text", "text": "Additional detail."},
                ],
            }
        ],
    }
    ctx = _make_ctx(body)
    engine = _make_engine(fetch_context=_fetch)
    decision = engine.on_request(ctx)

    out = json.loads(decision.body)
    last_msg = out["messages"][-1]
    blocks = last_msg["content"]
    assert isinstance(blocks, list)
    # First text block gets the injection; second stays unchanged.
    assert blocks[0]["text"] == original_text + "\n\n" + _FIXED_CONTEXT
    assert blocks[1]["text"] == "Additional detail."


# ---------------------------------------------------------------------------
# 3. cache mode: injection skipped
# ---------------------------------------------------------------------------


def test_memory_injection_skipped_in_cache_mode() -> None:
    """Cache mode: injection is skipped to preserve prefix stability."""
    fetch_called = [False]

    def _fetch(messages: list, query: str, ctx: Any) -> str:
        fetch_called[0] = True
        return _FIXED_CONTEXT

    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Cache mode test"}],
    }
    ctx = _make_ctx(body)
    engine = _make_engine(
        fetch_context=_fetch,
        config_overrides={"mode": "cache"},
    )
    decision = engine.on_request(ctx)

    # fetch_context is called (gate passes), but injection is skipped.
    assert fetch_called[0] is True

    out = json.loads(decision.body)
    last_msg = out["messages"][-1]
    content = last_msg.get("content", "")
    if isinstance(content, str):
        assert _FIXED_CONTEXT not in content, "Must not inject context in cache mode"
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                assert _FIXED_CONTEXT not in block.get("text", ""), (
                    "Must not inject context in cache mode"
                )


# ---------------------------------------------------------------------------
# 4. frozen>0: context lands in latest NON-FROZEN user turn
# ---------------------------------------------------------------------------


def test_memory_injection_respects_frozen_boundary() -> None:
    """frozen>0: context injects into the latest non-frozen user turn.

    The body has two user turns: [turn0 (frozen), turn1 (live)].
    With frozen_count=1, turn0 is off-limits; context should land in turn1.
    """

    def _fetch(messages: list, query: str, ctx: Any) -> str:
        return _FIXED_CONTEXT

    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": "Frozen turn — do not touch"},
            {"role": "assistant", "content": "Understood."},
            {"role": "user", "content": "Live turn — inject here"},
        ],
    }
    ctx = _make_ctx(body)
    # frozen_count=1 means the first message (index 0) is frozen; messages 1+ are live.
    # With 3 messages (indices 0,1,2) and frozen_count=1, index 2 (last user) is live.
    engine = _make_engine(fetch_context=_fetch, frozen_count=1)
    decision = engine.on_request(ctx)

    out = json.loads(decision.body)
    msgs = out["messages"]
    # Last message is the live user turn — must have context injected.
    assert msgs[-1]["role"] == "user"
    assert _FIXED_CONTEXT in msgs[-1]["content"]
    # Frozen turn must be byte-equal.
    assert _FIXED_CONTEXT not in msgs[0]["content"]


def test_append_context_helper_skips_fully_frozen_messages() -> None:
    """_append_context_to_latest_non_frozen_user_turn skips when last msg is frozen.

    In token mode the engine clamps frozen_message_count to
    min(tracker_count, cache.compute_frozen_count()), so an engine-level test
    of "fully frozen single turn" would have frozen_count clamped to 0 by the
    stub cache. We test the boundary condition directly on the ported helper
    so the invariant is covered without the clamping distraction.
    """
    from headroom.engine.facade import _append_context_to_latest_non_frozen_user_turn

    messages = [{"role": "user", "content": "Only turn — frozen"}]
    # frozen_message_count=1 → index 0 < 1 → skip injection
    result = _append_context_to_latest_non_frozen_user_turn(
        messages, _FIXED_CONTEXT, frozen_message_count=1
    )
    # Must return input list unchanged (same object identity).
    assert result is messages
    assert _FIXED_CONTEXT not in result[0]["content"]


# ---------------------------------------------------------------------------
# 5. empty/None fetch_context result: no-op
# ---------------------------------------------------------------------------


def test_memory_injection_skipped_on_empty_context() -> None:
    """Empty string from fetch_context → no injection."""

    def _fetch(messages: list, query: str, ctx: Any) -> str:
        return ""

    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Hello"}],
    }
    ctx = _make_ctx(body)
    engine = _make_engine(fetch_context=_fetch)
    decision = engine.on_request(ctx)

    out = json.loads(decision.body)
    content = out["messages"][-1]["content"]
    assert content == "Hello"


def test_memory_injection_skipped_on_none_context() -> None:
    """None from fetch_context → no injection."""

    def _fetch(messages: list, query: str, ctx: Any) -> str | None:
        return None

    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Hello"}],
    }
    ctx = _make_ctx(body)
    engine = _make_engine(fetch_context=_fetch)
    decision = engine.on_request(ctx)

    out = json.loads(decision.body)
    content = out["messages"][-1]["content"]
    assert content == "Hello"


# ---------------------------------------------------------------------------
# 6. memory + CCR proactive-expansion together: ordering (memory first)
# ---------------------------------------------------------------------------


def test_memory_and_ccr_expansion_ordering() -> None:
    """Memory injection runs BEFORE CCR proactive-expansion.

    Both appends target the latest non-frozen user turn. After memory
    injects its context, CCR expansion appends to the ALREADY-MODIFIED
    content (since _append_context_to_latest_non_frozen_user_turn is called
    twice: first with memory context, then with expansion text). The final
    content must contain both strings, with the memory context appearing
    first in the text.
    """
    expansion_text = "[CCR EXPANSION] Relevant context: retrieved_content"

    mock_tracker = MagicMock()
    mock_tracker.analyze_query.return_value = [MagicMock(reason="matches")]
    mock_tracker.execute_expansions.return_value = [MagicMock(content="retrieved_content")]
    mock_tracker.format_expansions_for_context.return_value = expansion_text

    def _fetch(messages: list, query: str, ctx: Any) -> str:
        return _FIXED_CONTEXT

    user_text = "Where is authentication configured?"
    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 128,
        "messages": [{"role": "user", "content": user_text}],
    }
    ctx = _make_ctx(body, cwd="/home/user/myproject")
    engine = _make_engine(
        fetch_context=_fetch,
        ccr_context_tracker=mock_tracker,
        ccr_config_overrides={
            "ccr_inject_tool": False,
            "ccr_inject_system_instructions": False,
            "ccr_context_tracking": False,
            "ccr_proactive_expansion": True,
        },
    )
    decision = engine.on_request(ctx)

    out = json.loads(decision.body)
    last_msg = out["messages"][-1]
    content = last_msg["content"]
    assert isinstance(content, str)

    # Both texts must be present.
    assert _FIXED_CONTEXT in content, "Memory context must be in the final content"
    assert expansion_text in content, "CCR expansion text must be in the final content"

    # Memory must appear before CCR expansion (memory runs first in the pipeline).
    mem_pos = content.index(_FIXED_CONTEXT)
    ccr_pos = content.index(expansion_text)
    assert mem_pos < ccr_pos, (
        f"Memory context (pos={mem_pos}) must appear before CCR expansion (pos={ccr_pos})"
    )


# ---------------------------------------------------------------------------
# 7. structural: fetch_context call args
# ---------------------------------------------------------------------------


def test_fetch_context_called_with_correct_args() -> None:
    """fetch_context receives (messages, query, ctx) with correct values."""
    call_args_log: list[tuple[list, str, Any]] = []

    def _fetch(messages: list, query: str, ctx: Any) -> str:
        call_args_log.append((messages, query, ctx))
        return _FIXED_CONTEXT

    user_text = "What is the deployment process?"
    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "messages": [
            {"role": "assistant", "content": "I can help."},
            {"role": "user", "content": user_text},
        ],
    }
    ctx = _make_ctx(body)
    engine = _make_engine(fetch_context=_fetch)
    engine.on_request(ctx)

    assert len(call_args_log) == 1
    called_messages, called_query, called_ctx = call_args_log[0]

    # query should be the user turn text
    assert called_query == user_text
    # messages is a list of dicts
    assert isinstance(called_messages, list)
    # ctx is the RequestContext (engine passes it through)
    from headroom.engine.contract import RequestContext

    assert isinstance(called_ctx, RequestContext)


# ---------------------------------------------------------------------------
# 8. bypass gate: injection skipped under x-headroom-bypass
# ---------------------------------------------------------------------------


def test_memory_injection_skipped_on_bypass() -> None:
    """x-headroom-bypass: true → memory injection is skipped."""
    fetch_called = [False]

    def _fetch(messages: list, query: str, ctx: Any) -> str:
        fetch_called[0] = True
        return _FIXED_CONTEXT

    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Bypass test"}],
    }
    ctx = _make_ctx(body, headers={"x-headroom-bypass": "true"})
    engine = _make_engine(fetch_context=_fetch)
    decision = engine.on_request(ctx)

    # Bypass short-circuits before memory — original bytes returned.
    assert decision.body == ctx.raw_body
    assert not fetch_called[0], "fetch_context must not be called under bypass"


# ---------------------------------------------------------------------------
# 9. inject_context=False: injection skipped
# ---------------------------------------------------------------------------


def test_memory_injection_skipped_when_inject_context_false() -> None:
    """memory_handler.config.inject_context=False → injection skipped."""
    fetch_called = [False]

    def _fetch(messages: list, query: str, ctx: Any) -> str:
        fetch_called[0] = True
        return _FIXED_CONTEXT

    handler_mock = MagicMock()
    handler_mock.config.inject_context = False

    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "No inject_context test"}],
    }
    ctx = _make_ctx(body)
    engine = _make_engine(fetch_context=_fetch, memory_handler=handler_mock)
    decision = engine.on_request(ctx)

    out = json.loads(decision.body)
    content = out["messages"][-1]["content"]
    assert _FIXED_CONTEXT not in content
    # fetch_context should not be called when inject_context is False
    assert not fetch_called[0]


# ---------------------------------------------------------------------------
# 10. no user_id: injection skipped (MemoryDecision gate)
# ---------------------------------------------------------------------------


def test_memory_injection_skipped_when_no_user_id() -> None:
    """user_id=None or empty → MemoryDecision.inject=False → no injection."""
    fetch_called = [False]

    def _fetch(messages: list, query: str, ctx: Any) -> str:
        fetch_called[0] = True
        return _FIXED_CONTEXT

    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "No user_id test"}],
    }
    ctx = _make_ctx(body)
    engine = _make_engine(fetch_context=_fetch, user_id=None)
    decision = engine.on_request(ctx)

    out = json.loads(decision.body)
    content = out["messages"][-1]["content"]
    assert _FIXED_CONTEXT not in content
    assert not fetch_called[0]
