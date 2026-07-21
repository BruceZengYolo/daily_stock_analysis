from types import SimpleNamespace

import pytest

from src.analyzer import GeminiAnalyzer, _AllModelsFailedError


def _analyzer_without_init() -> GeminiAnalyzer:
    analyzer = GeminiAnalyzer.__new__(GeminiAnalyzer)
    import threading

    analyzer._quota_circuit_lock = threading.Lock()
    analyzer._quota_circuit_provider = None
    return analyzer


def test_quota_circuit_opens_for_same_provider_hard_quota_error() -> None:
    analyzer = _analyzer_without_init()

    analyzer._open_quota_circuit_if_safe(
        models=["gemini/gemini-3-flash-preview", "gemini/gemini-3.1-pro-preview"],
        model_list=[],
        error=RuntimeError("You exceeded your current quota; check billing details"),
    )

    assert analyzer._quota_circuit_provider == "gemini"
    with pytest.raises(_AllModelsFailedError, match="API call skipped"):
        analyzer._raise_if_quota_circuit_open()


def test_quota_circuit_stays_closed_for_transient_rate_limit() -> None:
    analyzer = _analyzer_without_init()

    analyzer._open_quota_circuit_if_safe(
        models=["gemini/gemini-3-flash-preview"],
        model_list=[],
        error=RuntimeError("429 too many requests; retry after 2 seconds"),
    )

    assert analyzer._quota_circuit_provider is None


def test_quota_circuit_stays_closed_for_cross_provider_fallback() -> None:
    analyzer = _analyzer_without_init()

    analyzer._open_quota_circuit_if_safe(
        models=["gemini/gemini-3-flash-preview", "openai/gpt-5-mini"],
        model_list=[],
        error=RuntimeError("insufficient_quota"),
    )

    assert analyzer._quota_circuit_provider is None
