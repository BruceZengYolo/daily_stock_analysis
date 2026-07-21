from unittest.mock import Mock, patch

from src.search_service import GDELTSearchProvider


def _response(status: int, payload=None):
    response = Mock(status_code=status)
    response.json.return_value = payload or {}
    return response


def test_gdelt_maps_article_list_response() -> None:
    provider = GDELTSearchProvider(enabled=True)
    provider._blocked_until = 0
    type(provider)._next_request_at = 0
    with patch(
        "src.search_service.requests.get",
        return_value=_response(
            200,
            {
                "articles": [
                    {
                        "title": "Company reports earnings",
                        "url": "https://example.com/story",
                        "domain": "example.com",
                        "seendate": "20260721T120000Z",
                    }
                ]
            },
        ),
    ):
        result = provider.search("Company stock", max_results=3, days=3)

    assert result.success is True
    assert result.provider == "GDELT"
    assert result.results[0].published_date == "2026-07-21"


def test_gdelt_429_opens_cooldown_and_skips_next_network_call() -> None:
    provider = GDELTSearchProvider(enabled=True)
    provider._blocked_until = 0
    type(provider)._next_request_at = 0
    with patch("src.search_service.requests.get", return_value=_response(429)) as request:
        first = provider.search("Company stock")
        second = provider.search("Other company stock")

    assert first.success is False
    assert "429" in (first.error_message or "")
    assert "cooldown" in (second.error_message or "")
    request.assert_called_once()
