from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llm_gateway.auth.models import AuthenticatedUser
from llm_gateway.rate_limiting.denial_event import DENIAL_EVENT_NAME, PosthogDenialCapturer
from llm_gateway.rate_limiting.runner import ThrottleRunner
from llm_gateway.rate_limiting.throttles import Throttle, ThrottleContext, ThrottleResult


def _user(
    user_id: int = 42,
    team_id: int | None = 7,
    auth_method: str = "personal_api_key",
    distinct_id: str | None = None,
    application_id: str | None = "app-1",
) -> AuthenticatedUser:
    return AuthenticatedUser(
        user_id=user_id,
        team_id=team_id,
        auth_method=auth_method,
        distinct_id=distinct_id or f"distinct-{user_id}",
        scopes=["llm_gateway:read"],
        application_id=application_id,
    )


def _context(user: AuthenticatedUser | None = None, end_user_id: str | None = None) -> ThrottleContext:
    return ThrottleContext(user=user or _user(), product="posthog_code", end_user_id=end_user_id)


class _DeniedThrottle(Throttle):
    scope = "user_cost_burst"

    def __init__(self, result: ThrottleResult):
        self._result = result

    async def allow_request(self, context: ThrottleContext) -> ThrottleResult:
        return self._result


class TestRunnerCallsCapturer:
    @pytest.mark.asyncio
    async def test_capturer_called_on_denial(self) -> None:
        captured: list[tuple[ThrottleContext, ThrottleResult, str]] = []

        def capturer(ctx: ThrottleContext, result: ThrottleResult, scope: str) -> None:
            captured.append((ctx, result, scope))

        runner = ThrottleRunner(
            throttles=[
                _DeniedThrottle(
                    ThrottleResult.deny(
                        scope="user_cost_burst",
                        retry_after=120,
                        used_usd=12.5,
                        limit_usd=10.0,
                    )
                )
            ],
            denial_capturer=capturer,
        )
        await runner.check(_context())

        assert len(captured) == 1
        _, result, scope = captured[0]
        assert scope == "user_cost_burst"
        assert result.used_usd == 12.5
        assert result.limit_usd == 10.0

    @pytest.mark.asyncio
    async def test_capturer_not_called_when_allowed(self) -> None:
        calls = 0

        def capturer(ctx: ThrottleContext, result: ThrottleResult, scope: str) -> None:
            nonlocal calls
            calls += 1

        class AllowAll(Throttle):
            scope = "always_allow"

            async def allow_request(self, context: ThrottleContext) -> ThrottleResult:
                return ThrottleResult.allow()

        runner = ThrottleRunner(throttles=[AllowAll()], denial_capturer=capturer)
        await runner.check(_context())

        assert calls == 0

    @pytest.mark.asyncio
    async def test_capturer_exception_does_not_break_throttling(self) -> None:
        def broken(ctx: ThrottleContext, result: ThrottleResult, scope: str) -> None:
            raise RuntimeError("boom")

        runner = ThrottleRunner(
            throttles=[_DeniedThrottle(ThrottleResult.deny(scope="user_cost_burst", retry_after=1))],
            denial_capturer=broken,
        )
        # Should still surface the denial result, not raise.
        result = await runner.check(_context())
        assert result.allowed is False


class TestPosthogDenialCapturer:
    @pytest.fixture
    def capturer(self) -> PosthogDenialCapturer:
        return PosthogDenialCapturer(api_key="phc_test", host="https://us.i.posthog.com")

    def test_event_payload_for_personal_api_key(self, capturer: PosthogDenialCapturer) -> None:
        recorded: dict[str, Any] = {}

        def fake_sync(**kwargs: Any) -> None:
            recorded.update(kwargs)

        with patch.object(capturer, "_capture_sync", side_effect=fake_sync):
            capturer(
                _context(
                    user=_user(user_id=42, team_id=7, auth_method="personal_api_key"),
                    end_user_id="end-user-123",
                ),
                ThrottleResult.deny(
                    scope="user_cost_burst",
                    retry_after=60,
                    used_usd=15.25,
                    limit_usd=10.0,
                ),
                scope="user_cost_burst",
            )

        assert recorded["event"] == DENIAL_EVENT_NAME
        # personal_api_key path prefers end_user_id.
        assert recorded["distinct_id"] == "end-user-123"
        props = recorded["properties"]
        assert props["product"] == "posthog_code"
        assert props["scope"] == "user_cost_burst"
        assert props["used_usd"] == 15.25
        assert props["limit_usd"] == 10.0
        assert props["retry_after_seconds"] == 60
        assert props["team_id"] == 7
        assert props["auth_method"] == "personal_api_key"
        assert props["gateway_user_id"] == 42
        assert props["application_id"] == "app-1"
        assert recorded["groups"] == {"project": 7}

    def test_oauth_uses_auth_distinct_id(self, capturer: PosthogDenialCapturer) -> None:
        recorded: dict[str, Any] = {}

        def fake_sync(**kwargs: Any) -> None:
            recorded.update(kwargs)

        with patch.object(capturer, "_capture_sync", side_effect=fake_sync):
            capturer(
                _context(
                    user=_user(
                        user_id=1,
                        auth_method="oauth_access_token",
                        distinct_id="oauth-distinct",
                    ),
                    end_user_id="ignored",
                ),
                ThrottleResult.deny(scope="user_cost_sustained", retry_after=10),
                scope="user_cost_sustained",
            )
        assert recorded["distinct_id"] == "oauth-distinct"

    def test_no_team_id_omits_groups(self, capturer: PosthogDenialCapturer) -> None:
        recorded: dict[str, Any] = {}

        def fake_sync(**kwargs: Any) -> None:
            recorded.update(kwargs)

        with patch.object(capturer, "_capture_sync", side_effect=fake_sync):
            capturer(
                _context(user=_user(team_id=None)),
                ThrottleResult.deny(scope="user_cost_burst", retry_after=5),
                scope="user_cost_burst",
            )
        assert "groups" not in recorded
        assert "team_id" not in recorded["properties"]

    def test_capture_sync_swallows_posthog_failures(self, capturer: PosthogDenialCapturer) -> None:
        broken_client = MagicMock()
        broken_client.capture.side_effect = RuntimeError("network down")
        with patch("llm_gateway.rate_limiting.denial_event.Posthog", return_value=broken_client):
            # Must not raise.
            capturer._capture_sync(distinct_id="d", event=DENIAL_EVENT_NAME, properties={})
        broken_client.shutdown.assert_called_once()
