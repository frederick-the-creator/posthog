import hashlib
from typing import Any

from asgiref.sync import sync_to_async
from temporalio import activity
from temporalio.exceptions import ApplicationError

from posthog.models import Team
from posthog.session_recordings.queries.session_replay_events import SessionReplayEvents

from products.replay_vision.backend.temporal.constants import MAX_ACTIVE_SECONDS_FOR_VIDEO_LENS_S
from products.replay_vision.backend.temporal.state import (
    StateActivitiesEnum,
    get_redis_state_client,
    store_data_in_redis,
)
from products.replay_vision.backend.temporal.types import (
    EventTable,
    FetchSessionEventsInputs,
    LensLlmInputs,
    SessionMetadata,
)

# Pagination shape mirrors session_summary's fetcher; without it HogQL applies LimitContext.QUERY's default of 100.
_EVENTS_PER_PAGE = 3000
_MAX_EVENT_PAGES = 100

# Noisy SDK-internal events that add no signal for the LLM.
_EVENTS_TO_IGNORE = ["$feature_flag_called"]

# `properties.*` is the HogQL-side prefix for JSON-property fields; bare names refer to top-level columns.
# We deliberately omit `elements_chain`, `$exception_sources`, `$exception_fingerprint_record`,
# `$exception_functions`, `uuid` from session_summary's `EXTRA_SUMMARY_EVENT_FIELDS` — too noisy or
# only useful as a server-side lookup key.
_EXTRA_FIELDS = ["elements_chain_ids", "properties.$exception_types", "properties.$exception_values"]

# Token names for URL and window-id simplification — referenced by `base.jinja`'s resolver instructions.
_URL_PREFIX = "url"
_WINDOW_PREFIX = "window"
# Cap how many distinct URLs/windows we'll intern; beyond this, leave the raw value in place.
# Long SPA sessions with UUIDs in the path can otherwise produce huge mapping tables.
_MAX_URL_INTERN = 500
_MAX_WINDOW_INTERN = 50
# Per-value cap to keep one oversized field from blowing the prompt token budget.
_MAX_FIELD_LEN = 2000
# 64-bit hex hash for event dedup — 32-bit (8 hex) hits 50% birthday-collision around 77k events.
_EVENT_ID_BYTES = 8


@activity.defn
async def fetch_session_events_activity(inputs: FetchSessionEventsInputs) -> None:
    """Fetch analytics events for a session and stash in Redis; idempotent — a second call finds the key and returns."""
    redis_client, redis_key = get_redis_state_client(
        label=StateActivitiesEnum.SESSION_EVENTS,
        state_id=str(inputs.observation_id),
    )
    if await redis_client.exists(redis_key):
        return

    payload = await sync_to_async(_fetch_payload)(inputs.team_id, inputs.session_id)
    if payload is None:
        raise ApplicationError(
            f"Session {inputs.session_id} has no events to analyze",
            non_retryable=True,
        )

    await store_data_in_redis(redis_client, redis_key, payload.model_dump_json())


def _fetch_payload(team_id: int, session_id: str) -> LensLlmInputs | None:
    team = Team.objects.get(pk=team_id)
    events_obj = SessionReplayEvents()
    metadata = events_obj.get_metadata(session_id=session_id, team=team)
    if metadata is None:
        raise ApplicationError(f"No replay metadata found for session {session_id}", non_retryable=True)
    if metadata["active_seconds"] > MAX_ACTIVE_SECONDS_FOR_VIDEO_LENS_S:
        raise ApplicationError(
            f"Session {session_id} has {metadata['active_seconds']}s of active interaction; max is {MAX_ACTIVE_SECONDS_FOR_VIDEO_LENS_S}s",
            non_retryable=True,
        )

    columns: list[str] | None = None
    all_rows: list[list[Any]] = []
    for page in range(_MAX_EVENT_PAGES):
        page_columns, page_rows = events_obj.get_events(
            session_id=session_id,
            team=team,
            metadata=metadata,
            events_to_ignore=_EVENTS_TO_IGNORE,
            extra_fields=_EXTRA_FIELDS,
            limit=_EVENTS_PER_PAGE,
            page=page,
        )
        if page_columns and columns is None:
            columns = list(page_columns)
        if not page_rows:
            break
        all_rows.extend(list(row) for row in page_rows)
        if len(page_rows) < _EVENTS_PER_PAGE:
            break

    if columns is None or not all_rows:
        return None

    processed_columns, processed_rows, url_mapping, window_mapping = _process_events(columns, all_rows)

    return LensLlmInputs(
        session_id=session_id,
        team_id=team_id,
        session_end_time=metadata["end_time"],
        events=EventTable(columns=processed_columns, rows=processed_rows),
        url_mapping=url_mapping,
        window_mapping=window_mapping,
        metadata=SessionMetadata(
            start_time=metadata["start_time"],
            duration_seconds=float(metadata["duration"]),
            active_seconds=metadata.get("active_seconds"),
            inactive_seconds=metadata.get("inactive_seconds"),
            click_count=metadata.get("click_count"),
            keypress_count=metadata.get("keypress_count"),
            mouse_activity_count=metadata.get("mouse_activity_count"),
            start_url=metadata.get("start_url"),
            console_error_count=metadata.get("console_error_count"),
        ),
    )


def _process_events(
    raw_columns: list[str], raw_rows: list[list[Any]]
) -> tuple[list[str], list[list[Any]], dict[str, str], dict[str, str]]:
    """Truncate oversized fields, simplify URLs and window IDs, dedup by content hash, prepend event_id + event_index.

    Returns (columns, rows, url_mapping, window_mapping) with mappings as short->actual
    so the LLM can resolve `url_1` / `window_1` tokens back to source values.
    """
    url_index = raw_columns.index("$current_url") if "$current_url" in raw_columns else None
    window_index = raw_columns.index("$window_id") if "$window_id" in raw_columns else None

    url_interner = _Interner(_URL_PREFIX, _MAX_URL_INTERN)
    window_interner = _Interner(_WINDOW_PREFIX, _MAX_WINDOW_INTERN)
    seen_hashes: set[str] = set()
    processed: list[list[Any]] = []

    for index, row in enumerate(raw_rows):
        # Hash the raw row first so duplicate-row rejection short-circuits the more expensive work below.
        raw_hash = _row_hash(row)
        if raw_hash in seen_hashes:
            continue
        seen_hashes.add(raw_hash)

        simplified = [_truncate(v) for v in row]
        if url_index is not None:
            simplified[url_index] = url_interner.intern(simplified[url_index])
        if window_index is not None:
            simplified[window_index] = window_interner.intern(simplified[window_index])
        processed.append([raw_hash, index, *simplified])

    columns = ["event_id", "event_index", *raw_columns]
    return columns, processed, url_interner.mapping, window_interner.mapping


class _Interner:
    """Map distinct string values to short `<prefix>_N` tokens, capped at `max_entries`.

    Past the cap, the raw value passes through so the mapping table itself doesn't blow up
    on sessions with many unique URLs. Non-string inputs pass through unchanged.
    """

    __slots__ = ("_prefix", "_max", "mapping", "_reverse")

    def __init__(self, prefix: str, max_entries: int) -> None:
        self._prefix = prefix
        self._max = max_entries
        self.mapping: dict[str, str] = {}  # token -> actual; serialized into the prompt
        self._reverse: dict[str, str] = {}  # actual -> token; lookup hot-path

    def intern(self, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        if value in self._reverse:
            return self._reverse[value]
        if len(self.mapping) >= self._max:
            return value
        token = f"{self._prefix}_{len(self.mapping) + 1}"
        self.mapping[token] = value
        self._reverse[value] = token
        return token


def _truncate(value: Any) -> Any:
    """Cap a single string field so one oversized value (long stack trace, big elements_chain) can't blow the prompt."""
    if isinstance(value, str) and len(value) > _MAX_FIELD_LEN:
        return value[:_MAX_FIELD_LEN] + "…[truncated]"
    return value


def _row_hash(row: list[Any]) -> str:
    """Deterministic 16-char (64-bit) hex of the row contents; identical events collapse to the same id."""
    joined = "\0".join(str(v) if v is not None else "" for v in row)
    return hashlib.sha256(joined.encode()).hexdigest()[: _EVENT_ID_BYTES * 2]
