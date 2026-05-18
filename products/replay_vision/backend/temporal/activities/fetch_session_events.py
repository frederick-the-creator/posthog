import re
import hashlib
from typing import Any

from asgiref.sync import sync_to_async
from temporalio import activity
from temporalio.exceptions import ApplicationError

from posthog.models import Team
from posthog.session_recordings.queries.session_replay_events import SessionReplayEvents

from products.replay_vision.backend.temporal.constants import (
    MAX_ACTIVE_SECONDS_FOR_VIDEO_LENS_S,
    MIN_ACTIVE_SECONDS_FOR_VIDEO_LENS_S,
    MIN_SESSION_DURATION_FOR_VIDEO_LENS_S,
)
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
_MAX_EVENT_PAGES = 10  # Hard cap on prompt size for very chatty sessions; sets `events_truncated` when reached.

# Noisy SDK-internal events that add no signal for the LLM.
_EVENTS_TO_IGNORE = ["$feature_flag_called"]

# Fetched only to power `_skip_exception_without_valid_context`; dropped before the row reaches the LLM.
_EXTRA_FIELDS_FOR_FILTER = [
    "properties.$exception_fingerprint_record",
    "properties.$exception_functions",
    "properties.$exception_sources",
]
# Bare column names (HogQL strips the `properties.` prefix) that are excluded from the prompt projection.
_FILTER_ONLY_COLUMNS = frozenset({"$exception_fingerprint_record", "$exception_functions", "$exception_sources"})

# `properties.*` is the HogQL-side prefix for JSON-property fields; bare names refer to top-level columns.
_EXTRA_FIELDS = [
    "elements_chain_ids",
    "properties.$exception_types",
    "properties.$exception_values",
    *_EXTRA_FIELDS_FOR_FILTER,
]

# Keywords that suggest a network/auth-class exception worth surfacing even without a wide fingerprint count.
_EXCEPTION_KEYWORD_RE = re.compile(
    r"api|http|fetch|request|post|put|delete|response|xhr|ajax|graphql|socket|websocket|auth|token|login",
    re.IGNORECASE,
)
_MIN_EXCEPTION_FINGERPRINTS = 5

# Token names for URL and window-id simplification — referenced by `base.jinja`'s resolver instructions.
_URL_PREFIX = "url"
_WINDOW_PREFIX = "window"
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
    duration_seconds = float(metadata["duration"])
    if duration_seconds < MIN_SESSION_DURATION_FOR_VIDEO_LENS_S:
        raise ApplicationError(
            f"Session {session_id} is only {duration_seconds}s long; min is {MIN_SESSION_DURATION_FOR_VIDEO_LENS_S}s",
            non_retryable=True,
        )
    active_seconds = metadata.get("active_seconds")
    if active_seconds is not None and active_seconds < MIN_ACTIVE_SECONDS_FOR_VIDEO_LENS_S:
        raise ApplicationError(
            f"Session {session_id} has only {active_seconds}s of active interaction; min is {MIN_ACTIVE_SECONDS_FOR_VIDEO_LENS_S}s",
            non_retryable=True,
        )
    if metadata["active_seconds"] > MAX_ACTIVE_SECONDS_FOR_VIDEO_LENS_S:
        raise ApplicationError(
            f"Session {session_id} has {metadata['active_seconds']}s of active interaction; max is {MAX_ACTIVE_SECONDS_FOR_VIDEO_LENS_S}s",
            non_retryable=True,
        )

    columns: list[str] | None = None
    all_rows: list[list[Any]] = []
    events_truncated = False
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
        if page == _MAX_EVENT_PAGES - 1:
            # Last page filled — more events exist that we won't fetch. Tell the LLM so it can adjust confidence.
            events_truncated = True

    if columns is None or not all_rows:
        return None

    processed_columns, processed_rows, url_mapping, window_mapping = _process_events(columns, all_rows)
    # `RecordingMetadata` doesn't carry inactive_seconds directly; derive from duration minus active.
    inactive_seconds = int(duration_seconds - active_seconds) if active_seconds is not None else None

    return LensLlmInputs(
        session_id=session_id,
        team_id=team_id,
        session_end_time=metadata["end_time"],
        events=EventTable(columns=processed_columns, rows=processed_rows),
        url_mapping=url_mapping,
        window_mapping=window_mapping,
        metadata=SessionMetadata(
            start_time=metadata["start_time"],
            duration_seconds=duration_seconds,
            active_seconds=active_seconds,
            inactive_seconds=inactive_seconds,
            click_count=metadata.get("click_count"),
            keypress_count=metadata.get("keypress_count"),
            mouse_activity_count=metadata.get("mouse_activity_count"),
            start_url=metadata.get("first_url"),
            console_error_count=metadata.get("console_error_count"),
            events_truncated=events_truncated,
        ),
    )


def _process_events(
    raw_columns: list[str], raw_rows: list[list[Any]]
) -> tuple[list[str], list[list[Any]], dict[str, str], dict[str, str]]:
    """Skip-filter, dedup, truncate, intern, then project out filter-only columns; prepend event_id + event_index."""
    column_indexes = {col: i for i, col in enumerate(raw_columns)}
    url_index = column_indexes.get("$current_url")
    window_index = column_indexes.get("$window_id")
    keep_indexes = [i for i, col in enumerate(raw_columns) if col not in _FILTER_ONLY_COLUMNS]

    url_tokens: dict[str, str] = {}  # actual -> token; flipped at the end for the prompt
    window_tokens: dict[str, str] = {}
    seen_hashes: set[str] = set()
    processed: list[list[Any]] = []

    for index, row in enumerate(raw_rows):
        if _should_skip_event(row, column_indexes):
            continue
        # Hash after the skip filter but before truncation so dup-collapse uses the raw bytes.
        raw_hash = _row_hash(row)
        if raw_hash in seen_hashes:
            continue
        seen_hashes.add(raw_hash)

        simplified = [_truncate(v) for v in row]
        if url_index is not None:
            simplified[url_index] = _intern(simplified[url_index], url_tokens, _URL_PREFIX)
        if window_index is not None:
            simplified[window_index] = _intern(simplified[window_index], window_tokens, _WINDOW_PREFIX)
        projected = [simplified[i] for i in keep_indexes]
        processed.append([raw_hash, index, *projected])

    output_columns = ["event_id", "event_index", *(raw_columns[i] for i in keep_indexes)]
    url_mapping = {token: actual for actual, token in url_tokens.items()}
    window_mapping = {token: actual for actual, token in window_tokens.items()}
    return output_columns, processed, url_mapping, window_mapping


def _should_skip_event(row: list[Any], indexes: dict[str, int]) -> bool:
    """Mirror session_summary's silent-skip heuristics: drop exceptions without actionable context and empty $autocapture."""
    event_idx = indexes.get("event")
    if event_idx is None:
        return False
    event = row[event_idx]
    if event == "$exception":
        return _skip_exception_without_valid_context(row, indexes)
    return _skip_event_without_valid_context(row, indexes)


def _skip_exception_without_valid_context(row: list[Any], indexes: dict[str, int]) -> bool:
    """Skip exceptions with <5 fingerprints AND no API/auth-class keyword in functions/sources/values."""
    fingerprints = _list_column(row, indexes.get("$exception_fingerprint_record"))
    if len(fingerprints) >= _MIN_EXCEPTION_FINGERPRINTS:
        return False
    for column in ("$exception_functions", "$exception_sources", "$exception_values"):
        values = _list_column(row, indexes.get(column))
        if any(isinstance(v, str) and _EXCEPTION_KEYWORD_RE.search(v) for v in values):
            return False
    return True


def _skip_event_without_valid_context(row: list[Any], indexes: dict[str, int]) -> bool:
    """Skip single-token `$autocapture` with no element context; multi-word names and other system events stay."""
    event_idx = indexes.get("event")
    if event_idx is None:
        return False
    event = row[event_idx]
    if not isinstance(event, str):
        return False
    # Descriptive multi-token event names carry their own context.
    if " " in event or "." in event or "_" in event:
        return False
    for column in ("elements_chain_texts", "elements_chain_elements", "elements_chain_href", "elements_chain_ids"):
        idx = indexes.get(column)
        if idx is not None and row[idx]:
            return False
    # Only context-less `$autocapture` is dropped; `$pageview`, `$set`, etc. stay.
    return event == "$autocapture"


def _list_column(row: list[Any], idx: int | None) -> list[Any]:
    """Read an `Array(...)` column safely — HogQL hands these back as Python lists."""
    if idx is None:
        return []
    value = row[idx]
    return value if isinstance(value, list) else []


def _intern(value: Any, mapping: dict[str, str], prefix: str) -> Any:
    """Replace string `value` with a `<prefix>_N` token, mutating `mapping` (actual -> token); non-strings pass through."""
    if not isinstance(value, str):
        return value
    if value in mapping:
        return mapping[value]
    token = f"{prefix}_{len(mapping) + 1}"
    mapping[value] = token
    return token


def _truncate(value: Any) -> Any:
    """Cap a single field so one oversized value (long stack trace, big elements_chain list) can't blow the prompt."""
    if isinstance(value, str) and len(value) > _MAX_FIELD_LEN:
        return value[:_MAX_FIELD_LEN] + "…[truncated]"
    if isinstance(value, list):
        return [_truncate(v) for v in value]
    return value


def _row_hash(row: list[Any]) -> str:
    """Deterministic 16-char (64-bit) hex of the row contents; identical events collapse to the same id."""
    # `repr` quotes strings and renders `None` as `None`, so a literal "" and a None column don't collide.
    joined = "\0".join(repr(v) for v in row)
    return hashlib.sha256(joined.encode()).hexdigest()[: _EVENT_ID_BYTES * 2]
