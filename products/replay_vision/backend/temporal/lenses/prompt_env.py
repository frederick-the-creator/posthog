"""Jinja2 environment for lens prompts.

Autoescape is on, but uses a custom escape that replaces `<` with `<` (JSON-style
Unicode escape) instead of the HTML `&lt;` jinja2 defaults to. The threat being mitigated
is user content forging delimiter tags (`</events>`, `</lens_intent>`, …); the escape
form is chosen to read naturally in an LLM prompt rather than as HTML entities.

Jinja2 doesn't expose a public escape-function override, so we route through `finalize`
(called on every `{{ var }}` value) and return `Markup` so jinja2's HTML autoescape is
bypassed in favor of ours.

Templates live in `prompts/`. Each lens type has its own `<lens_type>.jinja` extending
`base.jinja` and filling the `task` block.
"""

from typing import Any

from jinja2 import Environment, PackageLoader, StrictUndefined
from markupsafe import Markup


def _prompt_escape(value: Any) -> Markup:
    """Escape `<` so user content can't forge a delimiter tag inside the prompt."""
    if isinstance(value, Markup):
        # Already escaped (e.g. `tojson` output) — skip the full-string copy + replace.
        return value
    return Markup(str(value).replace("<", "\\u003c"))


_env = Environment(
    loader=PackageLoader("products.replay_vision.backend.temporal.lenses", "prompts"),
    autoescape=True,
    trim_blocks=False,
    lstrip_blocks=False,
    keep_trailing_newline=True,
    undefined=StrictUndefined,
    finalize=_prompt_escape,
)
# Compact `tojson` output — Gemini parses fine without whitespace, and indent burns prompt tokens.
_env.policies["json.dumps_kwargs"] = {"separators": (",", ":")}


def render_prompt(template_name: str, **context: Any) -> str:
    """Render a lens prompt template with the given context."""
    return _env.get_template(template_name).render(**context)
