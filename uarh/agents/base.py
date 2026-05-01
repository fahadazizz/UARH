"""Base agent abstraction — wraps LiteLLM with structured Pydantic output.

Every specialised agent inherits from ``BaseAgent`` and only needs to
define its system prompt, user-prompt builder, and the Pydantic model
it must return.  The base class handles:

* LLM invocation via ``litellm.completion``
* Automatic retry on malformed JSON (up to 2 retries)
* Structured output parsing into the target Pydantic model
* Logging of token usage and latency
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, Type, TypeVar

import litellm
from pydantic import BaseModel, ValidationError

from uarh.core.config import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class BaseAgent(ABC, Generic[T]):
    """Abstract base for all UARH cognitive agents.

    Subclasses must implement:
        * ``system_prompt``  — the agent's persona and rules.
        * ``build_user_message`` — constructs the user message from state.
        * ``output_schema``  — the Pydantic class to parse into.
    """

    # Maximum LLM retries when output JSON is malformed
    MAX_PARSE_RETRIES: int = 2

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.4,
    ) -> None:
        settings = get_settings()
        self._model = model or settings.elite_model
        self._temperature = temperature

    # ── Abstract Interface ─────────────────────────────────────

    @property
    @abstractmethod
    def agent_name(self) -> str:
        """Human-readable name for logging."""
        ...

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """The agent's full system prompt."""
        ...

    @abstractmethod
    def build_user_message(self, state: Dict[str, Any]) -> str:
        """Build the user-turn content from the current harness state."""
        ...

    @property
    @abstractmethod
    def output_schema(self) -> Type[T]:
        """The Pydantic model this agent must return."""
        ...

    # ── Core Invocation ────────────────────────────────────────

    def invoke(self, state: Dict[str, Any]) -> T:
        """Call the LLM and return a validated Pydantic object.

        Retries up to ``MAX_PARSE_RETRIES`` times if the output cannot
        be parsed into the target schema.
        """
        user_msg = self.build_user_message(state)
        schema_json = json.dumps(self.output_schema.model_json_schema(), indent=2)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg},
        ]

        last_error: Optional[str] = None

        for attempt in range(1, self.MAX_PARSE_RETRIES + 2):  # +1 for initial attempt
            try:
                t0 = time.perf_counter()
                response = litellm.completion(
                    model=self._model,
                    messages=messages,
                    temperature=self._temperature,
                    response_format={"type": "json_object"},
                    max_tokens=8192,
                )
                elapsed = time.perf_counter() - t0

                raw = response.choices[0].message.content
                usage = response.usage
                logger.info(
                    "[%s] LLM call #%d — %.1fs, %d prompt tokens, %d completion tokens",
                    self.agent_name,
                    attempt,
                    elapsed,
                    usage.prompt_tokens if usage else 0,
                    usage.completion_tokens if usage else 0,
                )

                # Parse JSON from response
                parsed_json = self._extract_json(raw)
                result = self.output_schema.model_validate(parsed_json)
                logger.info("[%s] Output validated successfully.", self.agent_name)
                return result

            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = str(exc)
                logger.warning(
                    "[%s] Parse error (attempt %d/%d): %s",
                    self.agent_name,
                    attempt,
                    self.MAX_PARSE_RETRIES + 1,
                    last_error,
                )
                # Add a correction message and retry
                messages.append({"role": "assistant", "content": raw if 'raw' in dir() else ""})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your response could not be parsed. Error: {last_error}\n\n"
                            f"You MUST respond with valid JSON matching this schema:\n{schema_json}\n\n"
                            "Respond ONLY with the corrected JSON object, no other text."
                        ),
                    }
                )

            except Exception as exc:
                logger.error("[%s] LLM invocation failed: %s", self.agent_name, exc)
                raise

        raise RuntimeError(
            f"[{self.agent_name}] Failed to produce valid output after "
            f"{self.MAX_PARSE_RETRIES + 1} attempts. Last error: {last_error}"
        )

    # ── Helpers ────────────────────────────────────────────────

    @staticmethod
    def _extract_json(raw: str) -> Dict[str, Any]:
        """Extract JSON from LLM output, handling markdown fences."""
        text = raw.strip()

        # Strip markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last fence lines
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        return json.loads(text)
