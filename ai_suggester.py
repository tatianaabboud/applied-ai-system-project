"""PawPal+ AI layer: retrieval-augmented task suggestions.

Pipeline (one call per pet):
    knowledge/*.md  ->  retrieve()  ->  LLM  ->  validate  ->  Suggestion list
                                                                   |
                         owner reviews  ->  Suggestion.to_task()  ->  Pet.add_task()

The LLM only sees the passages that retrieve() picked for this pet, and every
suggestion must cite one of them. Anything that fails validation is dropped, and
any failure of the whole call falls back to "no suggestions", so the manual
scheduling flow always keeps working.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, time
from pathlib import Path
from typing import Callable, Optional

from pawpal_system import FREQUENCIES, Pet, Priority, Task

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
LOG_FILE = Path(__file__).parent / "pawpal_ai.log"
DEFAULT_MODEL = "claude-opus-5-5"

MAX_PASSAGES = 10
MAX_SUGGESTIONS = 8
MAX_DURATION_MINUTES = 240  # same cap as the UI's duration field
MAX_TITLE_LENGTH = 60

log = logging.getLogger("pawpal.ai")


def configure_logging() -> None:
    """Send AI log records to pawpal_ai.log (safe to call more than once)."""
    if any(isinstance(h, logging.FileHandler) for h in log.handlers):
        return
    handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)


# ---------------------------------------------------------------- knowledge base


@dataclass(frozen=True)
class Passage:
    """One retrievable guideline: a `##` section of a knowledge file."""
    source: str  # what a task cites, e.g. "dog.md, Puppy feeding"
    species: str  # dog | cat | other | all
    age: str  # puppy | kitten | young | adult | senior | all
    topic: str
    text: str


def load_passages(directory: Path = KNOWLEDGE_DIR) -> list[Passage]:
    """Parse every `##` section of every knowledge/*.md file into a Passage."""
    passages: list[Passage] = []
    for path in sorted(directory.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        for section in path.read_text(encoding="utf-8").split("\n## ")[1:]:
            heading, _, body = section.partition("\n")
            meta_line, _, text = body.strip().partition("\n")
            meta = {}
            for part in meta_line.split("|"):
                key, _, value = part.partition(":")
                meta[key.strip().lower()] = value.strip().lower()
            if not {"species", "age", "topic"} <= meta.keys():
                log.warning("skipping %s section %r: bad metadata line", path.name, heading)
                continue
            # The "Source:" line is bookkeeping; the LLM gets it separately.
            text = "\n".join(
                line for line in text.strip().splitlines() if not line.startswith("Source:")
            ).strip()
            passages.append(
                Passage(
                    source=f"{path.name}, {heading.strip()}",
                    species=meta["species"],
                    age=meta["age"],
                    topic=meta["topic"],
                    text=text,
                )
            )
    return passages


def life_stage(age_years: int) -> str:
    """Map an age in whole years to a life stage used by the knowledge base."""
    if age_years < 1:
        return "young"
    return "senior" if age_years >= 7 else "adult"


def _age_matches(passage_age: str, stage: str) -> bool:
    if passage_age == "all":
        return True
    if stage == "young":
        return passage_age in ("puppy", "kitten", "young")
    return passage_age == stage


def retrieve(pet: Pet, passages: Optional[list[Passage]] = None,
             limit: int = MAX_PASSAGES) -> list[Passage]:
    """Return the passages that apply to this pet, best match first.

    Plain keyword matching, no vector database. A passage must match the pet's
    species (or be for "all") and life stage (or be for "all"). Specific matches
    outrank general ones, and a breed word found in the text adds a bonus.
    """
    passages = load_passages() if passages is None else passages
    species = pet.species.strip().lower()
    if species not in ("dog", "cat"):
        species = "other"
    stage = life_stage(pet.age)
    breed_words = [w.rstrip("s") for w in pet.breed.lower().split() if len(w) > 3]

    scored = []
    for order, p in enumerate(passages):
        if p.species not in (species, "all") or not _age_matches(p.age, stage):
            continue
        score = (3 if p.species == species else 1) + (3 if p.age != "all" else 1)
        score += 2 * sum(1 for w in breed_words if w in p.text.lower())
        scored.append((-score, order, p))
    scored.sort(key=lambda s: s[:2])
    return [p for _, _, p in scored[:limit]]


# ------------------------------------------------------------------- suggestions


@dataclass
class Suggestion:
    title: str
    duration_minutes: int
    priority: Priority
    frequency: str
    preferred_time: Optional[time]
    source: str
    rationale: str

    def to_task(self) -> Task:
        """Build a Task for Pet.add_task(); the notes cite the knowledge source."""
        return Task(
            title=self.title,
            duration_minutes=self.duration_minutes,
            priority=self.priority,
            preferred_time=self.preferred_time,
            frequency=self.frequency,
            notes=f"{self.rationale} (Source: {self.source})",
        )


@dataclass
class SuggestionResult:
    suggestions: list[Suggestion] = field(default_factory=list)
    passages: list[Passage] = field(default_factory=list)  # what the LLM was shown
    error: Optional[str] = None  # set when we fell back to no suggestions


SUGGESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "duration_minutes": {"type": "integer"},
                    "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                    "frequency": {"type": "string", "enum": list(FREQUENCIES)},
                    "preferred_time": {"type": "string"},
                    "source": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["title", "duration_minutes", "priority", "frequency",
                             "preferred_time", "source", "rationale"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["tasks"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = f"""\
You plan daily care tasks for a pet-care scheduling app.
You are given a pet profile and numbered guideline passages. Suggest care tasks for \
this pet using ONLY what the passages say: take durations, number of times per day, \
and timing from them. When a passage asks for several feedings or walks a day, \
suggest each as its own task with a different preferred time.
Rules:
- At most {MAX_SUGGESTIONS} tasks. Do not repeat tasks the pet already has.
- duration_minutes is a whole number from 1 to {MAX_DURATION_MINUTES}.
- priority: high for feeding and medication, medium for exercise and vet visits, \
low for grooming and enrichment, unless a passage says otherwise.
- frequency is once, daily or weekly. Use "once" for a single vet visit.
- preferred_time is HH:MM (24-hour), or "" if the time does not matter.
- source must be copied exactly from the "Source" of the passage you used.
- rationale is one short sentence saying why, based on the pet's age and species.
Return only JSON matching the schema."""


def build_prompt(pet: Pet, passages: list[Passage]) -> str:
    """Compose the user message: pet profile, existing tasks and the passages."""
    lines = [
        "Pet profile:",
        f"- name: {pet.name}",
        f"- species: {pet.species}",
        f"- age: {pet.age} year(s) ({life_stage(pet.age)})",
        f"- breed: {pet.breed or 'unknown'}",
        "Existing tasks: " + (", ".join(t.title for t in pet.tasks) or "none"),
        "",
        "Guideline passages:",
    ]
    for i, p in enumerate(passages, 1):
        lines += [f"[{i}] Source: {p.source}", p.text, ""]
    return "\n".join(lines)


def call_claude(system: str, prompt: str) -> str:
    """Ask Claude for the JSON suggestions. Raises on any API problem."""
    import anthropic  # imported here so the rest of the app runs without it

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    response = client.beta.messages.create(
        model=os.environ.get("PAWPAL_MODEL", DEFAULT_MODEL),
        max_tokens=4000,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        output_config={
            "effort": "low",
            "format": {"type": "json_schema", "schema": SUGGESTION_SCHEMA},
        },
        # If a safety classifier declines, the API retries on a fallback model.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    log.info("claude request_id=%s stop_reason=%s", response._request_id, response.stop_reason)
    if response.stop_reason == "refusal":
        raise RuntimeError("the model declined this request")
    return next(b.text for b in response.content if b.type == "text")


def _parse_time(value: object) -> Optional[time]:
    text = str(value or "").strip()
    return datetime.strptime(text, "%H:%M").time() if text else None


def validate(raw: str, passages: list[Passage], existing_titles: list[str]) -> list[Suggestion]:
    """Turn the LLM's JSON text into safe Suggestions.

    The whole response is rejected if it isn't the expected JSON shape. Individual
    tasks are dropped (and logged) if a field is out of range, the source isn't one
    of the retrieved passages, or the title duplicates an existing or earlier task.
    """
    data = json.loads(raw)
    items = data["tasks"]
    if not isinstance(items, list):
        raise ValueError("'tasks' must be a list")

    sources = {p.source for p in passages}
    seen = {t.strip().lower() for t in existing_titles}
    accepted: list[Suggestion] = []
    for item in items[:MAX_SUGGESTIONS]:
        try:
            title = str(item["title"]).strip()
            duration = item["duration_minutes"]
            if isinstance(duration, bool) or not isinstance(duration, int):
                raise ValueError("duration_minutes must be an integer")
            if not 1 <= duration <= MAX_DURATION_MINUTES:
                raise ValueError(f"duration_minutes {duration} out of range")
            if not title or len(title) > MAX_TITLE_LENGTH:
                raise ValueError("bad title")
            if title.lower() in seen:
                raise ValueError("duplicate title")
            if item["priority"] not in ("low", "medium", "high"):
                raise ValueError(f"unknown priority {item['priority']!r}")
            if item["frequency"] not in FREQUENCIES:
                raise ValueError(f"unknown frequency {item['frequency']!r}")
            if item["source"] not in sources:
                raise ValueError(f"source {item['source']!r} was not retrieved")
            suggestion = Suggestion(
                title=title,
                duration_minutes=duration,
                priority=Priority[item["priority"].upper()],
                frequency=item["frequency"],
                preferred_time=_parse_time(item.get("preferred_time")),
                source=item["source"],
                rationale=str(item.get("rationale", "")).strip(),
            )
        except (KeyError, TypeError, ValueError) as err:
            log.warning("dropped suggestion %r: %s", item, err)
            continue
        seen.add(title.lower())
        accepted.append(suggestion)
    return accepted


def suggest_tasks(
    pet: Pet,
    llm: Callable[[str, str], str] = call_claude,
    passages: Optional[list[Passage]] = None,
) -> SuggestionResult:
    """Retrieve guidelines for `pet`, ask the LLM for tasks and validate them.

    Never raises: on any failure it returns an empty result with `error` set.
    `llm` takes (system_prompt, user_prompt) and returns JSON text.
    """
    try:
        retrieved = retrieve(pet, passages)
    except OSError as err:  # unreadable knowledge folder
        log.error("pet=%s retrieval failed: %s", pet.name, err)
        return SuggestionResult(error=f"Could not read the knowledge base: {err}")

    if not retrieved:
        log.info("pet=%s species=%s age=%s: no matching passages", pet.name, pet.species, pet.age)
        return SuggestionResult(error="No guidelines match this pet.")

    log.info("pet=%s species=%s age=%s breed=%r retrieved=%s", pet.name, pet.species, pet.age,
             pet.breed, [p.source for p in retrieved])
    try:
        raw = llm(SYSTEM_PROMPT, build_prompt(pet, retrieved))
        suggestions = validate(raw, retrieved, [t.title for t in pet.tasks])
    except Exception as err:  # API, network, key, JSON or schema problems
        log.error("pet=%s suggestion failed: %s: %s", pet.name, type(err).__name__, err)
        if "authentication method" in str(err):
            reason = "no API key found. Set the ANTHROPIC_API_KEY environment variable"
        else:
            reason = str(err)
        return SuggestionResult(passages=retrieved, error=f"AI suggestions unavailable: {reason}")

    log.info("pet=%s accepted=%d titles=%s", pet.name, len(suggestions),
             [s.title for s in suggestions])
    return SuggestionResult(suggestions=suggestions, passages=retrieved)
