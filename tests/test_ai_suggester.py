import json
from datetime import date, time

import pytest

import ai_suggester as ai
from pawpal_system import Owner, Pet, Priority, Scheduler


def sources(passages):
    return {p.source for p in passages}


def make_json(passages, **overrides):
    """One valid suggestion citing the first retrieved passage, with fields overridden."""
    task = {
        "title": "Evening walk", "duration_minutes": 15, "priority": "medium",
        "frequency": "daily", "preferred_time": "18:00",
        "source": passages[0].source, "rationale": "Short walks suit this pet.",
    }
    task.update(overrides)
    return json.dumps({"tasks": [task]})


# ---------------------------------------------------------------- knowledge base


def test_knowledge_base_loads_with_valid_metadata():
    passages = ai.load_passages()
    assert len(passages) >= 25
    for p in passages:
        assert p.species in ("dog", "cat", "other", "all")
        assert p.age in ("puppy", "kitten", "young", "adult", "senior", "all")
        assert p.text and p.source


def test_life_stage_boundaries():
    assert [ai.life_stage(a) for a in (0, 1, 6, 7, 15)] == [
        "young", "adult", "adult", "senior", "senior"]


# --------------------------------------------------------------------- retrieval


def test_puppy_gets_puppy_passages_not_senior_or_cat():
    found = ai.retrieve(Pet("Rex", "dog", age=0))
    assert "dog.md, Puppy feeding" in sources(found)
    assert "dog.md, Puppy exercise" in sources(found)
    assert not any("Senior dog" in s or s.startswith("cat.md") for s in sources(found))


def test_senior_dog_gets_senior_exercise_guideline():
    found = ai.retrieve(Pet("Old", "dog", age=11))
    assert "dog.md, Senior dog exercise" in sources(found)
    assert "dog.md, Adult dog exercise" not in sources(found)


def test_general_passages_apply_to_every_species():
    assert "medication.md, Give medication on a fixed schedule" in sources(
        ai.retrieve(Pet("Tweety", "other", age=2)))


def test_most_specific_passages_come_first():
    found = ai.retrieve(Pet("Rex", "dog", age=0), limit=3)
    assert all(p.species == "dog" and p.age == "puppy" for p in found)


def test_breed_keyword_boosts_matching_passage():
    plain = ai.retrieve(Pet("Rex", "dog", age=3), limit=1)
    retriever = ai.retrieve(Pet("Rex", "dog", age=3, breed="Golden Retriever"), limit=1)
    assert plain[0].source != "dog.md, Adult dog exercise"
    assert retriever[0].source == "dog.md, Adult dog exercise"


def test_unknown_species_falls_back_to_other_pets():
    found = ai.retrieve(Pet("Spike", "iguana", age=2))
    assert any(p.source.startswith("other_pets.md") for p in found)
    assert not any(p.source.startswith(("dog.md", "cat.md")) for p in found)


# ------------------------------------------------------------------- validation


@pytest.fixture
def puppy():
    return Pet("Rex", "dog", age=0)


@pytest.fixture
def passages(puppy):
    return ai.retrieve(puppy)


def test_valid_suggestion_becomes_task_with_cited_notes(puppy, passages):
    [s] = ai.validate(make_json(passages), passages, [])
    task = s.to_task()
    assert (task.title, task.duration_minutes, task.priority) == ("Evening walk", 15, Priority.MEDIUM)
    assert task.preferred_time == time(18, 0)
    assert f"Source: {passages[0].source}" in task.notes


@pytest.mark.parametrize("override", [
    {"duration_minutes": 0},
    {"duration_minutes": 999},
    {"duration_minutes": "20"},
    {"duration_minutes": True},
    {"priority": "urgent"},
    {"frequency": "hourly"},
    {"preferred_time": "25:99"},
    {"title": ""},
    {"title": "x" * 200},
    {"source": "made_up.md, Invented rule"},
])
def test_invalid_suggestion_is_dropped(passages, override):
    assert ai.validate(make_json(passages, **override), passages, []) == []


def test_duplicates_of_existing_or_earlier_tasks_are_dropped(passages):
    one = json.loads(make_json(passages))["tasks"][0]
    raw = json.dumps({"tasks": [one, dict(one), dict(one, title="Other task")]})
    titles = [s.title for s in ai.validate(raw, passages, [])]
    assert titles == ["Evening walk", "Other task"]
    assert ai.validate(make_json(passages), passages, ["evening WALK"]) == []


def test_bad_items_do_not_discard_good_ones(passages):
    good = json.loads(make_json(passages))["tasks"][0]
    raw = json.dumps({"tasks": [dict(good, duration_minutes=-5), good]})
    assert [s.title for s in ai.validate(raw, passages, [])] == ["Evening walk"]


def test_blank_preferred_time_means_no_time(passages):
    [s] = ai.validate(make_json(passages, preferred_time=""), passages, [])
    assert s.preferred_time is None


def test_suggestion_count_is_capped(passages):
    item = json.loads(make_json(passages))["tasks"][0]
    raw = json.dumps({"tasks": [dict(item, title=f"Task {i}") for i in range(20)]})
    assert len(ai.validate(raw, passages, [])) == ai.MAX_SUGGESTIONS


# -------------------------------------------------- suggest_tasks and its fallback


def test_suggest_tasks_end_to_end_with_mock_llm(puppy):
    seen = {}

    def fake_llm(system, prompt):
        seen["prompt"] = prompt
        passages = ai.retrieve(puppy)
        return make_json(passages, title="Puppy lunch", duration_minutes=10, priority="high")

    result = ai.suggest_tasks(puppy, llm=fake_llm)

    assert result.error is None
    assert [s.title for s in result.suggestions] == ["Puppy lunch"]
    # The model was shown the retrieved guideline text, not just a bare question.
    assert "3-4 small meals a day" in seen["prompt"]
    assert "Puppy feeding" in seen["prompt"]


def test_prompt_lists_existing_tasks_and_pet_profile():
    pet = Pet("Rex", "dog", age=0, breed="Beagle")
    pet.add_task(ai.Suggestion("Morning walk", 20, Priority.HIGH, "daily", None, "s", "r").to_task())
    prompt = ai.build_prompt(pet, ai.retrieve(pet))
    assert "Morning walk" in prompt and "Beagle" in prompt and "young" in prompt


@pytest.mark.parametrize("bad_output", ["not json at all", "[]", '{"tasks": "oops"}', '{"nope": 1}'])
def test_malformed_llm_output_falls_back_to_no_suggestions(puppy, bad_output):
    result = ai.suggest_tasks(puppy, llm=lambda s, p: bad_output)
    assert result.suggestions == []
    assert result.error


def test_llm_exception_falls_back_without_raising(puppy):
    def boom(system, prompt):
        raise RuntimeError("no API key")

    result = ai.suggest_tasks(puppy, llm=boom)
    assert result.suggestions == []
    assert "no API key" in result.error


def test_no_matching_passages_skips_the_llm(puppy):
    def must_not_be_called(system, prompt):
        raise AssertionError("LLM should not be called")

    result = ai.suggest_tasks(puppy, llm=must_not_be_called, passages=[])
    assert result.suggestions == [] and result.error


def test_suggest_tasks_logs_the_call(puppy, caplog):
    with caplog.at_level("INFO", logger="pawpal.ai"):
        ai.suggest_tasks(puppy, llm=lambda s, p: make_json(ai.retrieve(puppy)))
    text = caplog.text
    assert "retrieved=" in text and "accepted=1" in text


def test_validation_pass_rate_over_canned_responses(passages):
    """A small reliability check: the validator keeps good output, rejects bad."""
    good = json.loads(make_json(passages))["tasks"][0]
    cases = [dict(good, title=f"Good {i}") for i in range(5)] + [
        dict(good, title="Bad 1", duration_minutes=0),
        dict(good, title="Bad 2", source="nowhere.md, Nothing"),
        dict(good, title="Bad 3", priority="meh"),
    ]
    accepted = ai.validate(json.dumps({"tasks": cases}), passages, [])
    assert len(accepted) == 5
    assert all(s.title.startswith("Good") for s in accepted)


# ------------------------------------------------------- integration with scheduler


def test_accepted_suggestions_flow_into_the_scheduler(puppy):
    passages = ai.retrieve(puppy)
    raw = json.dumps({"tasks": [
        {"title": f"Meal {h}", "duration_minutes": 10, "priority": "high", "frequency": "daily",
         "preferred_time": f"{h}:00", "source": "dog.md, Puppy feeding", "rationale": "Puppies eat often."}
        for h in ("08", "12", "16", "20")
    ]})
    result = ai.suggest_tasks(puppy, llm=lambda s, p: raw, passages=passages)
    for s in result.suggestions:
        puppy.add_task(s.to_task())

    owner = Owner("Sam", available_minutes=120, pets=[puppy])
    items, skipped = Scheduler(owner).generate_plan(date.today())

    assert len(puppy.tasks) == 4 and not skipped
    assert all(i.task.notes.endswith("(Source: dog.md, Puppy feeding)") for i in items)
