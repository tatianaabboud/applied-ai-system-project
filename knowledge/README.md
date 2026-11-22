# PawPal+ knowledge base

Short pet-care guidelines that the retriever searches before the LLM suggests tasks.
These are general guidelines for planning routines, not veterinary advice.

## File format

- One topic per file. Each `##` heading is one retrievable passage.
- Every passage starts with a metadata line: `species: ... | age: ... | topic: ...`
  - `species` is `dog`, `cat`, `other` or `all`.
  - `age` is `puppy`/`kitten`/`young` (under 1 year), `adult` (1-7 years), `senior` (7+ years) or `all`.
  - `topic` is one of `feeding`, `exercise`, `grooming`, `medication`, `vaccination`, `enrichment`.
- The text after it states the guideline with concrete numbers (minutes, times per day) so it can become a task.
- The `Source:` line is what gets cited in a task's notes.

| File | Contents |
|------|----------|
| `dog.md` | Feeding, exercise, grooming and enrichment for dogs by age |
| `cat.md` | Feeding, play, grooming and enrichment for cats by age |
| `other_pets.md` | General routines for small pets and other species |
| `medication.md` | Timing rules for giving medication |
| `vaccinations.md` | Typical vaccination and checkup schedules |
