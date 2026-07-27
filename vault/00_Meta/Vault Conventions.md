# Vault Conventions

Last reviewed: 2026-07-26

## Authority

Use this order when sources disagree:

1. live code, Git state, inspected data, tests, and regenerated results;
2. `01_Project/Current State.md` and
   `05_Handoff/Handoff - Latest.md`;
3. durable decisions and project notes;
4. dated session history.

Use explicit labels where ambiguity matters:

- **Confirmed:** supported by live evidence or a direct user statement;
- **Hypothesis:** plausible but not adequately verified;
- **Superseded:** retained because it explains history but is no longer
  current.

## Note ownership

- `01_Project/Project Brief.md`: stable purpose, scope, dependencies, and code
  map.
- `01_Project/Current State.md`: currently verified technical snapshot.
- `02_Algorithm/`: optional durable scientific interpretation.
- `03_Decisions/Decision Log.md`: durable decisions and rationale.
- `04_Sessions/`: dated history only when a meaningful checkpoint merits it.
- `05_Handoff/Handoff - Latest.md`: concise next action, risks, verification
  entry points, and structured `Portfolio impact`.

Link to the note that owns a detail rather than copying the same narrative.

## Cross-session synchronization

Project agents update this vault automatically after durable milestones. They
do not edit the central second brain.

At every meaningful checkpoint, refresh the latest handoff with:

- `Central update needed: Yes` or `No`;
- portfolio-level changes limited to status, priority, deadline, next
  decision, or research significance;
- a one-to-three-sentence sync summary.

Skip vault updates for transient exploration, unchanged state, and explicitly
read-only work.

## Writing rules

- Replace obsolete live state instead of appending a diary.
- Preserve concise evidence and repository-relative paths.
- Do not paste raw logs, large diffs, or terminal transcripts.
- Never add later work to an older dated session note.
- Do not claim publication, model, or validation progress from a
  documentation-only change.
- Ignore local Obsidian state under `.obsidian/`; version the project memory.
