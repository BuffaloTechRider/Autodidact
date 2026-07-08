# Issue Tracker — Local Markdown

Issues for this repo are tracked as local markdown files, not in a remote issue tracker.

## Layout

```
.scratch/
  <feature-or-bug-slug>/
    issue.md          # the issue description (front-matter + body)
    notes.md          # optional working notes, investigation logs
    patch.diff        # optional patch if the fix is ready
```

## Creating an issue

Create a directory under `.scratch/` named after the feature or bug (kebab-case). Write `issue.md` with YAML front-matter:

```markdown
---
title: Short descriptive title
status: needs-triage
labels: []
created: 2026-05-01
---

## Description

What needs to happen and why.

## Acceptance criteria

- [ ] Criterion 1
- [ ] Criterion 2
```

## Status values

The `status` field in front-matter uses the triage label vocabulary defined in `docs/agents/triage-labels.md`.

## Conventions

- One issue per directory under `.scratch/`.
- The directory name is the issue slug (used in commit messages, branch names).
- When an issue is resolved, move its directory to `.scratch/_done/` or delete it.
- Issues are not committed to the main branch — `.scratch/` is gitignored or kept on a working branch.
