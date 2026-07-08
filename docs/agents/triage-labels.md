# Triage Labels

Label vocabulary for the triage state machine. These strings appear in issue front-matter (`status:` field) and are used by the `triage` skill to move issues through their lifecycle.

## Label mapping

| Role | Label string | Meaning |
|---|---|---|
| Needs triage | `needs-triage` | Maintainer needs to evaluate this issue |
| Needs info | `needs-info` | Waiting on the reporter for more details or reproduction steps |
| Ready for agent | `ready-for-agent` | Fully specified — an AI agent can pick this up with no additional human context |
| Ready for human | `ready-for-human` | Needs human implementation (too complex, too risky, or requires judgment) |
| Won't fix | `wontfix` | Will not be actioned — out of scope, duplicate, or not reproducible |

## How the triage skill uses these

1. New issues start at `needs-triage`.
2. The triage skill evaluates the issue and moves it to one of the other four states.
3. `ready-for-agent` issues can be picked up by autonomous agent skills (e.g., `tdd`, `diagnose`).
4. `ready-for-human` issues are left for a human to implement.
5. `needs-info` issues are parked until the reporter responds.

## Customizing

Edit the "Label string" column above to match your preferred vocabulary. The triage skill reads this file at runtime — no code changes needed.
