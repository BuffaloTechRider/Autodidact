"""Multi-step tool-task corpus for the A7 baseline (NFR-2 ROI gate).

A7 asks: on *real multi-step tasks*, what is the cloud/local/memory split, and
is step difficulty *separable* (a task cleanly splits into cheap + hard steps)
or *entangled* (every step needs the whole task's context, so per-step routing
degenerates to whole-task routing and the apprentice-agent win evaporates)?

The existing benchmarks are all single-shot Q&A (MMLU-Pro, TriviaQA), which
can't answer that. This corpus is ~50 hand-authored tasks that each require
several tool calls against the file/terminal tools. Every task runs in a fresh
sandbox directory (the file tools confine to the working dir), optionally
pre-seeded with ``setup_files``.

``difficulty`` is an *a-priori* label (my guess before running). A7 compares it
to the observed per-step tier: if easy tasks route mostly local and hard tasks
mostly cloud, difficulty is separable and the routing bet holds. If the split
is flat across labels, difficulty is entangled and the bet is weaker.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MultiStepTask:
    task_id: str
    prompt: str
    difficulty: str  # "easy" | "medium" | "hard" (a-priori guess)
    expected_min_steps: int  # rough lower bound on tool calls
    setup_files: dict[str, str] = field(default_factory=dict)
    tags: tuple[str, ...] = ()


# A small reusable Python module used as seed content for several tasks.
_SAMPLE_PY = '''\
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b
'''

_SAMPLE_CSV = "name,age,city\nalice,30,paris\nbob,25,berlin\ncara,35,rome\n"

_SAMPLE_LOG = "\n".join(
    f"2026-01-{d:02d} {'ERROR' if d % 4 == 0 else 'INFO'} event {d}"
    for d in range(1, 21)
) + "\n"

_README = "# Project\n\nA demo project.\n\n## Install\n\nRun setup.\n\n## Usage\n\nSee docs.\n"


TASKS: list[MultiStepTask] = [
    # ── easy: mostly mechanical, low reasoning per step ────────────────
    MultiStepTask(
        "e01", "List the files in the current directory, then read config.txt and tell me what port it sets.",
        "easy", 2, {"config.txt": "host=localhost\nport=8080\ndebug=false\n"}, ("read", "list"),
    ),
    MultiStepTask(
        "e02", "Create a file notes.txt containing the single line 'hello', then read it back to confirm.",
        "easy", 2, {}, ("write", "read"),
    ),
    MultiStepTask(
        "e03", "Read data.csv and tell me how many data rows (excluding the header) it has.",
        "easy", 1, {"data.csv": _SAMPLE_CSV}, ("read",),
    ),
    MultiStepTask(
        "e04", "List the directory, then report how many files end in .py.",
        "easy", 1, {"a.py": "x=1\n", "b.py": "y=2\n", "c.txt": "hi\n"}, ("list",),
    ),
    MultiStepTask(
        "e05", "Read README.md and list the section headings it contains.",
        "easy", 1, {"README.md": _README}, ("read",),
    ),
    MultiStepTask(
        "e06", "Create three empty files a.txt, b.txt, c.txt, then list the directory to confirm all three exist.",
        "easy", 4, {}, ("write", "list"),
    ),
    MultiStepTask(
        "e07", "Read app.log and count how many lines contain the word ERROR.",
        "easy", 1, {"app.log": _SAMPLE_LOG}, ("read", "terminal"),
    ),
    MultiStepTask(
        "e08", "Write the numbers 1 through 5, one per line, into nums.txt, then read it back.",
        "easy", 2, {}, ("write", "read"),
    ),
    MultiStepTask(
        "e09", "Read greeting.txt and create shout.txt containing its content in all uppercase.",
        "easy", 2, {"greeting.txt": "hello world\n"}, ("read", "write"),
    ),
    MultiStepTask(
        "e10", "List the directory and tell me which single file is the largest by name length.",
        "easy", 1, {"short.txt": "a", "a_much_longer_filename.txt": "b"}, ("list",),
    ),
    MultiStepTask(
        "e11", "Read settings.ini and report the value of the 'timeout' key.",
        "easy", 1, {"settings.ini": "[main]\ntimeout=30\nretries=3\n"}, ("read",),
    ),
    MultiStepTask(
        "e12", "Create dir listing: write the current file list into manifest.txt, one filename per line.",
        "easy", 2, {"one.txt": "1", "two.txt": "2"}, ("list", "write"),
    ),
    MultiStepTask(
        "e13", "Read colors.txt and create sorted.txt with the same lines sorted alphabetically.",
        "easy", 2, {"colors.txt": "red\ngreen\nblue\nyellow\n"}, ("read", "write"),
    ),
    MultiStepTask(
        "e14", "Read version.txt, then create a copy named version.bak with identical content.",
        "easy", 2, {"version.txt": "v1.2.3\n"}, ("read", "write"),
    ),
    MultiStepTask(
        "e15", "Count the total number of lines across all .txt files in the directory.",
        "easy", 2, {"x.txt": "a\nb\n", "y.txt": "c\nd\ne\n"}, ("terminal", "list"),
    ),

    # ── medium: at least one step needs real reasoning/synthesis ───────
    MultiStepTask(
        "m01", "Read calc.py and add a divide(a, b) function that guards against division by zero, then confirm the file now defines four functions.",
        "medium", 3, {"calc.py": _SAMPLE_PY}, ("read", "edit", "verify"),
    ),
    MultiStepTask(
        "m02", "Find every file that mentions 'TODO', then summarize what each TODO asks for.",
        "medium", 2,
        {"a.py": "# TODO: handle empty input\nx=1\n", "b.py": "y=2  # TODO: add type hints\n"},
        ("search", "read"),
    ),
    MultiStepTask(
        "m03", "Read data.csv, then create adults.csv containing only the rows where age >= 30 (keep the header).",
        "medium", 2, {"data.csv": _SAMPLE_CSV}, ("read", "write"),
    ),
    MultiStepTask(
        "m04", "Read calc.py and rename the 'multiply' function to 'times' everywhere, then verify no 'multiply' remains.",
        "medium", 3, {"calc.py": _SAMPLE_PY}, ("read", "edit", "search"),
    ),
    MultiStepTask(
        "m05", "Read app.log and write error_summary.txt listing only the ERROR lines with their dates.",
        "medium", 2, {"app.log": _SAMPLE_LOG}, ("read", "write"),
    ),
    MultiStepTask(
        "m06", "Read README.md and add a new '## License' section at the end stating 'MIT', then confirm it was added.",
        "medium", 3, {"README.md": _README}, ("read", "edit", "read"),
    ),
    MultiStepTask(
        "m07", "Search the directory for the function named 'subtract', open the file it's in, and describe what it does.",
        "medium", 2, {"calc.py": _SAMPLE_PY}, ("search", "read"),
    ),
    MultiStepTask(
        "m08", "Read config.json and create config.prod.json with the same keys but 'debug' set to false.",
        "medium", 2, {"config.json": '{"debug": true, "port": 8080, "host": "localhost"}\n'}, ("read", "write"),
    ),
    MultiStepTask(
        "m09", "Read numbers.txt (one integer per line) and write stats.txt reporting the min, max, and sum.",
        "medium", 2, {"numbers.txt": "12\n7\n33\n5\n19\n"}, ("read", "write"),
    ),
    MultiStepTask(
        "m10", "Find all .py files, then report which one has the most lines.",
        "medium", 2, {"a.py": "x=1\n", "b.py": _SAMPLE_PY, "c.py": "z=3\nw=4\n"}, ("search", "read"),
    ),
    MultiStepTask(
        "m11", "Read words.txt and create unique.txt with duplicate lines removed, order preserved.",
        "medium", 2, {"words.txt": "cat\ndog\ncat\nbird\ndog\ncat\n"}, ("read", "write"),
    ),
    MultiStepTask(
        "m12", "Read calc.py and add a module-level docstring describing the module, then confirm the docstring is present.",
        "medium", 3, {"calc.py": _SAMPLE_PY}, ("read", "edit", "read"),
    ),
    MultiStepTask(
        "m13", "Read inventory.csv and write low_stock.txt naming every item whose quantity is below 10.",
        "medium", 2, {"inventory.csv": "item,qty\napples,4\nbananas,20\ncherries,7\n"}, ("read", "write"),
    ),
    MultiStepTask(
        "m14", "Search for any file containing a hard-coded password, and report the file and the offending line.",
        "medium", 2, {"settings.py": 'PASSWORD = "hunter2"\nDEBUG = True\n', "ok.py": "X = 1\n"}, ("search", "read"),
    ),
    MultiStepTask(
        "m15", "Read poem.txt and create reversed.txt with the lines in reverse order.",
        "medium", 2, {"poem.txt": "line one\nline two\nline three\n"}, ("read", "write"),
    ),
    MultiStepTask(
        "m16", "Read calc.py, then create test_calc.py with one assert-based test per function.",
        "medium", 2, {"calc.py": _SAMPLE_PY}, ("read", "write"),
    ),
    MultiStepTask(
        "m17", "Find the file that defines 'add', add a comment above it explaining it returns the sum, and verify the comment is there.",
        "medium", 3, {"calc.py": _SAMPLE_PY}, ("search", "edit", "read"),
    ),
    MultiStepTask(
        "m18", "Read hosts.txt and create sorted_hosts.txt with the hostnames sorted and de-duplicated.",
        "medium", 2, {"hosts.txt": "web2\nweb1\ndb1\nweb1\ndb1\n"}, ("read", "write"),
    ),

    # ── hard: multi-hop, entangled context, or careful editing ─────────
    MultiStepTask(
        "h01", "Read calc.py, then refactor all three functions to use type hints (int arguments and return), and confirm the file still defines add, subtract, and multiply.",
        "hard", 4, {"calc.py": _SAMPLE_PY}, ("read", "edit", "search"),
    ),
    MultiStepTask(
        "h02", "There is a bug: divide.py raises on zero. Read it, fix it to return None on division by zero, and add a comment explaining the fix.",
        "hard", 3, {"divide.py": "def divide(a, b):\n    return a / b\n"}, ("read", "edit"),
    ),
    MultiStepTask(
        "h03", "Read the two modules a.py and b.py, then create combined.py that imports and re-exports every function they define.",
        "hard", 3, {"a.py": "def foo():\n    return 1\n", "b.py": "def bar():\n    return 2\n"}, ("read", "write"),
    ),
    MultiStepTask(
        "h04", "Read orders.csv, compute total revenue per city, and write revenue.txt with one 'city: total' line per city sorted by total descending.",
        "hard", 2,
        {"orders.csv": "city,amount\nparis,100\nberlin,50\nparis,75\nrome,200\nberlin,30\n"},
        ("read", "write"),
    ),
    MultiStepTask(
        "h05", "Read config.py, find every setting that is currently True, and write enabled.txt listing just those setting names.",
        "hard", 2, {"config.py": "A = True\nB = False\nC = True\nD = 0\nE = True\n"}, ("read", "write"),
    ),
    MultiStepTask(
        "h06", "The function 'multiply' in calc.py should be 'product' and every caller updated. Read calc.py and callers.py, rename consistently across both, then verify no 'multiply' remains anywhere.",
        "hard", 4,
        {"calc.py": _SAMPLE_PY, "callers.py": "from calc import multiply\nprint(multiply(2, 3))\n"},
        ("read", "edit", "search"),
    ),
    MultiStepTask(
        "h07", "Read matrix.txt (rows of space-separated integers) and write transposed.txt with the matrix transposed.",
        "hard", 2, {"matrix.txt": "1 2 3\n4 5 6\n"}, ("read", "write"),
    ),
    MultiStepTask(
        "h08", "Read events.log, then write report.txt stating how many events occurred per severity level (INFO vs ERROR).",
        "hard", 2, {"events.log": _SAMPLE_LOG}, ("read", "write"),
    ),
    MultiStepTask(
        "h09", "Read schema.sql and generate model.py with a Python dataclass matching the table's columns and types.",
        "hard", 2,
        {"schema.sql": "CREATE TABLE users (\n  id INTEGER,\n  name TEXT,\n  active BOOLEAN\n);\n"},
        ("read", "write"),
    ),
    MultiStepTask(
        "h10", "Read prices.csv, apply a 10% discount to every price, and write discounted.csv preserving the header and rounding to two decimals.",
        "hard", 2, {"prices.csv": "product,price\npen,1.50\nbook,12.00\nlamp,45.99\n"}, ("read", "write"),
    ),
    MultiStepTask(
        "h11", "Read the recursive factorial in math_utils.py, rewrite it iteratively, and confirm it still defines a 'factorial' function.",
        "hard", 3,
        {"math_utils.py": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n"},
        ("read", "edit", "search"),
    ),
    MultiStepTask(
        "h12", "Read deps.txt (a package per line) and requirements.in, then write missing.txt listing packages in deps.txt that are absent from requirements.in.",
        "hard", 2,
        {"deps.txt": "requests\nnumpy\nflask\n", "requirements.in": "numpy\nflask\npytest\n"},
        ("read", "write"),
    ),
    MultiStepTask(
        "h13", "Read story.txt and write summary.txt containing a one-sentence summary of each paragraph (paragraphs are separated by blank lines).",
        "hard", 2,
        {"story.txt": "The sun rose over the hills. Birds began to sing.\n\nBy noon the market was busy. Vendors called out prices.\n\nAt night the town fell quiet. Only the river could be heard.\n"},
        ("read", "write"),
    ),
    MultiStepTask(
        "h14", "Read grades.csv, compute each student's average across their subjects, and write averages.txt with 'student: average' sorted by average descending.",
        "hard", 2,
        {"grades.csv": "student,subject,score\namy,math,90\namy,science,80\nben,math,70\nben,science,60\n"},
        ("read", "write"),
    ),
    MultiStepTask(
        "h15", "Read handler.py, wrap the body of the 'process' function in a try/except that logs and re-raises, and confirm 'try' and 'except' now appear in the file.",
        "hard", 3,
        {"handler.py": "def process(item):\n    result = item * 2\n    return result\n"},
        ("read", "edit", "search"),
    ),
    MultiStepTask(
        "h16", "Read urls.txt and write domains.txt containing just the unique domain of each URL, sorted.",
        "hard", 2,
        {"urls.txt": "https://example.com/a\nhttp://test.org/x\nhttps://example.com/b\n"},
        ("read", "write"),
    ),
    MultiStepTask(
        "h17", "Read config.yaml, then produce config.env with the same key/values flattened into KEY=value lines (uppercase keys).",
        "hard", 2,
        {"config.yaml": "host: localhost\nport: 5432\nname: mydb\n"},
        ("read", "write"),
    ),
]


# ── Hardening: compute tasks a 7B model plausibly gets wrong ────────
#
# The pilot's "100% local, 0% cloud-intent" was suspicious: my first-cut tasks
# were too easy. These demand multi-step arithmetic / careful transformation
# where a small model is more likely to be confidently wrong — the case that
# justifies routing. Verified deterministically below.
_HARD_COMPUTE_TASKS: list[MultiStepTask] = [
    MultiStepTask(
        "h18", "Read amounts.txt (one integer per line) and write cumsum.txt with the running cumulative sum, one value per line.",
        "hard", 2, {"amounts.txt": "12\n7\n33\n5\n19\n"}, ("read", "write", "compute"),
    ),
    MultiStepTask(
        "h19", "Read grid.txt (rows of space-separated integers) and write rowsums.txt with the sum of each row, one per line.",
        "hard", 2, {"grid.txt": "1 2 3\n4 5 6\n"}, ("read", "write", "compute"),
    ),
    MultiStepTask(
        "h20", "Read scores.csv (name,score). Compute the mean score, then report which single person scored furthest above the mean.",
        "hard", 2, {"scores.csv": "name,score\nalice,50\nbob,55\ncara,95\ndan,60\n"}, ("read", "compute"),
    ),

    # ── Independently-hard: multi-hop reasoning + error-prone transforms ──
    # These target the concern that the first-cut corpus was too easy for a 7B
    # model. Each needs several dependent inference steps where a small model
    # tends to slip (off-by-one, wrong aggregation key, missed edge case), so
    # the correctness-vs-confidence signal has room to show miscalibration.
    MultiStepTask(
        "x01", "Read log.txt. Each line is 'YYYY-MM-DD LEVEL msg'. Write daily_errors.txt with 'DATE count' for each date that has at least one ERROR, sorted by date, counting only ERROR lines.",
        "hard", 2,
        {"log.txt": "2026-01-01 INFO a\n2026-01-01 ERROR b\n2026-01-02 ERROR c\n2026-01-02 ERROR d\n2026-01-02 INFO e\n2026-01-03 INFO f\n"},
        ("read", "write", "compute"),
    ),
    MultiStepTask(
        "x02", "Read transactions.csv (id,type,amount). Compute net balance = sum of credits minus sum of debits, and write balance.txt with just the number.",
        "hard", 2,
        {"transactions.csv": "id,type,amount\n1,credit,100\n2,debit,30\n3,credit,50\n4,debit,20\n"},
        ("read", "write", "compute"),
    ),
    MultiStepTask(
        "x03", "Read sequence.txt (one integer per line). Write fib_like.txt containing only the numbers that equal the sum of the two preceding numbers in the list (skip the first two).",
        "hard", 2,
        {"sequence.txt": "2\n3\n5\n8\n10\n18\n"},  # 5=2+3✓ 8=3+5✓ 10≠5+8 18=8+10✓
        ("read", "write", "compute"),
    ),
    MultiStepTask(
        "x04", "Read employees.csv (name,dept,salary). Write dept_avg.txt with 'dept: avg_salary' for each department, averages rounded to the nearest integer, sorted by department name.",
        "hard", 2,
        {"employees.csv": "name,dept,salary\na,eng,100\nb,eng,140\nc,sales,80\nd,sales,100\ne,sales,90\n"},
        ("read", "write", "compute"),
    ),
    MultiStepTask(
        "x05", "Read inventory.csv (sku,qty,reorder). Write reorder.txt listing the skus where qty is strictly less than reorder, sorted by how far below reorder they are (largest shortfall first).",
        "hard", 2,
        {"inventory.csv": "sku,qty,reorder\naaa,2,10\nbbb,9,10\nccc,20,5\nddd,0,4\n"},  # shortfalls: aaa 8, ddd 4, bbb 1
        ("read", "write", "compute"),
    ),
    MultiStepTask(
        "x06", "Read votes.txt (one candidate name per line). Determine the winner by plurality; if there is a tie for first, write 'TIE' to winner.txt, otherwise write the winning name.",
        "hard", 2,
        {"votes.txt": "amy\nbob\namy\ncara\nbob\namy\n"},  # amy 3, bob 2, cara 1 → amy
        ("read", "write", "compute"),
    ),
]


# ── Correctness verifiers ──────────────────────────────────────────
#
# A7's pilot showed the local model *completes* every task, but "completed"
# only means the loop returned an answer — not a correct one. High self-
# confidence (avg_logprob) on a wrong step is exactly the failure routing
# exists to catch. These verifiers check the *actual* sandbox state / answer
# after a run, so we can measure correctness-vs-confidence. Each takes
# (sandbox_dir, final_answer) and returns True iff the task was done right.
# Only deterministic tasks have verifiers; the rest are excluded from the
# correctness rate (reported as "unchecked").

from pathlib import Path as _Path
from typing import Callable, Optional


def _lines(p: _Path) -> list[str]:
    if not p.exists():
        return []
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]


def _has_all(answer: str, *needles: str) -> bool:
    return all(n.lower() in answer.lower() for n in needles)


VERIFIERS: dict[str, Callable[[_Path, str], bool]] = {
    # Answer-based (report tasks): the fact must appear in the final answer.
    "e01": lambda d, a: _has_all(a, "8080"),
    "e03": lambda d, a: "3" in a,
    "e07": lambda d, a: "5" in a,           # 5 ERROR lines (d%4==0 in 1..20)
    "e11": lambda d, a: "30" in a,
    # File-state (write tasks): the produced file must have the right content.
    "e09": lambda d, a: "HELLO WORLD" in (d / "shout.txt").read_text().upper()
                        if (d / "shout.txt").exists() else False,
    "e13": lambda d, a: _lines(d / "sorted.txt") == ["blue", "green", "red", "yellow"],
    "m03": lambda d, a: (
        (d / "adults.csv").exists()
        and "alice" in (t := (d / "adults.csv").read_text().lower())
        and "cara" in t and "bob" not in t
    ),
    "m09": lambda d, a: (
        (d / "stats.txt").exists()
        and _has_all((d / "stats.txt").read_text(), "5", "33", "76")  # min max sum
    ),
    "m11": lambda d, a: _lines(d / "unique.txt") == ["cat", "dog", "bird"],
    "m13": lambda d, a: (
        (d / "low_stock.txt").exists()
        and "apples" in (t := (d / "low_stock.txt").read_text().lower())
        and "cherries" in t and "bananas" not in t
    ),
    "m18": lambda d, a: _lines(d / "sorted_hosts.txt") == ["db1", "web1", "web2"],
    # Compute-heavy (the ones a 7B model plausibly gets wrong):
    "h04": lambda d, a: _rev_ok(d / "revenue.txt"),
    "h07": lambda d, a: _lines(d / "transposed.txt") == ["1 4", "2 5", "3 6"],
    "h10": lambda d, a: (
        (d / "discounted.csv").exists()
        and _has_all((d / "discounted.csv").read_text(), "1.35", "10.80", "41.39")
    ),
    "h12": lambda d, a: (
        (d / "missing.txt").exists()
        and "requests" in (t := (d / "missing.txt").read_text().lower())
        and "numpy" not in t and "flask" not in t
    ),
    "h14": lambda d, a: (
        (d / "averages.txt").exists()
        and (t := (d / "averages.txt").read_text().lower()).index("amy") < t.index("ben")
        and _has_all(t, "85", "65")
    ),
    "h16": lambda d, a: _lines(d / "domains.txt") == ["example.com", "test.org"],
    "h18": lambda d, a: _lines(d / "cumsum.txt") == ["12", "19", "52", "57", "76"],
    "h19": lambda d, a: _lines(d / "rowsums.txt") == ["6", "15"],
    "h20": lambda d, a: _has_all(a, "cara"),
    # Broadened coverage on existing tasks (were previously unchecked).
    "e04": lambda d, a: "2" in a,                       # 2 .py files
    "e05": lambda d, a: _has_all(a, "install", "usage"),  # README headings
    "m05": lambda d, a: (
        (d / "error_summary.txt").exists()
        and "error" in (t := (d / "error_summary.txt").read_text().lower())
        and "info" not in t  # only ERROR lines
    ),
    "m15": lambda d, a: _lines(d / "reversed.txt") == ["line three", "line two", "line one"],
    "h05": lambda d, a: (
        (d / "enabled.txt").exists()
        and set(_lines(d / "enabled.txt")) == {"A", "C", "E"}  # the True settings
    ),
    # Independently-hard hardening tasks.
    "x01": lambda d, a: _lines(d / "daily_errors.txt") == ["2026-01-01 1", "2026-01-02 2"],
    "x02": lambda d, a: (d / "balance.txt").exists()
                        and (d / "balance.txt").read_text().strip().lstrip("+") == "100",
    "x03": lambda d, a: _lines(d / "fib_like.txt") == ["5", "8", "18"],
    "x04": lambda d, a: (
        (d / "dept_avg.txt").exists()
        and (t := (d / "dept_avg.txt").read_text().lower()).index("eng") < t.index("sales")
        and _has_all(t, "120", "90")  # eng (100,140)->120, sales (80,100,90)->90
    ),
    "x05": lambda d, a: _first_tokens(d / "reorder.txt", 3) == ["aaa", "ddd", "bbb"],
    "x06": lambda d, a: (d / "winner.txt").exists()
                        and (d / "winner.txt").read_text().strip().lower() == "amy",
}


def _first_tokens(p: _Path, n: int) -> list[str]:
    """First whitespace/comma token of each of the first n non-empty lines,
    lowercased — tolerant of 'sku' vs 'sku: shortfall' output formats."""
    import re
    out = []
    for ln in _lines(p)[:n]:
        tok = re.split(r"[\s,:]+", ln.strip())[0]
        out.append(tok.lower())
    return out


def _rev_ok(p: _Path) -> bool:
    """revenue.txt must list rome(200) > paris(175) > berlin(80) in that order.

    Output format varies (colons, $, spacing), so match on order of city names
    and presence of each correct total rather than exact strings.
    """
    if not p.exists():
        return False
    text = p.read_text().lower()
    try:
        order_ok = text.index("rome") < text.index("paris") < text.index("berlin")
    except ValueError:
        return False
    return order_ok and _has_all(text, "200", "175", "80")


def verifier_for(task_id: str) -> Optional[Callable[[_Path, str], bool]]:
    return VERIFIERS.get(task_id)


def load_tasks() -> list[MultiStepTask]:
    """The full corpus (50 original + hardening compute tasks)."""
    return list(TASKS) + _HARD_COMPUTE_TASKS


def pilot_tasks(n: int = 10, seed: int = 42) -> list[MultiStepTask]:
    """A stratified pilot: sample evenly across difficulty for a cheap dry run."""
    import random

    by_diff: dict[str, list[MultiStepTask]] = {"easy": [], "medium": [], "hard": []}
    for t in TASKS:
        by_diff[t.difficulty].append(t)

    rng = random.Random(seed)
    for bucket in by_diff.values():
        rng.shuffle(bucket)

    picked: list[MultiStepTask] = []
    order = ["easy", "medium", "hard"]
    i = 0
    while len(picked) < n:
        made_progress = False
        for diff in order:
            if i < len(by_diff[diff]):
                picked.append(by_diff[diff][i])
                made_progress = True
                if len(picked) >= n:
                    break
        if not made_progress:
            break
        i += 1
    return picked
