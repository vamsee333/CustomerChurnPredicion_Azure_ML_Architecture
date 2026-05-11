Totally valid — YAML syntax is unforgiving (indentation errors break everything silently) and there's a lot of keywords to learn. But the good news is you don't need to memorise it. You need to understand the **structure**, and then you can read and write it confidently.

Let me break it down from scratch using your actual `ci.yml`.

---

## The mental model — a yml file is just a job description

Think of it like writing instructions for a brand new intern who has a fresh laptop and knows nothing about your project. You tell them:

- **When** to start working
- **What machine** to use
- **What steps** to follow in order

That's all a yml file is.

---

## The 4 building blocks — everything fits into these

```yaml
name:     # what to call this workflow (shows in GitHub Actions tab)
on:       # WHEN to run it
jobs:     # WHAT to do — contains one or more jobs
  job_name:
    runs-on:   # which machine to use
    steps:     # the ordered list of commands
```

Every single CI/CD yml file in the world, at any company, is just these four things arranged differently.

---

## Now read your ci.yml line by line

Let me walk through it as plain English:

```yaml
name: CI — Unit Tests & Coverage
```
"Call this workflow CI — Unit Tests & Coverage. This name appears in the GitHub Actions tab."

```yaml
on:
  pull_request:
    branches: [main]
    paths:
      - "src/**.py"
      - "tests/**.py"
```
"Run this workflow WHEN someone opens or updates a Pull Request targeting the main branch, BUT ONLY IF the changed files are inside src/ or tests/. If someone only edits README.md, don't bother running."

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
```
"There is one job called `test`. Run it on a fresh Ubuntu Linux machine (GitHub provides this for free)."

```yaml
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
```
"Step 1 — Download the repo code onto this machine. `uses:` means use a pre-built action someone else wrote. `actions/checkout@v4` is GitHub's official one — everyone uses it."

```yaml
      - name: Set up Python 3.10
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"
          cache: pip
```
"Step 2 — Install Python 3.10 on this machine. `with:` passes settings into the action. `cache: pip` means remember downloaded packages between runs so it's faster next time."

```yaml
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install scikit-learn==1.5.0 pandas==2.2.2 ...
```
"`run:` means execute a shell command directly — no pre-built action, just your own commands. The `|` means multiple lines follow. This is exactly what you'd type in your terminal."

```yaml
      - name: Run pytest with coverage
        env:
          PYTHONPATH: src
        run: |
          pytest tests/test_pipeline.py --cov=src --cov-fail-under=80 -v
```
"`env:` sets environment variables for this step only. `PYTHONPATH: src` is the Windows fix we talked about — tells Python where src/ is. Then runs pytest."

---

## The 5 keywords you actually need to know

That's it. Everything else is just variations of these:

| Keyword | What it means | When you write it |
|---|---|---|
| `uses:` | Use a pre-built action from GitHub marketplace | Checkout, setup-python, login to Azure |
| `run:` | Run a shell command directly | pip install, pytest, python script.py |
| `with:` | Pass settings into a `uses:` action | python-version, cache, credentials |
| `env:` | Set environment variables | PYTHONPATH, secrets |
| `needs:` | This job must wait for another job to finish first | CD job waiting for CI job |

---

## The indentation rule — the one thing that trips everyone up

YAML uses spaces (never tabs) to show hierarchy. The rule is simple: **things at the same level must have the same indentation.**

```yaml
jobs:           # level 0
  test:         # level 1 — belongs to jobs
    runs-on:    # level 2 — belongs to test
    steps:      # level 2 — belongs to test
      - name:   # level 3 — belongs to steps (- means list item)
        uses:   # level 3 — belongs to this step
        with:   # level 3 — belongs to this step
          python-version:  # level 4 — belongs to with
```

The most common error is mixing 2-space and 4-space indentation in the same file. Pick one and stick to it.

---

## How to write one from scratch — the template

When you need a new workflow, start from this skeleton and fill in the blanks:

```yaml
name: <what this does>

on:
  push:              # or pull_request
    branches: [main]

jobs:
  <job-name>:
    runs-on: ubuntu-latest
    steps:

      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Install dependencies
        run: pip install <your packages>

      - name: <your actual task>
        run: <your command>
```

Every workflow you'll ever write for this project is this skeleton with different steps plugged in.

---

## Where to find pre-built actions (so you don't write from scratch)

The two places professionals actually look things up:

**GitHub Actions Marketplace** — `github.com/marketplace?type=actions`
Search "azure login", "setup python", "slack notify" — thousands of ready-made actions with copy-paste examples.

**Your own ci.yml as a reference** — once you understand what each block does, you reuse your own file as a template. Most engineers do this rather than memorising syntax.

---

## The fastest way to get comfortable

Take your `ci.yml`, read each block and try to say it out loud in plain English. If you can do that for every line, you can write one. The syntax is just the translation layer between English and YAML — and you now know enough of that translation to work with it confidently.