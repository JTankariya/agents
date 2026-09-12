# Agentic Data Analyst Playground

A GitHub-ready Streamlit monorepo demonstrating a Groq-powered data analyst that:

- accepts CSV uploads up to 5 MB;
- reads no more than 50,001 rows and retains at most 50,000;
- sends the user's written request, schema metadata, and a redacted structural
  sample to Groq, never raw CSV rows automatically;
- uses the configurable `openai/gpt-oss-20b` default through the official `groq`
  Python SDK;
- rejects out-of-scope requests with `GUARDRAIL_TRIGGERED`;
- generates transparent pandas code and attempts one self-healing correction;
- executes code with native `exec()` inside a disposable, AST-restricted process;
- renders text, DataFrames, Matplotlib, and Plotly outputs;
- applies a three-run browser-session cap plus a shared in-process request limit;
- supports star feedback and a Formspree contact form; and
- is structured for additional agent apps and shared packages.

## Architecture at a glance

```text
Browser CSV
    │
    ▼
CSV loader ── 5 MB / 50,000-row / memory / schema guards
    │
    ├──► Local preview and local DataFrame
    │
    └──► Redacted profile + user request ──► Groq structured JSON plan
                                                │
                                                ▼
                                     AST policy validation
                                                │
                                                ▼
                                  Disposable child-process exec()
                                                │
                              ┌─────────────────┴──────────────────┐
                              ▼                                    ▼
                    Safe result/chart                      One repair call
                              │                           after first failure
                              ▼
                         Streamlit UI
```

## Repository structure

```text
.
├── streamlit_app.py                 # Community Cloud entrypoint
├── apps/
│   └── data_analyst/app.py          # Streamlit UI
├── packages/
│   └── agent_core/                  # Agent, Groq, privacy, sandbox, rendering
├── tests/                           # Unit, orchestration, integration, security
├── sample_data/sales_demo.csv
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DEPLOYMENT.md
│   ├── SECURITY.md
│   ├── BRD_DECISIONS.md
│   └── VALIDATION.md
├── .streamlit/config.toml           # 5 MB upload limit and safe error display
└── .streamlit/secrets.toml.example  # Placeholder values only
```

## Local quick start

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit .streamlit/secrets.toml and add your real Groq key.
streamlit run streamlit_app.py
```

Use `sample_data/sales_demo.csv` for the first run.

## Required Streamlit secrets

```toml
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
GROQ_MODEL = "openai/gpt-oss-20b"
FORMSPREE_FORM_ID = "optional_form_id"
ENABLE_LEGACY_ANALYTICS = false
MAX_RUNS_PER_SESSION = 3
GLOBAL_LLM_REQUESTS_PER_MINUTE = 20
```

`FORMSPREE_FORM_ID` is optional. The app remains usable without it.

The model is configurable because provider availability and quotas change. The BRD
example `llama-3.3-70b-versatile` should only be selected when the target Groq
project currently exposes it. Recheck the project's Models and Limits pages before
public deployment.

## Privacy behavior

The BRD’s requested `df.head().to_markdown()` contains original cell values and
conflicts with “zero raw data exposure.” This build sends a five-row Markdown table
of dtype placeholders instead. The user's natural-language request is still sent
to Groq, so the UI warns users not to paste sensitive values into it. See
[docs/BRD_DECISIONS.md](docs/BRD_DECISIONS.md).

## Deployment

Follow [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for account setup, local testing,
GitHub push checks, Streamlit Community Cloud deployment, Formspree configuration,
and acceptance testing.

## Security boundary

This is a strong portfolio demonstration, not a formal hostile-code execution
service. The sandbox combines AST restrictions, a small namespace, deep-copied
data, a child process, timeout termination, optional Unix resource limits, and
output-size checks. See [docs/SECURITY.md](docs/SECURITY.md).

## Test and lint

```bash
pytest
ruff check .
```

The packaged build passed 40 automated tests. See
[docs/VALIDATION.md](docs/VALIDATION.md) for the acceptance-criteria matrix and the
live checks that still require deployment credentials.

## License

MIT
