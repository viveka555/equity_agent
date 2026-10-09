# Institutional Equity Research Agent

A laptop-local, Python-based research assistant built with LangGraph. It
routes requests to specialist workflows for company news, annual-report
question answering, financial ratios, DCF valuation, risk analysis, and
evidence-grounded institutional research reports. Conversation history is
stored locally in SQLite.

## Project status

| Phase | Area | Status |
| --- | --- | --- |
| 1 | LangGraph foundation and planner | Complete |
| 2 | Groq and company news | Complete |
| 3 | Annual-report RAG | Complete |
| 4 | Financial ratio analysis | Complete |
| 5 | DCF valuation | Complete |
| 6 | Evidence-based risk analysis | Complete |
| 7 | Laptop-local conversation memory | Complete |
| 8 | Institutional research report generator | In progress |

The report generator synthesizes evidence already present in graph state. It
does not fetch financial statements or run every specialist automatically:
the project currently has no external financial-data provider. The interactive
app can load a local JSON file containing evidence or specialist inputs.
When evidence is missing, the report identifies the gaps instead of
fabricating analysis.

## Requirements

- Windows 10/11, macOS, or Linux
- Python 3.11 or newer
- A Groq API key for LLM-backed planner, RAG, risk, and report workflows
- Internet access for Groq requests, company news search, and first-time
  downloads of the local embedding model
- An annual-report PDF in `rag/annual_reports/` for annual-report RAG

The application and conversation database run on the laptop. Chroma data,
SQLite history, and downloaded model files are local. Groq inference and web
search still require network access.

## Setup on Windows

Open PowerShell in the project folder:

```powershell
cd C:\Users\USER\Desktop\equity_agent
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Edit `.env` and set your own `GROQ_API_KEY`. Do not commit `.env` or share the
key. If PowerShell blocks virtual-environment activation, run the project
using `.\.venv\Scripts\python.exe` directly or adjust the execution policy
for your user according to your normal Windows setup.

On macOS or Linux, create and activate a virtual environment with `python3`
and install the same `requirements.txt`.

## Run the app

```powershell
python app.py
```

Choose a conversation ID to resume local history, or press Enter to use the
`local` conversation. Then optionally enter the path to a research-input JSON
file (for example, `data/research_inputs.json`) or press Enter to skip it.
Type `exit` or `quit` to close the app.

## Research workflows

The planner recognizes `news`, `rag`, `ratio`, `dcf`, `risk`, and `report`.
Ratio and DCF calculations expect structured financial inputs in graph state;
the current interactive app does not connect to a market-data API. Annual
report questions require a PDF to be ingested into the local RAG store.

For a full report, the report node accepts these evidence keys in graph state:

- `news`: supplied news results
- `annual_report_analysis`: annual-report question and answer
- `financial_data`: normalized financial statement values
- `ratio_analysis`: calculated financial ratios
- `dcf_analysis`: DCF assumptions and valuation outputs
- `risk_analysis`: structured risk findings

The local app reads JSON keys for those evidence sources, and also accepts
`dcf_forecast_input`, `dcf_assumptions`, `dcf_wacc_inputs`, and
`dcf_sensitivity_inputs` for DCF requests. See
`data/research_inputs.example.json` for the accepted top-level structure.
Copy it to `data/research_inputs.json` and replace the empty values with
company-specific research. The real input file is ignored by Git.

Every generated finding should cite one of the supplied source keys. If no
evidence is available, the report returns an evidence-gap checklist without
making an LLM request. This keeps the result honest, but a complete report
requires those source outputs to be collected first.

## Local files

- `data/conversations.sqlite3`: local conversation history (ignored by Git)
- `vector_db/`: local Chroma database (ignored by Git)
- `rag/annual_reports/`: PDFs supplied for annual-report research
- `data/research_inputs.json`: optional local research inputs (ignored by Git)
- `.env`: local API credentials (ignored by Git)

Back up any local conversation database or vector store you want to preserve;
they are not part of the Git repository.

## Tests

Run focused report tests and then the full suite:

```powershell
python -m pytest tests/test_report_agent.py tests/test_graph.py -v
python -m pytest -v
```

Tests should not require live Groq calls; the optional connectivity test is
skipped unless explicitly enabled.
