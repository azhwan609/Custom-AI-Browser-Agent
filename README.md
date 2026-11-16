# AI Browser Agent

A production-oriented FastAPI browser agent for safe, read-only exploration of public websites. Users submit a natural-language task in the single-page chat UI. Groq produces a strict JSON action plan, Pydantic validates it, and Playwright executes only the allowlisted actions.

## Safety boundaries

The agent uses a fresh anonymous Chromium context for every request. It accepts only public HTTP(S) destinations, rejects private or reserved network addresses and unsafe ports, blocks downloads and service workers, and re-checks browser subrequests. The only supported actions are public navigation, ordinary link clicks, non-sensitive text/search/URL field fills without submission, and limited extraction. It does not authenticate, access private systems, reveal secrets, submit forms, make payments, purchase or book items, delete data, send messages, change settings, download files, upload files, or run model-generated code.

Technical telemetry remains server-side. The frontend renders only the final answer or a friendly error.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY="your-key"
export GROQ_MODEL="openai/gpt-oss-20b"
export PLAYWRIGHT_BROWSERS_PATH=0
playwright install chromium
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000`. Example API request:

```bash
curl -X POST http://127.0.0.1:8000/execute \
  -H 'content-type: application/json' \
  -d '{"task":"Go to https://example.com and tell me the page title."}'
```

## Render deployment

The included `render.yaml` creates a free Python web service from the `main` branch, installs Chromium during the build, uses the package-local browser path, and starts FastAPI with Uvicorn. Configure `GROQ_API_KEY` as a Render secret. `GROQ_MODEL` defaults to `openai/gpt-oss-20b`; do not use retired models such as `llama3-8b-8192`. `PLAYWRIGHT_BROWSERS_PATH=0` is configured in the Blueprint for both build and runtime.

## Customization

Change the theme variables at the top of `index.html` to update the app name, colors, typography, background, corner radius, and other visual details. Interface copy is located in the same file. Backend limits, timeouts, and safety rules are centralized near the top of `main.py`.

## Operational limits

The service intentionally runs one browser task at a time, with a maximum of six planned actions and conservative planning/navigation/action timeouts. It is designed for safe public browsing, not high-volume automation. Higher traffic should add authentication, rate limiting, queueing, monitoring, and a paid hosting plan after a separate review.
