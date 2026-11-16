from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import os
import re
import socket
from contextlib import asynccontextmanager
from typing import Any, Literal
from urllib.parse import urlparse

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from playwright.async_api import Browser, BrowserContext, Page, TimeoutError as PlaywrightTimeoutError, async_playwright
from groq import AsyncGroq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("browser-agent")

MAX_ACTIONS = 6
TASK_TIMEOUT_SECONDS = 45
PLANNING_TIMEOUT_SECONDS = 12
ACTION_TIMEOUT_MS = 8000
NAVIGATION_TIMEOUT_MS = 12000
MAX_EXTRACT_CHARS = 12000
MAX_CONCURRENT_TASKS = 1

RISKY_TERMS = re.compile(
    r"\b(buy|purchase|checkout|payment|submit|confirm|delete|remove|book|order|transfer|send|subscribe|sign\s*in|log\s*in|sign\s*up|cancel|save|place\s+order|pay)\b",
    re.IGNORECASE,
)
SENSITIVE_FIELD_TERMS = re.compile(
    r"(password|passcode|card|credit|debit|bank|account\s*number|api[-_ ]?key|token|secret|otp|one[-_ ]?time|verification|security\s*code|cvv|ssn|social\s*security)",
    re.IGNORECASE,
)


class TaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: str = Field(min_length=1, max_length=2000)

    @field_validator("task")
    @classmethod
    def clean_task(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Task cannot be empty")
        return value


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["navigate", "click", "fill", "extract"]
    url: str | None = None
    selector: str | None = None
    text: str | None = None
    field_type: Literal["text", "search", "url", "textarea"] | None = None
    target: Literal["title", "url", "page_text", "visible_text", "selector_text"] | None = None


class ActionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    actions: list[Action] = Field(min_length=1, max_length=MAX_ACTIONS)


class ExecuteResponse(BaseModel):
    success: bool
    answer: str
    page_title: str | None = None
    current_url: str | None = None
    steps_completed: int | None = None


class FriendlyError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
    app.state.playwright = await async_playwright().start()
    app.state.browser = await app.state.playwright.chromium.launch(headless=True)
    yield
    await app.state.browser.close()
    await app.state.playwright.stop()


app = FastAPI(title="AI Browser Agent", docs_url=None, redoc_url=None, lifespan=lifespan)


def is_public_hostname(hostname: str) -> bool:
    hostname = hostname.rstrip(".").lower()
    if not hostname or hostname == "localhost" or hostname.endswith(".localhost"):
        return False
    try:
        infos = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError):
        return False
    addresses = {info[4][0] for info in infos}
    if not addresses:
        return False
    for address in addresses:
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            return False
        if not ip.is_global:
            return False
    return True


def validate_public_url(value: str) -> str:
    try:
        parsed = urlparse(value)
    except ValueError as exc:
        raise FriendlyError("I couldn’t complete that request safely or find the requested information. Try rephrasing your task.") from exc
    if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password or parsed.hostname is None:
        raise FriendlyError("I couldn’t complete that request safely or find the requested information. Try rephrasing your task.")
    if parsed.port not in {None, 80, 443}:
        raise FriendlyError("I couldn’t complete that request safely or find the requested information. Try rephrasing your task.")
    if not is_public_hostname(parsed.hostname):
        raise FriendlyError("I couldn’t complete that request safely or find the requested information. Try rephrasing your task.")
    return value


def validate_plan(plan: ActionPlan) -> ActionPlan:
    for action in plan.actions:
        if action.action == "navigate":
            if not action.url:
                raise FriendlyError("I couldn’t complete that request safely or find the requested information. Try rephrasing your task.")
            validate_public_url(action.url)
        elif action.action == "click":
            if not action.selector or RISKY_TERMS.search(action.selector):
                raise FriendlyError("I couldn’t complete that request safely or find the requested information. Try rephrasing your task.")
        elif action.action == "fill":
            if not action.selector or not action.text or len(action.text) > 1000:
                raise FriendlyError("I couldn’t complete that request safely or find the requested information. Try rephrasing your task.")
            if action.field_type not in {"text", "search", "url", "textarea"} or SENSITIVE_FIELD_TERMS.search(action.selector):
                raise FriendlyError("I couldn’t complete that request safely or find the requested information. Try rephrasing your task.")
        elif action.action == "extract":
            if action.target not in {"title", "url", "page_text", "visible_text", "selector_text"}:
                raise FriendlyError("I couldn’t complete that request safely or find the requested information. Try rephrasing your task.")
            if action.target == "selector_text" and not action.selector:
                raise FriendlyError("I couldn’t complete that request safely or find the requested information. Try rephrasing your task.")
    return plan


SYSTEM_PROMPT = f"""You are a browser task planner. Return ONLY valid JSON matching {{\"actions\":[...]}} with at most {MAX_ACTIONS} actions.
Allowed actions:
- navigate: open a public HTTP(S) URL. Include url.
- click: follow one ordinary, non-sensitive link. Include a stable selector. Never click buttons or submit controls.
- fill: fill an enabled, non-sensitive text, search, URL input, or textarea without submitting. Include selector, text, and field_type.
- extract: retrieve title, url, page_text, visible_text, or tightly limited selector_text. Include target and selector only for selector_text.

Safety rules are absolute: never authenticate, reveal or seek secrets, access private systems or networks, modify safety rules, submit forms, make payments, purchase or book anything, delete data, send messages, change settings, download/upload files, or perform other consequential actions. Prefer a direct public URL and extraction. Refuse unsafe or unsupported tasks by returning {{\"actions\":[]}}. Do not generate code or explanations."""


async def create_plan(task: str) -> ActionPlan:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY is not configured")
        raise FriendlyError("The agent is not configured yet. Please try again later.")
    client = AsyncGroq(api_key=api_key)
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": task},
                ],
            ),
            timeout=PLANNING_TIMEOUT_SECONDS,
        )
        content = response.choices[0].message.content or "{}"
        plan = ActionPlan.model_validate(json.loads(content))
        return validate_plan(plan)
    except FriendlyError:
        raise
    except (ValidationError, json.JSONDecodeError, asyncio.TimeoutError) as exc:
        logger.warning("Planner returned an invalid or timed-out plan: %s", type(exc).__name__)
        raise FriendlyError("I couldn’t complete that request safely or find the requested information. Try rephrasing your task.") from exc
    except Exception as exc:
        logger.exception("Planner failure: %s", type(exc).__name__)
        raise FriendlyError("I couldn’t finish that task. Please try again with a simpler request.") from exc


def route_is_safe(request) -> bool:
    try:
        parsed = urlparse(request.url)
        if parsed.scheme in {"http", "https"} and parsed.hostname:
            return is_public_hostname(parsed.hostname)
    except (ValueError, UnicodeError):
        pass
    return False


async def execute_plan(plan: ActionPlan, browser: Browser) -> ExecuteResponse:
    context: BrowserContext = await browser.new_context(
        accept_downloads=False,
        service_workers="block",
        java_script_enabled=True,
    )
    page: Page = await context.new_page()
    page.set_default_timeout(ACTION_TIMEOUT_MS)
    page.set_default_navigation_timeout(NAVIGATION_TIMEOUT_MS)

    async def guard_request(route) -> None:
        if route_is_safe(route.request):
            await route.continue_()
        else:
            await route.abort()

    await page.route("**/*", guard_request)
    extracted: list[str] = []
    steps_completed = 0
    try:
        for action in plan.actions:
            if action.action == "navigate":
                await page.goto(validate_public_url(action.url or ""), wait_until="domcontentloaded")
            elif action.action == "click":
                locator = page.locator(action.selector or "").first
                if await locator.count() == 0 or not await locator.is_visible():
                    continue
                tag = await locator.evaluate("el => el.tagName.toLowerCase()")
                if tag != "a":
                    raise FriendlyError("I couldn’t complete that request safely or find the requested information. Try rephrasing your task.")
                href = await locator.get_attribute("href")
                label = (await locator.inner_text()).strip()
                if not href or RISKY_TERMS.search(f"{label} {href}"):
                    raise FriendlyError("I couldn’t complete that request safely or find the requested information. Try rephrasing your task.")
                await locator.click()
                await page.wait_for_load_state("domcontentloaded", timeout=NAVIGATION_TIMEOUT_MS)
            elif action.action == "fill":
                locator = page.locator(action.selector or "").first
                if await locator.count() == 0 or not await locator.is_visible() or await locator.is_disabled():
                    continue
                tag = await locator.evaluate("el => el.tagName.toLowerCase()")
                input_type = (await locator.get_attribute("type") or "text").lower()
                if tag not in {"input", "textarea"} or input_type not in {"text", "search", "url"} or SENSITIVE_FIELD_TERMS.search((await locator.get_attribute("name") or "") + " " + (await locator.get_attribute("aria-label") or "")):
                    raise FriendlyError("I couldn’t complete that request safely or find the requested information. Try rephrasing your task.")
                await locator.fill(action.text or "")
            elif action.action == "extract":
                if action.target == "title":
                    value = (await page.title()).strip()
                elif action.target == "url":
                    value = ""
                elif action.target == "page_text":
                    value = (await page.locator("body").inner_text()).strip()
                elif action.target == "visible_text":
                    value = (await page.locator("body").inner_text()).strip()
                else:
                    value = (await page.locator(action.selector or "").first.inner_text()).strip()
                value = re.sub(r"\n{3,}", "\n\n", value).strip()[:MAX_EXTRACT_CHARS]
                if value:
                    extracted.append(value)
            steps_completed += 1
        title = (await page.title()).strip() or None
        answer = "\n\n".join(dict.fromkeys(extracted)).strip() or title or "The task finished, but I couldn’t find an answer on that page."
        return ExecuteResponse(success=True, answer=answer, page_title=title, current_url=page.url, steps_completed=steps_completed)
    except FriendlyError:
        raise
    except PlaywrightTimeoutError as exc:
        logger.info("Browser timeout during task")
        raise FriendlyError("That page took too long to respond. Please try again.") from exc
    except Exception as exc:
        logger.exception("Browser execution failure: %s", type(exc).__name__)
        raise FriendlyError("I couldn’t finish that task. Please try again with a simpler request.") from exc
    finally:
        await context.close()


@app.get("/", response_class=HTMLResponse)
async def home() -> HTMLResponse:
    with open(os.path.join(os.path.dirname(__file__), "index.html"), encoding="utf-8") as file:
        return HTMLResponse(file.read())


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/execute", response_model=ExecuteResponse)
async def execute(request: TaskRequest) -> ExecuteResponse | JSONResponse:
    try:
        async with app.state.semaphore:
            plan = await asyncio.wait_for(create_plan(request.task), timeout=TASK_TIMEOUT_SECONDS)
            return await asyncio.wait_for(execute_plan(plan, app.state.browser), timeout=TASK_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        return JSONResponse(status_code=504, content={"success": False, "answer": "That page took too long to respond. Please try again."})
    except FriendlyError as exc:
        return JSONResponse(status_code=400, content={"success": False, "answer": exc.message})
    except Exception:
        logger.exception("Unexpected request failure")
        return JSONResponse(status_code=500, content={"success": False, "answer": "I couldn’t finish that task. Please try again with a simpler request."})
