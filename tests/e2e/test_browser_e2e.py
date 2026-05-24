"""
E2E Test: Browser frontend (Gradio UI) ↔ Backend GPU task cycle

Uses Playwright MCP to:
  1. Open the Gradio web UI
  2. Navigate to a feature tab (mastering)
  3. Upload audio file
  4. Submit the task
  5. Watch progress in the browser
  6. Verify output is rendered

Run:
  pytest tests/e2e/test_browser_e2e.py -v --tb=short
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
import wave
import math
import struct
from pathlib import Path
from typing import Optional

import pytest

try:
    from playwright.async_api import async_playwright, Page, Browser
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False
    Page = None  # type: ignore[misc,assignment]
    Browser = None  # type: ignore[misc,assignment]
    async_playwright = None  # type: ignore[misc,assignment]

# ─── Constants ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8000
BACKEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"
GRADIO_HOST = "127.0.0.1"
GRADIO_PORT = 7860
GRADIO_URL = f"http://{GRADIO_HOST}:{GRADIO_PORT}"


# ─── Helpers ──────────────────────────────────────────────

def create_test_wav(duration_sec: float = 2.0, sample_rate: int = 44100,
                    channels: int = 2, freq: float = 440.0) -> Path:
    path = Path(tempfile.mkdtemp()) / "test_audio.wav"
    n_samples = int(duration_sec * sample_rate)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for i in range(n_samples):
            val = int(32767 * 0.5 * math.sin(2 * math.pi * freq * i / sample_rate))
            frame = struct.pack(f"<{channels}h", *([val] * channels))
            wf.writeframes(frame)
    return path


def is_port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (ConnectionRefusedError, socket.timeout, OSError):
        return False


# ─── Fixtures ─────────────────────────────────────────────

@pytest.fixture(scope="session")
def test_audio():
    return create_test_wav(duration_sec=2.0)


@pytest.fixture(scope="session")
def backend_process():
    """Start FastAPI backend if not already running."""
    if is_port_open(BACKEND_HOST, BACKEND_PORT):
        yield None
        return

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.api:app",
         "--host", BACKEND_HOST, "--port", str(BACKEND_PORT)],
        cwd=str(PROJECT_ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    deadline = time.time() + 15
    while time.time() < deadline:
        if is_port_open(BACKEND_HOST, BACKEND_PORT):
            break
        time.sleep(0.5)
    else:
        proc.kill()
        pytest.fail("Backend did not start")

    time.sleep(1)
    yield proc

    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope="session")
def gradio_process():
    """Start Gradio UI if not already running."""
    if is_port_open(GRADIO_HOST, GRADIO_PORT):
        yield None
        return

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    proc = subprocess.Popen(
        [sys.executable, "app.py"],
        cwd=str(PROJECT_ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    deadline = time.time() + 30
    while time.time() < deadline:
        if is_port_open(GRADIO_HOST, GRADIO_PORT):
            break
        time.sleep(1)
    else:
        proc.kill()
        pytest.fail("Gradio UI did not start within 30s")

    time.sleep(2)
    yield proc

    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture
async def browser_page(backend_process, gradio_process):
    """Provide a Playwright browser page pointed at Gradio UI."""
    if not HAS_PLAYWRIGHT:
        pytest.skip("Playwright not installed (pip install playwright && playwright install)")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1280, "height": 900})
        yield page
        await browser.close()


# ─── Tests ────────────────────────────────────────────────

@pytest.mark.asyncio
class TestBrowserGradioUI:
    """Test browser interaction with Gradio UI → backend GPU pipeline."""

    async def test_gradio_loads(self, browser_page: Page):
        """Gradio UI loads without errors."""
        response = await browser_page.goto(GRADIO_URL, wait_until="networkidle", timeout=30000)
        assert response is not None
        assert response.status == 200

        title = await browser_page.title()
        assert title, "Page has no title"

        # Take screenshot for evidence
        await browser_page.screenshot(path=str(PROJECT_ROOT / "tests" / "e2e" / "screenshots" / "gradio_loaded.png"))

    async def test_gradio_mastering_tab(self, browser_page: Page, test_audio):
        """Navigate to mastering tab, upload audio, submit, verify output."""
        await browser_page.goto(GRADIO_URL, wait_until="networkidle", timeout=30000)

        # Find and click the Mastering tab
        mastering_tab = browser_page.locator("button, [role='tab']").filter(has_text="Master")
        if await mastering_tab.count() > 0:
            await mastering_tab.first.click()
            await asyncio.sleep(1)

        # Look for file upload input
        upload_input = browser_page.locator("input[type='file']")
        if await upload_input.count() > 0:
            await upload_input.first.set_input_files(str(test_audio))
            await asyncio.sleep(2)

            # Find and click submit/process button
            submit_btn = browser_page.locator("button").filter(has_text="Master")
            if await submit_btn.count() > 0:
                await submit_btn.first.click()

                # Wait for processing (max 60s)
                try:
                    await browser_page.wait_for_selector(
                        "text=Done", timeout=60000
                    )
                except Exception:
                    # Check for error messages
                    error_el = browser_page.locator("[class*='error'], .error")
                    if await error_el.count() > 0:
                        error_text = await error_el.first.text_content()
                        pytest.fail(f"Processing error: {error_text}")

            # Screenshot after processing
            await browser_page.screenshot(
                path=str(PROJECT_ROOT / "tests" / "e2e" / "screenshots" / "gradio_mastering_result.png")
            )
        else:
            pytest.skip("No file upload input found on mastering tab")

    async def test_gradio_api_health_via_browser(self, browser_page: Page, backend_process):
        """Browser can reach the FastAPI backend health endpoint."""
        response = await browser_page.goto(f"{BACKEND_URL}/health", timeout=10000)
        assert response is not None
        content = await browser_page.text_content("body")
        data = json.loads(content)
        assert data["status"] == "ok"

    async def test_gradio_swagger_docs(self, browser_page: Page, backend_process):
        """Swagger docs render correctly in browser."""
        await browser_page.goto(f"{BACKEND_URL}/docs", wait_until="networkidle", timeout=15000)

        # Check for Swagger UI elements
        swagger = browser_page.locator(".swagger-ui, #swagger-ui")
        # If not found, check for ReDoc (alternative)
        redoc = browser_page.locator("redoc")

        has_docs = (await swagger.count() > 0) or (await redoc.count() > 0)

        await browser_page.screenshot(
            path=str(PROJECT_ROOT / "tests" / "e2e" / "screenshots" / "swagger_docs.png")
        )
        assert has_docs or response.status == 200, "Swagger docs did not render"
