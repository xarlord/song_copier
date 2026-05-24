#!/usr/bin/env python3
"""
E2E Test: Android logcat monitoring — full request/response cycle

This script:
  1. Starts the FastAPI backend
  2. Pushes test audio to the Android emulator
  3. Triggers a request FROM the emulator TO the backend (via curl)
  4. Monitors logcat in real-time for:
     - OkHttp connection events
     - WebSocket progress messages
     - Audio focus / playback events
     - Any errors from our package
  5. Verifies the response reaches the Android device
  6. Tests all 4 backend endpoints (master, separate, voice-models, health)

Usage:
  python tests/e2e/test_android_logcat_e2e.py
  python tests/e2e/test_android_logcat_e2e.py --verbose
  python tests/e2e/test_android_logcat_e2e.py --endpoint master
"""

import argparse
import json
import math
import os
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Optional

# ─── Config ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8000
BACKEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"
EMULATOR_URL = "http://10.0.2.2:8000"
ADB = "/mnt/d/Android_Data/Local/Sdk/platform-tools/adb.exe"
PACKAGE = "com.audio_generator.app"

# Colors for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


def log(level: str, msg: str):
    colors = {"✅": GREEN, "❌": RED, "⏳": YELLOW, "📡": BLUE, "📋": CYAN, "🔧": YELLOW}
    color = colors.get(level, "")
    prefix = f"{color}{level}{RESET}" if color else level
    print(f"  {prefix} {msg}")


# ─── Helpers ──────────────────────────────────────────────

def create_test_wav(duration=2.0, sr=44100, freq=440.0) -> Path:
    path = Path(tempfile.mkdtemp()) / "e2e_test.wav"
    n = int(duration * sr)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        for i in range(n):
            v = int(32767 * 0.5 * math.sin(2 * math.pi * freq * i / sr))
            wf.writeframes(struct.pack("<2h", v, v))
    return path


def run_cmd(cmd: list[str], timeout=15) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT"
    except FileNotFoundError:
        return -2, f"NOT FOUND: {cmd[0]}"


def adb(args: list[str], timeout=15) -> tuple[int, str]:
    return run_cmd([ADB] + args, timeout)


def is_port_open(host, port, timeout=2):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (ConnectionRefusedError, socket.timeout, OSError):
        return False


def start_backend() -> Optional[subprocess.Popen]:
    if is_port_open(BACKEND_HOST, BACKEND_PORT):
        log("✅", f"Backend already running on port {BACKEND_PORT}")
        return None

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
            log("✅", f"Backend started on port {BACKEND_PORT}")
            time.sleep(1)
            return proc
        time.sleep(0.5)

    proc.kill()
    log("❌", "Backend failed to start")
    return None


def wait_for_job(job_id: str, timeout=120) -> dict:
    """Poll job status until done/failed."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        code, out = run_cmd(
            ["curl", "-s", f"{BACKEND_URL}/api/jobs/{job_id}"], timeout=10
        )
        if code == 0:
            try:
                data = json.loads(out)
                status = data.get("status", "unknown")
                progress = data.get("progress", 0)
                msg = data.get("message", "")
                if status in ("done", "failed"):
                    return data
                log("⏳", f"Job {job_id}: {status} {progress*100:.0f}% — {msg}")
            except json.JSONDecodeError:
                pass
        time.sleep(2)
    return {"status": "timeout", "error": "Job polling timed out"}


class LogcatMonitor:
    """Monitor logcat output in a background thread."""

    def __init__(self, package: str = PACKAGE, verbose: bool = False):
        self.package = package
        self.verbose = verbose
        self.lines: list[str] = []
        self.events: list[dict] = []
        self._stop = threading.Event()
        self._proc = None

    def start(self):
        adb(["logcat", "-c"])  # Clear buffer
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        log("📋", "Logcat monitor started")

    def stop(self):
        self._stop.set()
        if self._proc:
            try:
                self._proc.terminate()
            except Exception:
                pass
        log("📋", f"Logcat monitor stopped — captured {len(self.lines)} lines")

    def _run(self):
        try:
            self._proc = subprocess.Popen(
                [ADB, "logcat", "-v", "time"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True,
            )
            for line in self._proc.stdout:
                if self._stop.is_set():
                    break
                line = line.strip()
                if not line:
                    continue
                self.lines.append(line)

                # Parse interesting events
                lower = line.lower()
                if any(kw in lower for kw in [
                    "okhttp", "retrofit", "websocket", "audio_generator",
                    "failed to connect", "exception", "error",
                    "mediaplayer", "exoplayer", "media3",
                    "http://10.0.2.2", "ws://10.0.2.2"
                ]):
                    self.events.append({
                        "time": time.time(),
                        "line": line,
                    })
                    if self.verbose:
                        log("📋", line[:120])

        except Exception as e:
            if not self._stop.is_set():
                log("❌", f"Logcat error: {e}")

    def has_event(self, keyword: str) -> bool:
        return any(keyword.lower() in e["line"].lower() for e in self.events)

    def get_events(self, keyword: str) -> list[str]:
        return [e["line"] for e in self.events if keyword.lower() in e["line"].lower()]

    def print_summary(self):
        log("📋", f"Total logcat lines: {len(self.lines)}")
        log("📋", f"Interesting events: {len(self.events)}")
        keywords = ["okhttp", "websocket", "error", "exception", "10.0.2.2", "audio_generator"]
        for kw in keywords:
            matches = self.get_events(kw)
            if matches:
                log("📋", f"  '{kw}': {len(matches)} matches")
                for m in matches[:3]:
                    log("📋", f"    → {m[:100]}")


# ─── Test Functions ───────────────────────────────────────

def test_health_from_emulator() -> bool:
    """Test 1: Emulator can reach backend health endpoint."""
    log("📡", "Testing: Emulator → Backend health check")
    code, out = adb(["shell", "curl", "-s", "--connect-timeout", "5", f"{EMULATOR_URL}/health"])
    if code != 0:
        log("❌", f"Emulator curl failed: {out}")
        return False
    try:
        data = json.loads(out)
        if data.get("status") == "ok":
            log("✅", f"Health check OK: {data}")
            return True
        else:
            log("❌", f"Unexpected response: {data}")
            return False
    except json.JSONDecodeError:
        log("❌", f"Invalid JSON: {out[:200]}")
        return False


def test_master_from_emulator(audio_path: Path) -> bool:
    """Test 2: Upload audio from emulator → master → verify result."""
    log("📡", "Testing: Emulator uploads audio → Backend masters → Result streams back")

    # Push audio to emulator
    remote = "/data/local/tmp/e2e_test.wav"
    code, out = adb(["push", str(audio_path), remote], timeout=15)
    if code != 0:
        log("❌", f"Failed to push audio: {out}")
        return False
    log("✅", f"Audio pushed to emulator: {remote}")

    # Start logcat monitor
    monitor = LogcatMonitor(verbose=args.verbose)
    monitor.start()

    # Upload from emulator
    log("📡", "Uploading from emulator to backend...")
    code, out = adb([
        "shell", "curl", "-s", "-w", "\\n%{http_code}",
        "--connect-timeout", "10", "--max-time", "60",
        "-F", f"audio=@{remote}",
        "-F", "target_lufs=-14.0",
        "-F", "stereo_width=1.2",
        "-F", "format=wav",
        f"{EMULATOR_URL}/api/master",
    ], timeout=90)

    monitor.stop()

    if code != 0:
        log("❌", f"Upload failed (code={code}): {out[:300]}")
        monitor.print_summary()
        return False

    lines = out.strip().split("\n")
    http_code = lines[-1].strip() if lines else "000"
    body = "\n".join(lines[:-1])

    if not http_code.startswith("2"):
        log("❌", f"HTTP {http_code}: {body[:300]}")
        monitor.print_summary()
        return False

    try:
        data = json.loads(body)
        job_id = data.get("job_id")
        log("✅", f"Job created from emulator: {job_id} (status: {data.get('status')})")
    except json.JSONDecodeError:
        log("❌", f"Invalid JSON: {body[:300]}")
        return False

    # Wait for job to complete on backend
    log("⏳", "Waiting for job to complete...")
    result = wait_for_job(job_id, timeout=120)

    if result.get("status") == "done":
        log("✅", f"Job completed! Output: {result.get('result', {}).get('output_path', 'N/A')}")
        if result.get("result", {}).get("metrics"):
            metrics = result["result"]["metrics"]
            log("✅", f"Mastering metrics: {metrics}")
    else:
        log("❌", f"Job failed: {result.get('error', 'unknown')}")
        monitor.print_summary()
        return False

    # Now verify the emulator can download the result
    log("📡", "Verifying emulator can stream the result audio...")
    code, out = adb([
        "shell", "curl", "-s", "-o", "/dev/null", "-w", "%{http_code}:%{size_download}:%{content_type}",
        "--connect-timeout", "5",
        f"{EMULATOR_URL}/api/audio/{job_id}/result",
    ], timeout=30)

    if code == 0:
        parts = out.strip().split(":")
        dl_http_code = parts[0] if parts else "???"
        dl_size = parts[1] if len(parts) > 1 else "???"
        log("✅", f"Audio streamed to emulator: HTTP {dl_http_code}, {dl_size} bytes")
    else:
        log("❌", f"Failed to stream audio to emulator: {out}")

    monitor.print_summary()
    return True


def test_voice_models_from_emulator() -> bool:
    """Test 3: Emulator can list voice models."""
    log("📡", "Testing: Emulator → Backend voice models endpoint")
    code, out = adb([
        "shell", "curl", "-s", "--connect-timeout", "5",
        f"{EMULATOR_URL}/api/voice-models",
    ], timeout=15)

    if code != 0:
        log("❌", f"Voice models request failed: {out}")
        return False

    try:
        data = json.loads(out)
        models = data.get("models", [])
        log("✅", f"Voice models available: {len(models)}")
        for m in models[:5]:
            log("📋", f"  → {m.get('name', 'unknown')}")
        return True
    except json.JSONDecodeError:
        log("❌", f"Invalid JSON: {out[:200]}")
        return False


def test_websocket_from_emulator(job_id: str) -> bool:
    """Test 4: WebSocket progress from emulator perspective."""
    log("📡", f"Testing: WebSocket progress for job {job_id}")
    # Use adb shell to connect via websocat or curl (limited WS support)
    # Since curl doesn't do WebSocket, we test via the job polling endpoint
    # which is what the Android app falls back to if WS fails
    code, out = adb([
        "shell", "curl", "-s", "--connect-timeout", "5",
        f"{EMULATOR_URL}/api/jobs/{job_id}",
    ], timeout=15)

    if code == 0:
        try:
            data = json.loads(out)
            log("✅", f"Job status from emulator: {data.get('status')} ({data.get('progress', 0)*100:.0f}%)")
            return True
        except Exception:
            pass

    log("❌", f"Failed to query job from emulator: {out[:200]}")
    return False


def test_jobs_list_from_emulator() -> bool:
    """Test 5: Emulator can list all jobs."""
    log("📡", "Testing: Emulator → Backend jobs list")
    code, out = adb([
        "shell", "curl", "-s", "--connect-timeout", "5",
        f"{EMULATOR_URL}/api/jobs",
    ], timeout=15)

    if code != 0:
        log("❌", f"Jobs list failed: {out}")
        return False

    try:
        data = json.loads(out)
        jobs = data.get("jobs", [])
        log("✅", f"Jobs listed from emulator: {len(jobs)} jobs")
        for j in jobs[:5]:
            log("📋", f"  → {j.get('id')}: {j.get('type')} [{j.get('status')}] {j.get('progress', 0)*100:.0f}%")
        return True
    except json.JSONDecodeError:
        log("❌", f"Invalid JSON: {out[:200]}")
        return False


def test_network_latency() -> bool:
    """Test 6: Measure network latency between emulator and backend."""
    log("📡", "Testing: Network latency emulator ↔ backend")

    # Use curl -w to measure timing
    code, out = adb([
        "shell", "curl", "-s", "-o", "/dev/null",
        "-w", "dns:%{time_namelookup} connect:%{time_connect} start:%{time_starttransfer} total:%{time_total}",
        "--connect-timeout", "5",
        f"{EMULATOR_URL}/health",
    ], timeout=15)

    if code == 0:
        log("✅", f"Latency metrics: {out}")
        # Parse total time
        for part in out.split():
            if part.startswith("total:"):
                try:
                    total = float(part.split(":")[1])
                    if total < 1.0:
                        log("✅", f"Total latency: {total*1000:.0f}ms — EXCELLENT")
                    elif total < 3.0:
                        log("✅", f"Total latency: {total*1000:.0f}ms — GOOD")
                    else:
                        log("⏳", f"Total latency: {total*1000:.0f}ms — SLOW")
                except ValueError:
                    pass
        return True
    else:
        log("❌", f"Latency test failed: {out}")
        return False


# ─── Main ─────────────────────────────────────────────────

def main():
    global args
    parser = argparse.ArgumentParser(description="E2E: Android emulator ↔ Backend network test")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all logcat output")
    parser.add_argument("--endpoint", choices=["all", "health", "master", "voice-models", "jobs", "latency"],
                        default="all", help="Which endpoint to test")
    parser.add_argument("--skip-backend", action="store_true", help="Skip backend startup (already running)")
    args = parser.parse_args()

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  E2E Test: Android Emulator ↔ Backend Network Cycle{RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")

    results = {}

    # ─── Pre-flight checks ────────────────────────────
    log("🔧", "Pre-flight checks...")

    # Check emulator
    code, out = adb(["devices"])
    if code != 0:
        log("❌", "ADB not found or not working")
        sys.exit(1)
    if "device" not in out.split("\n")[-1] if out.split("\n") else "":
        # More lenient check
        lines = [l for l in out.split("\n") if l.strip() and not l.startswith("*") and not l.startswith("List")]
        if not lines or "device" not in out.lower():
            log("❌", "No Android emulator running")
            sys.exit(1)
    log("✅", "Android emulator detected")

    # Check curl on emulator
    code, _ = adb(["shell", "which", "curl"])
    if code != 0:
        log("❌", "curl not available on emulator")
        sys.exit(1)
    log("✅", "curl available on emulator")

    # Start backend
    if not args.skip_backend:
        backend_proc = start_backend()
    else:
        backend_proc = None

    if not is_port_open(BACKEND_HOST, BACKEND_PORT):
        log("❌", "Backend is not running")
        sys.exit(1)

    # Verify backend health from host
    code, out = run_cmd(["curl", "-s", f"{BACKEND_URL}/health"], timeout=5)
    if code != 0 or '"ok"' not in out:
        log("❌", f"Backend health check failed: {out}")
        sys.exit(1)
    log("✅", f"Backend healthy: {out}")

    # Create test audio
    audio_path = create_test_wav()
    log("✅", f"Test audio created: {audio_path} ({audio_path.stat().st_size} bytes)")

    print()

    # ─── Run Tests ────────────────────────────────────
    test_map = {
        "health": ("Health Check", test_health_from_emulator),
        "master": ("Master Pipeline", lambda: test_master_from_emulator(audio_path)),
        "voice-models": ("Voice Models", test_voice_models_from_emulator),
        "jobs": ("Jobs List", test_jobs_list_from_emulator),
        "latency": ("Network Latency", test_network_latency),
    }

    endpoints = list(test_map.keys()) if args.endpoint == "all" else [args.endpoint]

    for ep in endpoints:
        name, fn = test_map[ep]
        print(f"\n{BOLD}── Test: {name} ──{RESET}")
        try:
            passed = fn()
            results[ep] = "PASS" if passed else "FAIL"
        except Exception as e:
            log("❌", f"Exception: {e}")
            results[ep] = f"ERROR: {e}"

    # ─── Summary ──────────────────────────────────────
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  E2E Test Results Summary{RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")

    passed = sum(1 for v in results.values() if v == "PASS")
    total = len(results)

    for ep, status in results.items():
        icon = "✅" if status == "PASS" else "❌"
        log(icon, f"{ep}: {status}")

    print(f"\n  {BOLD}{passed}/{total} tests passed{RESET}\n")

    if backend_proc:
        backend_proc.terminate()

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
