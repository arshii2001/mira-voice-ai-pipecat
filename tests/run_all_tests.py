#!/usr/bin/env python3
"""
Master test runner for Mira Voice AI.

Runs all test suites in sequence and produces a unified summary.
Works both locally (Docker) and in K8s (as a Job).

Usage:
    # Run all suites (default)
    python tests/run_all_tests.py

    # Run specific suites
    python tests/run_all_tests.py --suites language,classroom,text,rtvi

    # Run with verbose output
    python tests/run_all_tests.py --verbose

    # Target a specific server (default: auto-detect)
    python tests/run_all_tests.py --host http://localhost:7860

Environment variables (auto-set if --host is provided):
    PIPECAT_HTTP_URL    (default: http://mira-voice:7860)
    PIPECAT_WS_URL      (default: ws://mira-voice:7860/ws)
    CLASSROOM_WS_URL    (default: ws://mira-voice:7860/classroom/rooms)
"""

import argparse
import os
import subprocess
import sys
import time


# ─────────────────────────────────────────────────
# Test Suite Registry
# ─────────────────────────────────────────────────
# Each suite: (name, script_path, description, requires_llm)
SUITES = [
    ("language",   "tests/test_language_adherence.py", "Language adherence (11 tests)",       True),
    ("classroom",  "tests/test_classroom_mode.py",     "Classroom mode (25 tests)",           True),
    ("text",       "tests/test_text_mode.py",          "Text mode (8 tests)",                 True),
    ("rtvi",       "tests/test_rtvi.py",               "RTVI protocol (5 tests)",             False),
    ("performance","tests/test_performance.py",        "Performance benchmarks",              True),
]


def run_suite(name: str, script: str, verbose: bool = False) -> dict:
    """Run a single test suite, streaming output in real-time."""
    print()
    print("=" * 70)
    print(f"  SUITE: {name.upper()}")
    print("=" * 70)
    sys.stdout.flush()

    # performance suite uses --quick instead of --test all
    if name == "performance":
        cmd = [sys.executable, "-u", script, "--quick"]  # -u = unbuffered
    else:
        cmd = [sys.executable, "-u", script, "--test", "all"]  # -u = unbuffered
    if verbose:
        cmd.append("--verbose")

    t0 = time.time()
    try:
        # Stream output in real-time and capture it
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line-buffered
        )

        output_lines = []
        for line in process.stdout:
            print(line, end="", flush=True)
            output_lines.append(line)

        process.wait(timeout=300)
        duration = time.time() - t0
        exit_code = process.returncode

        # Extract pass/fail counts from captured output
        output = "".join(output_lines)
        passed = 0
        total = 0
        for line in output.split("\n"):
            line = line.strip()
            if "/" in line and "tests passed" in line:
                # e.g. "  23/25 tests passed"
                parts = line.split("/")
                try:
                    passed = int(parts[0].strip())
                    total = int(parts[1].split()[0])
                except (ValueError, IndexError):
                    pass

        return {
            "name": name,
            "exit_code": exit_code,
            "passed": passed,
            "total": total,
            "duration": duration,
            "error": "",
        }

    except subprocess.TimeoutExpired:
        process.kill()
        return {
            "name": name,
            "exit_code": -1,
            "passed": 0,
            "total": 0,
            "duration": 300,
            "error": "TIMEOUT (5 min)",
        }
    except Exception as e:
        return {
            "name": name,
            "exit_code": -1,
            "passed": 0,
            "total": 0,
            "duration": time.time() - t0,
            "error": str(e)[:200],
        }


def main():
    parser = argparse.ArgumentParser(description="Mira Voice AI - Master Test Runner")
    parser.add_argument(
        "--suites",
        type=str,
        default="all",
        help="Comma-separated list of suites to run (default: all). "
             f"Available: {', '.join(s[0] for s in SUITES)}"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output from each suite")
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Target server URL (e.g. http://localhost:7860). Auto-sets env vars."
    )
    args = parser.parse_args()

    # Set environment variables if --host is provided
    if args.host:
        host = args.host.rstrip("/")
        ws_host = host.replace("http://", "ws://").replace("https://", "wss://")
        os.environ["PIPECAT_HTTP_URL"] = host
        os.environ["PIPECAT_WS_URL"] = f"{ws_host}/ws"
        os.environ["CLASSROOM_WS_URL"] = f"{ws_host}/classroom/rooms"
        print(f"Target: {host}")

    # Filter suites
    if args.suites == "all":
        suites_to_run = SUITES
    else:
        requested = set(args.suites.split(","))
        suites_to_run = [s for s in SUITES if s[0] in requested]
        unknown = requested - {s[0] for s in SUITES}
        if unknown:
            print(f"WARNING: Unknown suites: {unknown}")
            print(f"Available: {', '.join(s[0] for s in SUITES)}")

    if not suites_to_run:
        print("No suites to run!")
        sys.exit(1)

    print()
    print("=" * 70)
    print("  MIRA VOICE AI — MASTER TEST RUNNER")
    print("=" * 70)
    print(f"  Suites: {', '.join(s[0] for s in suites_to_run)}")
    print(f"  Server: {os.getenv('PIPECAT_HTTP_URL', 'http://mira-voice:7860')}")
    print("=" * 70)

    t_start = time.time()
    results = []

    for name, script, desc, requires_llm in suites_to_run:
        result = run_suite(name, script, verbose=args.verbose)
        results.append(result)

    total_duration = time.time() - t_start

    # ─────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────
    print()
    print()
    print("=" * 70)
    print("  MASTER TEST RESULTS")
    print("=" * 70)

    total_passed = 0
    total_tests = 0
    all_suites_ok = True

    for r in results:
        status = "PASS" if r["exit_code"] == 0 else "FAIL"
        if r["exit_code"] != 0:
            all_suites_ok = False

        duration_str = f"{r['duration']:.1f}s"
        counts = f"{r['passed']}/{r['total']}" if r['total'] > 0 else ("OK" if r["exit_code"] == 0 else "ERR")

        print(f"  {status}  {r['name']:<20} {counts:<10} {duration_str}")
        if r["error"]:
            print(f"       error: {r['error'][:100]}")

        if r["total"] > 0:
            total_passed += r["passed"]
            total_tests += r["total"]

    print("-" * 70)
    print(f"  Total: {total_passed}/{total_tests} tests passed across {len(results)} suites")
    print(f"  Duration: {total_duration:.1f}s")
    print("=" * 70)

    if not all_suites_ok:
        failed_suites = [r["name"] for r in results if r["exit_code"] != 0]
        print(f"\n  FAILED suites: {', '.join(failed_suites)}")
        sys.exit(1)
    else:
        print("\n  ALL SUITES PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
