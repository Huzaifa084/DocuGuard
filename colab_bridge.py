#!/usr/bin/env python3
"""
DocuGuard+ Colab MCP Bridge
============================
CLI tool for VS Code / Copilot to interact with the remote MCP Debug Server
running on Google Colab.

Usage:
    python colab_bridge.py --url <MCP_TUNNEL_URL> <command> [args]

Commands:
    status          Full status (Ollama + GPU + system)
    gpu             GPU memory and utilization
    ollama          Ollama health and loaded models
    logs [N]        Last N lines of Ollama log (default: 50)
    error           Get last error traceback
    history [N]     Last N execution results (default: 10)
    system          CPU, RAM, disk usage
    run <code>      Execute Python code on Colab
    run-file <path> Execute a local Python file on Colab
    test            Quick connectivity + inference test

Examples:
    python colab_bridge.py --url https://abc-xyz.trycloudflare.com status
    python colab_bridge.py --url https://abc-xyz.trycloudflare.com logs 100
    python colab_bridge.py --url https://abc-xyz.trycloudflare.com run "print(1+1)"
    python colab_bridge.py --url https://abc-xyz.trycloudflare.com run-file fix_script.py
"""

import argparse
import json
import sys
import os

try:
    import requests
except ImportError:
    print("ERROR: 'requests' not installed. Run: pip install requests")
    sys.exit(1)


# ─── Helpers ──────────────────────────────────────────────

TIMEOUT = 30  # seconds


def _get(url: str, params: dict = None) -> dict:
    """GET request with error handling."""
    try:
        r = requests.get(url, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        print(f"ERROR: Cannot connect to {url}")
        print("       Is the Colab notebook running? Is the tunnel active?")
        sys.exit(1)
    except requests.Timeout:
        print(f"ERROR: Request timed out ({TIMEOUT}s)")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


def _post(url: str, data: dict) -> dict:
    """POST request with error handling."""
    try:
        r = requests.post(url, json=data, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        print(f"ERROR: Cannot connect to {url}")
        print("       Is the Colab notebook running? Is the tunnel active?")
        sys.exit(1)
    except requests.Timeout:
        print(f"ERROR: Request timed out ({TIMEOUT}s)")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


def _pp(data: dict):
    """Pretty-print JSON."""
    print(json.dumps(data, indent=2, default=str))


# ─── Commands ─────────────────────────────────────────────

def cmd_status(base_url: str, _args):
    """Full status report: health + ollama + GPU + system."""
    print("=" * 60)
    print("  DocuGuard+ Colab MCP Status")
    print("=" * 60)

    # Health
    health = _get(f"{base_url}/health")
    print(f"\n🔧 MCP Server: OK (uptime {health.get('uptime_min', '?')} min)")

    # Ollama
    ollama = _get(f"{base_url}/get_ollama_status")
    if ollama.get("running"):
        models = ", ".join(ollama.get("models", [])) or "none"
        print(f"🚀 Ollama: RUNNING ({ollama.get('model_count', 0)} models: {models})")
    else:
        print(f"⚠️  Ollama: NOT RUNNING — {ollama.get('error', 'unknown')}")

    # GPU
    gpu = _get(f"{base_url}/get_gpu_usage")
    if "error" not in gpu:
        print(f"🎮 GPU: {gpu.get('gpu_name', '?')} — "
              f"{gpu.get('memory_used_mb', '?')}/{gpu.get('memory_total_mb', '?')} MB used, "
              f"{gpu.get('gpu_utilization_pct', '?')}% util")
    else:
        print(f"⚠️  GPU: {gpu.get('error', 'unavailable')}")

    # System
    sys_info = _get(f"{base_url}/get_system_info")
    print(f"💻 System: CPU {sys_info.get('cpu_percent', '?')}% | "
          f"RAM {sys_info.get('ram_used_gb', '?')}/{sys_info.get('ram_total_gb', '?')} GB "
          f"({sys_info.get('ram_percent', '?')}%) | "
          f"Disk {sys_info.get('disk_free_gb', '?')} GB free")

    # Last error
    err = _get(f"{base_url}/get_last_error")
    if err.get("has_error"):
        print(f"\n❌ Last Error:\n{err.get('last_error', '')[:500]}")
    else:
        print(f"\n✅ No recent errors")

    print("=" * 60)


def cmd_gpu(base_url: str, _args):
    """GPU memory and utilization."""
    _pp(_get(f"{base_url}/get_gpu_usage"))


def cmd_ollama(base_url: str, _args):
    """Ollama health and loaded models."""
    _pp(_get(f"{base_url}/get_ollama_status"))


def cmd_logs(base_url: str, args):
    """Last N lines of Ollama log."""
    n = int(args[0]) if args else 50
    data = _get(f"{base_url}/get_logs", params={"lines": n})
    if data.get("error"):
        print(f"ERROR: {data['error']}")
    else:
        print(data.get("log", ""))


def cmd_error(base_url: str, _args):
    """Get last error traceback."""
    data = _get(f"{base_url}/get_last_error")
    if data.get("has_error"):
        print("❌ Last Error Traceback:")
        print(data.get("last_error", ""))
    else:
        print("✅ No errors recorded.")


def cmd_history(base_url: str, args):
    """Last N execution results."""
    n = int(args[0]) if args else 10
    _pp(_get(f"{base_url}/get_execution_history", params={"last": n}))


def cmd_system(base_url: str, _args):
    """CPU, RAM, disk usage."""
    _pp(_get(f"{base_url}/get_system_info"))


def cmd_run(base_url: str, args):
    """Execute Python code on Colab."""
    if not args:
        print("ERROR: No code provided. Usage: run \"print('hello')\"")
        sys.exit(1)
    code = " ".join(args)
    data = _post(f"{base_url}/run_cell", {"code": code})
    if data.get("status") == "success":
        print("✅ Execution successful")
        if data.get("output"):
            print("--- Output ---")
            print(data["output"])
    else:
        print("❌ Execution failed")
        print(data.get("traceback", data.get("error", "unknown error")))


def cmd_run_file(base_url: str, args):
    """Execute a local Python file on Colab."""
    if not args:
        print("ERROR: No file path provided. Usage: run-file script.py")
        sys.exit(1)
    filepath = args[0]
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        sys.exit(1)
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()
    print(f"Sending {len(code)} chars from {filepath}...")
    data = _post(f"{base_url}/run_cell", {"code": code})
    if data.get("status") == "success":
        print("✅ Execution successful")
        if data.get("output"):
            print("--- Output ---")
            print(data["output"])
    else:
        print("❌ Execution failed")
        print(data.get("traceback", data.get("error", "unknown error")))


def cmd_test(base_url: str, _args):
    """Connectivity + quick inference test."""
    print("Testing MCP connection...")
    health = _get(f"{base_url}/health")
    print(f"✅ MCP Server: {health}")

    print("\nTesting Ollama...")
    ollama = _get(f"{base_url}/get_ollama_status")
    print(f"   Running: {ollama.get('running')}, Models: {ollama.get('models')}")

    if ollama.get("running") and ollama.get("models"):
        model = ollama["models"][0]
        print(f"\nRunning inference test with {model}...")
        result = _post(f"{base_url}/run_cell", {
            "code": f"""
import requests, time
start = time.time()
r = requests.post("http://localhost:11434/api/generate", json={{
    "model": "{model}",
    "prompt": "Say hello in one sentence.",
    "stream": False,
}})
elapsed = time.time() - start
resp = r.json().get("response", "")
print(f"Model: {model}")
print(f"Response: {{resp[:200]}}")
print(f"Time: {{elapsed:.2f}}s")
"""
        })
        if result.get("status") == "success":
            print(result.get("output", ""))
        else:
            print(f"❌ Inference test failed: {result.get('error', '')}")
    else:
        print("⚠️ Skipping inference test — no models loaded")


# ─── CLI ──────────────────────────────────────────────────

COMMANDS = {
    "status": cmd_status,
    "gpu": cmd_gpu,
    "ollama": cmd_ollama,
    "logs": cmd_logs,
    "error": cmd_error,
    "history": cmd_history,
    "system": cmd_system,
    "run": cmd_run,
    "run-file": cmd_run_file,
    "test": cmd_test,
}


def main():
    parser = argparse.ArgumentParser(
        description="DocuGuard+ Colab MCP Bridge — Remote debug interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  status          Full status (Ollama + GPU + system + errors)
  gpu             GPU memory and utilization
  ollama          Ollama health and loaded models
  logs [N]        Last N lines of Ollama log (default: 50)
  error           Get last error traceback
  history [N]     Last N execution results (default: 10)
  system          CPU, RAM, disk usage
  run <code>      Execute Python code on Colab
  run-file <path> Execute a local Python file on Colab
  test            Quick connectivity + inference test
""",
    )
    parser.add_argument("--url", required=True, help="MCP tunnel URL (https://xxx.trycloudflare.com)")
    parser.add_argument("command", help="Command to execute")
    parser.add_argument("args", nargs="*", help="Command arguments")

    parsed = parser.parse_args()

    base_url = parsed.url.rstrip("/")
    cmd_name = parsed.command.lower()

    if cmd_name not in COMMANDS:
        print(f"Unknown command: {cmd_name}")
        print(f"Available: {', '.join(COMMANDS.keys())}")
        sys.exit(1)

    COMMANDS[cmd_name](base_url, parsed.args)


if __name__ == "__main__":
    main()
