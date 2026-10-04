
import argparse
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()

PROMPT_SIZES = {
    "small": "prompt_small.txt",
    "medium": "prompt_medium.txt",
    "large": "prompt_large.txt",
}

DEFAULT_RUNS = 10
DEFAULT_OUTPUT_TOKENS = 1536

GPU_LAYERS = 99
CONTEXT_SIZE = 4096
BATCH_SIZE = 512
UBATCH_SIZE = 256

RUN_COOLDOWN_SECONDS = 10
BLOCK_COOLDOWN_SECONDS = 120
ENERGYBRIDGE_INTERVAL_MS = 200
MONITOR_INTERVAL_SECONDS = 0.2

DEFAULT_SERVER_HOST = os.getenv("DEFAULT_SERVER_HOST")
DEFAULT_SERVER_PORT = os.getenv("DEFAULT_LLAMA_PORT")


def run_command(command, check=True):
    print("\nRunning:")
    print(" ".join(str(x) for x in command))
    print()
    result = subprocess.run(
        [str(x) for x in command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    combined = result.stdout + "\n" + result.stderr
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\n\n{combined}"
        )
    return result.stdout, result.stderr, result.returncode


def infer_architecture(model_path):
    name = Path(model_path).stem.lower()
    return "MoE" if ("a3b" in name or "moe" in name) else "Dense"


def get_model_name(model_path):
    return Path(model_path).stem


def save_metadata(path, metadata):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def sleep_with_message(seconds, reason):
    print(f"\n{reason}")
    print(f"Waiting {seconds} seconds...")
    for remaining in range(seconds, 0, -1):
        print(f"\rRemaining: {remaining:3d}s", end="", flush=True)
        time.sleep(1)
    print()


# ============================================================
# SYSTEM MONITOR
# ============================================================

def get_cpu_usage():
    try:
        import psutil
        return float(psutil.cpu_percent(interval=None))
    except Exception:
        return None


def get_memory_usage():
    try:
        import psutil
        memory = psutil.virtual_memory()
        return {
            "memory_used_mb": memory.used / 1024 / 1024,
            "memory_percent": memory.percent,
            "memory_available_mb": memory.available / 1024 / 1024,
        }
    except Exception:
        return {
            "memory_used_mb": None,
            "memory_percent": None,
            "memory_available_mb": None,
        }


def get_gpu_usage_macos():
    try:
        result = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=1,
        )
        for pattern in [
            r'"Device Utilization %" = (\d+)',
            r'"Device Utilization %"=(\d+)',
        ]:
            match = re.search(pattern, result.stdout)
            if match:
                return float(match.group(1))
    except Exception:
        pass
    return None


class SystemMonitor:
    def __init__(self, interval=MONITOR_INTERVAL_SECONDS):
        self.interval = interval
        self.running = False
        self.thread = None
        self.cpu_samples = []
        self.gpu_samples = []
        self.memory_samples = []
        self.memory_percent_samples = []

    def _monitor(self):
        try:
            import psutil
            psutil.cpu_percent(interval=None)
        except Exception:
            pass

        while self.running:
            cpu = get_cpu_usage()
            if cpu is not None:
                self.cpu_samples.append(cpu)

            gpu = get_gpu_usage_macos()
            if gpu is not None:
                self.gpu_samples.append(gpu)

            memory = get_memory_usage()
            if memory["memory_used_mb"] is not None:
                self.memory_samples.append(memory["memory_used_mb"])
                self.memory_percent_samples.append(memory["memory_percent"])

            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2)

    @staticmethod
    def average(values):
        return sum(values) / len(values) if values else None

    @staticmethod
    def maximum(values):
        return max(values) if values else None

    def results(self):
        return {
            "cpu_usage_avg_pct": self.average(self.cpu_samples),
            "cpu_usage_max_pct": self.maximum(self.cpu_samples),
            "gpu_usage_avg_pct": self.average(self.gpu_samples),
            "gpu_usage_max_pct": self.maximum(self.gpu_samples),
            "memory_avg_mb": self.average(self.memory_samples),
            "memory_max_mb": self.maximum(self.memory_samples),
            "memory_avg_pct": self.average(self.memory_percent_samples),
            "memory_max_pct": self.maximum(self.memory_percent_samples),
        }


# ============================================================
# LLAMA-SERVER
# ============================================================

def http_json(url, payload=None, timeout=30):
    if payload is None:
        request = urllib.request.Request(url, method="GET")
    else:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        return json.loads(body)


def get_server_metrics(base_url):
    with urllib.request.urlopen(
        f"{base_url}/metrics", timeout=10
    ) as response:
        text = response.read().decode("utf-8")

    metrics = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        # Prometheus names are currently exposed as llamacpp:<metric>.
        match = re.match(
            r"(?:llamacpp:)?([A-Za-z0-9_]+)(?:\{[^}]*\})?\s+([-+0-9.eE]+)$",
            line.strip(),
        )
        if match:
            metrics[match.group(1)] = float(match.group(2))
    return metrics


def wait_for_server(base_url, timeout=180):
    deadline = time.time() + timeout
    last_error = None

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"{base_url}/health", timeout=5
            ) as response:
                if response.status == 200:
                    return
        except Exception as error:
            last_error = error
        time.sleep(1)

    raise RuntimeError(
        f"llama-server did not become ready within {timeout}s. "
        f"Last error: {last_error}"
    )


def start_server(model, port, server_log):
    command = [
        "llama-server",
        "-m", str(model),
        "--host", DEFAULT_SERVER_HOST,
        "--port", str(port),
        "--metrics",
        "-ngl", str(GPU_LAYERS),
        "-c", str(CONTEXT_SIZE),
        "-b", str(BATCH_SIZE),
        "-ub", str(UBATCH_SIZE),
        "--reasoning", "off",
        "--no-cache-prompt",
        "--cache-ram", "0",
    ]

    log_file = open(server_log, "w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return process, log_file, command


def stop_server(process, log_file):
    if process is None:
        return
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    if log_file is not None:
        log_file.close()


# ============================================================
# INTERNAL REQUEST PROCESS
#
# This mode is executed INSIDE EnergiBridge. It:
#   1. reads llama-server metrics
#   2. sends exactly one HTTP request
#   3. reads llama-server metrics again
#   4. reports the counter deltas
#
# Therefore output token count comes from llama-server's own
# inference accounting, not re-tokenizing decoded text.
# ============================================================

def internal_request(args):
    base_url = f"http://{args.host}:{args.port}"

    before = get_server_metrics(base_url)

    payload = {
        "model": args.model_name,
        "messages": [
            {"role": "user", "content": args.prompt}
        ],
        "temperature": 0.6,
        "seed": 42,
        "max_tokens": args.max_tokens,
        "stream": False,
        "reasoning_effort": "none",
    }

    start = time.perf_counter()
    try:
        response = http_json(
            f"{base_url}/v1/chat/completions",
            payload=payload,
            timeout=args.timeout,
        )
    except Exception as error:
        print(json.dumps({
            "ok": False,
            "error": str(error),
        }))
        return 2

    elapsed = time.perf_counter() - start
    after = get_server_metrics(base_url)

    usage = response.get("usage") or {}
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}

    # Server metrics are authoritative for inference token counts.
    prompt_tokens = int(round(
        after.get("prompt_tokens_total", 0)
        - before.get("prompt_tokens_total", 0)
    ))
    output_tokens = int(round(
        after.get("tokens_predicted_total", 0)
        - before.get("tokens_predicted_total", 0)
    ))

    prompt_seconds = (
        after.get("prompt_seconds_total", 0)
        - before.get("prompt_seconds_total", 0)
    )
    generation_seconds = (
        after.get("tokens_predicted_seconds_total", 0)
        - before.get("tokens_predicted_seconds_total", 0)
    )

    result = {
        "ok": True,
        "response_text": message.get("content", ""),
        "reasoning_content": message.get("reasoning_content"),
        "finish_reason": choice.get("finish_reason"),
        "response_usage": usage,
        "server_metrics_before": before,
        "server_metrics_after": after,
        "prompt_tokens_runtime": prompt_tokens,
        "output_tokens_runtime": output_tokens,
        "prompt_processing_time_s": prompt_seconds,
        "generation_time_s": generation_seconds,
        "request_wall_time_s": elapsed,
        "prompt_tokens_per_second": (
            prompt_tokens / prompt_seconds
            if prompt_seconds > 0 else None
        ),
        "generation_tokens_per_second": (
            output_tokens / generation_seconds
            if generation_seconds > 0 else None
        ),
    }

    print(json.dumps(result, ensure_ascii=False))
    return 0


# ============================================================
# MEASURED RUN
# ============================================================

def run_single(
    model,
    quantization,
    architecture,
    prompt_size,
    prompt_file,
    run_number,
    output_tokens,
    raw_root,
    server_host,
    server_port,
    order_seed,
    schedule_index,
):
    model_name = get_model_name(model)

    run_dir = (
        raw_root
        / model_name
        / quantization
        / prompt_size
        / f"run_{run_number:02d}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    energy_csv = run_dir / "energy.csv"
    output_txt = run_dir / "output.txt"
    client_json = run_dir / "client_result.json"
    stderr_txt = run_dir / "stderr.txt"

    prompt = prompt_file.read_text(encoding="utf-8").strip()

    print()
    print("#" * 80)
    print(f"MODEL        : {model_name}")
    print(f"QUANTIZATION : {quantization}")
    print(f"ARCHITECTURE : {architecture}")
    print(f"PROMPT SIZE  : {prompt_size}")
    print(f"RUN          : {run_number}")
    print(f"OUTPUT TOKENS: {output_tokens}")
    print(f"SERVER       : http://{server_host}:{server_port}")
    print("#" * 80)

    # Run this same Python script as a child inside EnergiBridge.
    # The child performs exactly one HTTP inference request.
    request_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--internal-request",
        "--host", server_host,
        "--port", str(server_port),
        "--model-name", model_name,
        "--prompt-json", json.dumps(prompt),
        "--max-tokens", str(output_tokens),
        "--timeout", "900",
    ]

    client_result_tmp = run_dir / "energibridge_client_output.txt"

    energy_command = [
        "energibridge",
        "-o", str(energy_csv),
        "-c", str(client_result_tmp),
        "-i", str(ENERGYBRIDGE_INTERVAL_MS),
        "-g",
        "--summary",
        "--",
    ] + request_command

    monitor = SystemMonitor()
    start_datetime = datetime.now().isoformat()
    wall_start = time.perf_counter()

    monitor.start()
    try:
        stdout, stderr, returncode = run_command(energy_command)
    except Exception as error:
        stderr_txt.write_text(
            str(error) + "\n" + stderr,
            encoding="utf-8",
        )
        raise
    finally:
        monitor.stop()

    wall_end = time.perf_counter()
    wall_clock_time = wall_end - wall_start

    if returncode != 0:
        raise RuntimeError(
            f"EnergiBridge/client failed:\n{stdout}\n{stderr}"
        )

    try:
        client_output = client_result_tmp.read_text(
            encoding="utf-8", errors="replace"
        ).strip()
        client_result = json.loads(client_output)
    except Exception as error:
        stderr_txt.write_text(
            f"Could not parse client result: {error}\n"
            f"Raw client output:\n{client_output if 'client_output' in locals() else ''}\n"
            f"EnergiBridge stdout:\n{stdout}\n"
            f"EnergiBridge stderr:\n{stderr}",
            encoding="utf-8",
        )
        raise RuntimeError(
            f"Could not parse llama-server client result: {error}"
        )

    if not client_result.get("ok"):
        raise RuntimeError(
            f"llama-server request failed: {client_result.get('error')}"
        )

    response_text = client_result.get("response_text", "")
    output_txt.write_text(response_text, encoding="utf-8")
    client_json.write_text(
        json.dumps(client_result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    monitor_metrics = monitor.results()

    metadata = {
        "model": model_name,
        "architecture": architecture,
        "model_path": str(model),
        "quantization": quantization,
        "prompt_size": prompt_size,
        "prompt_file": str(prompt_file),
        "run": run_number,
        "schedule_index": schedule_index,
        "order_seed": order_seed,
        "requested_output_tokens": output_tokens,
        "actual_output_tokens": client_result.get("output_tokens_runtime"),
        "measurement_start": start_datetime,
        "measurement_end": datetime.now().isoformat(),
        "wall_clock_time_s": wall_clock_time,
        "energibridge_interval_ms": ENERGYBRIDGE_INTERVAL_MS,
        "energy_measurement": {
            "source": "EnergiBridge",
            "method": "Sampled system power integrated over time from energy.csv while one llama-server inference HTTP request was executed.",
            "unit": "J",
            "scope": "Local Mac system power interface exposed by EnergiBridge; not a wall-plug measurement unless that interface represents the complete system.",
        },
        "llama_metrics": {
            "prompt_tokens": client_result.get("prompt_tokens_runtime"),
            "output_tokens": client_result.get("output_tokens_runtime"),
            "prompt_token_count_source": "llama-server /metrics",
            "output_token_count_source": "llama-server /metrics",
            "prompt_eval_time_ms": (
                client_result.get("prompt_processing_time_s") * 1000
                if client_result.get("prompt_processing_time_s") is not None
                else None
            ),
            "generation_time_ms": (
                client_result.get("generation_time_s") * 1000
                if client_result.get("generation_time_s") is not None
                else None
            ),
            "prompt_tokens_per_second": client_result.get(
                "prompt_tokens_per_second"
            ),
            "generation_tokens_per_second": client_result.get(
                "generation_tokens_per_second"
            ),
            "request_wall_time_s": client_result.get(
                "request_wall_time_s"
            ),
            "finish_reason": client_result.get("finish_reason"),
            "response_usage": client_result.get("response_usage"),
        },
        "server_metrics": {
            "before": client_result.get("server_metrics_before"),
            "after": client_result.get("server_metrics_after"),
        },
        "system_metrics": monitor_metrics,
        "configuration": {
            "architecture": architecture,
            "deployment": "on-device",
            "runtime": "llama-server",
            "server_host": server_host,
            "server_port": server_port,
            "gpu_layers": GPU_LAYERS,
            "context_size": CONTEXT_SIZE,
            "batch_size": BATCH_SIZE,
            "ubatch_size": UBATCH_SIZE,
            "output_tokens": output_tokens,
            "temperature": 0.6,
            "seed": 42,
            "reasoning": "off",
            "prompt_cache_enabled": False,
            "cache_ram_mb": 0,
        },
    }

    save_metadata(run_dir / "metadata.json", metadata)

    # Keep the per-run artifacts self-contained.
    if client_result_tmp.exists():
        client_result_tmp.unlink()

    print("\nRun complete.")
    print(f"Wall time: {wall_clock_time:.3f}s")
    print("\nllama-server metrics:")
    for key in [
        "prompt_tokens",
        "output_tokens",
        "prompt_eval_time_ms",
        "generation_time_ms",
        "prompt_tokens_per_second",
        "generation_tokens_per_second",
    ]:
        print(f"  {key}: {metadata['llama_metrics'].get(key)}")

    print(
        f"  output_token_count_source: "
        f"{metadata['llama_metrics']['output_token_count_source']}"
    )

    print("\nSystem metrics:")
    for key, value in monitor_metrics.items():
        print(f"  {key}: {value}")

    return run_dir


# ============================================================
# WARM-UP
# ============================================================

def warmup(model_name, prompt_file, output_tokens, host, port):
    prompt = prompt_file.read_text(encoding="utf-8").strip()

    print()
    print("=" * 70)
    print(f"WARM-UP: {prompt_file.name}")
    print("=" * 70)

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.6,
        "seed": 42,
        "max_tokens": output_tokens,
        "stream": False,
        "reasoning_effort": "none",
    }

    response = http_json(
        f"http://{host}:{port}/v1/chat/completions",
        payload=payload,
        timeout=900,
    )

    usage = response.get("usage") or {}
    print(
        f"Warm-up complete. "
        f"prompt_tokens={usage.get('prompt_tokens')}, "
        f"completion_tokens={usage.get('completion_tokens')}"
    )


# ============================================================
# INTERNAL CLI
# ============================================================

def parse_internal_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--internal-request", action="store_true")
    parser.add_argument("--host", default=DEFAULT_SERVER_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_SERVER_PORT)
    parser.add_argument("--model-name", default="")
    parser.add_argument("--prompt-json", default="")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_OUTPUT_TOKENS)
    parser.add_argument("--timeout", type=int, default=900)
    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():
    # IMPORTANT: only parse the internal-request arguments when this
    # process was explicitly launched as the EnergiBridge child.
    # Calling parse_internal_args() unconditionally would make the
    # internal parser reject normal benchmark arguments such as
    # --model, --quantization, --runs, and --architecture.
    if "--internal-request" in sys.argv:
        internal = parse_internal_args()
        args = internal
        args.prompt = json.loads(args.prompt_json)
        return internal_request(args)

    parser = argparse.ArgumentParser(
        description="Run local LLM energy benchmark using llama-server."
    )

    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--quantization", required=True, help="Quantization label")
    parser.add_argument(
        "--architecture",
        choices=["Dense", "MoE"],
        default=None,
        help="Model architecture. If omitted, infer from model name.",
    )
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument(
        "--output-tokens",
        type=int,
        default=DEFAULT_OUTPUT_TOKENS,
    )
    parser.add_argument("--prompt-dir", default="prompts/")
    parser.add_argument("--results-dir", default="results/raw")
    parser.add_argument("--server-host", default=DEFAULT_SERVER_HOST)
    parser.add_argument("--server-port", type=int, default=DEFAULT_SERVER_PORT)
    parser.add_argument("--skip-cooldown", action="store_true")
    parser.add_argument(
        "--order-seed",
        type=int,
        default=42,
        help="Seed for balanced random prompt-size ordering.",
    )

    args = parser.parse_args()

    model = Path(args.model).expanduser().resolve()
    if not model.exists():
        print(f"ERROR: model does not exist: {model}")
        sys.exit(1)

    architecture = args.architecture or infer_architecture(model)
    prompt_dir = Path(args.prompt_dir).expanduser().resolve()
    raw_root = Path(args.results_dir).expanduser().resolve()

    for size, filename in PROMPT_SIZES.items():
        prompt_file = prompt_dir / filename
        if not prompt_file.exists():
            print(f"ERROR: missing prompt: {prompt_file}")
            sys.exit(1)

    print("=" * 70)
    print("LOCAL LLM ENERGY EXPERIMENT — LLAMA-SERVER")
    print("=" * 70)
    print(f"Model:          {model.name}")
    print(f"Quantization:   {args.quantization}")
    print(f"Architecture:   {architecture}")
    print(f"Runs/size:      {args.runs}")
    print(f"Output tokens:  {args.output_tokens}")
    print(f"GPU layers:     {GPU_LAYERS}")
    print(f"Context:        {CONTEXT_SIZE}")
    print(f"Batch:          {BATCH_SIZE}")
    print(f"Micro-batch:    {UBATCH_SIZE}")
    print(f"Server:         http://{args.server_host}:{args.server_port}")
    print("=" * 70)

    input("\nPress ENTER to start...")

    server_log = raw_root / "llama-server.log"
    server_log.parent.mkdir(parents=True, exist_ok=True)

    server = None
    log_file = None

    try:
        print("\nStarting llama-server...")
        server, log_file, server_command = start_server(
            model,
            args.server_port,
            server_log,
        )
        print(" ".join(server_command))
        wait_for_server(
            f"http://{args.server_host}:{args.server_port}"
        )
        print("llama-server is ready.")

        sizes = list(PROMPT_SIZES.items())
        order_rng = random.Random(args.order_seed)

        warmup_sizes = sizes.copy()
        order_rng.shuffle(warmup_sizes)
        for _, filename in warmup_sizes:
            warmup(
                model.name,
                prompt_dir / filename,
                args.output_tokens,
                args.server_host,
                args.server_port,
            )

        if not args.skip_cooldown:
            sleep_with_message(
                30,
                "Stabilizing after warm-up..."
            )

        # Each block contains one repetition of every prompt size. Shuffling
        # each block balances prompt sizes while avoiding fixed thermal order.
        schedule = []
        for run_number in range(1, args.runs + 1):
            block = sizes.copy()
            order_rng.shuffle(block)
            schedule.extend(
                (run_number, prompt_size, filename)
                for prompt_size, filename in block
            )

        print("\nMeasured prompt order:")
        print("  " + " -> ".join(size for _, size, _ in schedule))

        for schedule_index, (run_number, prompt_size, filename) in enumerate(schedule):
            prompt_file = prompt_dir / filename
            run_single(
                model=model,
                quantization=args.quantization,
                architecture=architecture,
                prompt_size=prompt_size,
                prompt_file=prompt_file,
                run_number=run_number,
                output_tokens=args.output_tokens,
                raw_root=raw_root,
                server_host=args.server_host,
                server_port=args.server_port,
                order_seed=args.order_seed,
                schedule_index=schedule_index,
            )

            if schedule_index < len(schedule) - 1 and not args.skip_cooldown:
                sleep_with_message(
                    RUN_COOLDOWN_SECONDS,
                    "Cooldown between runs..."
                )

    finally:
        print("\nStopping llama-server...")
        stop_server(server, log_file)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"Raw results saved to:\n{raw_root}")


if __name__ == "__main__":
    main()
