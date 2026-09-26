import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

PROMPT_SIZES = {
    "small": "prompt_small.txt",
    "medium": "prompt_medium.txt",
    "large": "prompt_large.txt",
}

DEFAULT_RUNS = 10

# IMPORTANT: fixed maximum output length
DEFAULT_OUTPUT_TOKENS = 1024

# Your verified working llama.cpp configuration
GPU_LAYERS = 99
CONTEXT_SIZE = 4096
BATCH_SIZE = 512
UBATCH_SIZE = 256

# Experiment timing
RUN_COOLDOWN_SECONDS = 10
BLOCK_COOLDOWN_SECONDS = 120

# EnergiBridge
ENERGYBRIDGE_INTERVAL_MS = 200

# External monitoring
MONITOR_INTERVAL_SECONDS = 0.2

# Name of the llama.cpp tokenizer helper binary used to get exact
# token counts (ships alongside llama-cli in every llama.cpp build).
LLAMA_TOKENIZE_BIN = "llama-tokenize"


# ============================================================
# COMMAND EXECUTION
# ============================================================

def run_command(command):
    print("\nRunning:")
    print(" ".join(command))
    print()

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    combined = result.stdout + "\n" + result.stderr

    # llama.cpp can occasionally return code 0 despite
    # reporting a generation/compute failure.
    failure_messages = [
        "Compute error",
        "Insufficient Memory",
        "failed to decode",
        "backend is in error state",
        "llama_decode() failed",
    ]

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code "
            f"{result.returncode}\n\n{combined}"
        )

    for message in failure_messages:
        if message in combined:
            raise RuntimeError(
                f"llama.cpp reported '{message}'\n\n"
                f"{combined}"
            )

    return result.stdout, result.stderr



def extract_llama_timing(text):

    metrics = {}

    # New compact footer, e.g.:
    #   [ Prompt: 50.5 t/s | Generation: 6.8 t/s ]
    match = re.search(
        r"\[\s*Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s\s*\]",
        text,
        re.IGNORECASE,
    )

    if match:
        metrics["prompt_tokens_per_second"] = float(match.group(1))
        metrics["generation_tokens_per_second"] = float(match.group(2))
        return metrics

    # Fallback: legacy verbose format, in case an older llama-cli
    # build is ever used again.
    match = re.search(
        r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens",
        text,
        re.IGNORECASE,
    )

    if match:
        metrics["prompt_eval_time_ms"] = float(match.group(1))
        metrics["prompt_tokens"] = int(match.group(2))

    match = re.search(
        r"(?<!prompt )eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*runs",
        text,
        re.IGNORECASE,
    )

    if match:
        metrics["generation_time_ms"] = float(match.group(1))
        metrics["output_tokens"] = int(match.group(2))

    return metrics


def extract_response_text(output_text):
    """
    Isolate just the model's generated reply from the raw llama-cli
    transcript captured in output_txt: everything after the echoed
    "> <prompt>" input line and before the closing
    "[ Prompt: ... | Generation: ... ]" footer.
    """

    lines = output_text.splitlines()

    start_idx = None

    for i, line in enumerate(lines):
        if line.startswith("> "):
            start_idx = i + 1
            break

    if start_idx is None:
        return None

    end_idx = len(lines)

    for i in range(start_idx, len(lines)):
        if re.match(r"^\s*\[\s*Prompt:.*t/s", lines[i], re.IGNORECASE):
            end_idx = i
            break

    response = "\n".join(lines[start_idx:end_idx]).strip()

    return response if response else None


def count_tokens(model, text, add_bos):
    """
    Exact token count via the `llama-tokenize` helper that ships
    with llama.cpp. Uses --ids so the output is a plain
    "[1, 2, 3, ...]" list we can count reliably, regardless of the
    exact wording any --show-count message uses in a given build.
    """

    if not text or not text.strip():
        return None

    tmp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            delete=False,
            encoding="utf-8",
        ) as tmp:

            tmp.write(text)
            tmp_path = tmp.name

        command = [
            LLAMA_TOKENIZE_BIN,
            "-m",
            str(model),
            "-f",
            tmp_path,
            "--ids",
            "--log-disable",
        ]

        if not add_bos:
            command.append("--no-bos")

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )

        match = re.search(r"\[([^\]]*)\]", result.stdout)

        if not match:
            print(
                f"Warning: could not parse token count from "
                f"llama-tokenize output: {result.stdout!r} "
                f"{result.stderr!r}"
            )
            return None

        ids_str = match.group(1).strip()

        if not ids_str:
            return 0

        return len(
            [x for x in ids_str.split(",") if x.strip() != ""]
        )

    except Exception as error:

        print(f"Warning: token counting failed: {error}")

        return None

    finally:

        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ============================================================
# SYSTEM MONITOR
# ============================================================

def get_cpu_usage():

    try:
        import psutil

        return float(
            psutil.cpu_percent(
                interval=None
            )
        )

    except Exception:
        return None


def get_memory_usage():

    try:
        import psutil

        memory = psutil.virtual_memory()

        return {
            "memory_used_mb":
                memory.used / 1024 / 1024,

            "memory_percent":
                memory.percent,

            "memory_available_mb":
                memory.available / 1024 / 1024,
        }

    except Exception:
        return {
            "memory_used_mb": None,
            "memory_percent": None,
            "memory_available_mb": None,
        }


def get_gpu_usage_macos():

    """
    Apple Silicon GPU utilization.

    Uses ioreg because the M4 does not expose GPU utilization
    through normal psutil APIs.

    Returns GPU utilization percentage when available.
    """

    try:

        result = subprocess.run(
            [
                "ioreg",
                "-r",
                "-d",
                "1",
                "-c",
                "IOAccelerator",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=1,
        )

        text = result.stdout

        patterns = [
            r'"Device Utilization %" = (\d+)',
            r'"Device Utilization %"=(\d+)',
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                text,
            )

            if match:
                return float(
                    match.group(1)
                )

    except Exception:
        pass

    return None


# ============================================================
# MONITOR THREAD
# ============================================================

class SystemMonitor:

    def __init__(
        self,
        interval=MONITOR_INTERVAL_SECONDS,
    ):

        self.interval = interval

        self.running = False

        self.thread = None

        self.cpu_samples = []
        self.gpu_samples = []

        self.memory_samples = []
        self.memory_percent_samples = []

    def _monitor(self):

        # Prime psutil CPU measurement
        try:
            import psutil
            psutil.cpu_percent(
                interval=None
            )
        except Exception:
            pass

        while self.running:

            # CPU
            cpu = get_cpu_usage()

            if cpu is not None:
                self.cpu_samples.append(cpu)

            # GPU
            gpu = get_gpu_usage_macos()

            if gpu is not None:
                self.gpu_samples.append(gpu)

            # Memory
            memory = get_memory_usage()

            if memory["memory_used_mb"] is not None:

                self.memory_samples.append(
                    memory["memory_used_mb"]
                )

                self.memory_percent_samples.append(
                    memory["memory_percent"]
                )

            time.sleep(
                self.interval
            )

    def start(self):

        self.running = True

        self.thread = threading.Thread(
            target=self._monitor,
            daemon=True,
        )

        self.thread.start()

    def stop(self):

        self.running = False

        if self.thread is not None:
            self.thread.join(
                timeout=2
            )

    @staticmethod
    def average(values):

        if not values:
            return None

        return sum(values) / len(values)

    @staticmethod
    def maximum(values):

        if not values:
            return None

        return max(values)

    def results(self):

        return {

            "cpu_usage_avg_pct":
                self.average(
                    self.cpu_samples
                ),

            "cpu_usage_max_pct":
                self.maximum(
                    self.cpu_samples
                ),

            "gpu_usage_avg_pct":
                self.average(
                    self.gpu_samples
                ),

            "gpu_usage_max_pct":
                self.maximum(
                    self.gpu_samples
                ),

            "memory_avg_mb":
                self.average(
                    self.memory_samples
                ),

            "memory_max_mb":
                self.maximum(
                    self.memory_samples
                ),

            "memory_avg_pct":
                self.average(
                    self.memory_percent_samples
                ),

            "memory_max_pct":
                self.maximum(
                    self.memory_percent_samples
                ),
        }


# ============================================================
# HELPERS
# ============================================================

def sleep_with_message(
    seconds,
    reason,
):

    print(
        f"\n{reason}"
    )

    print(
        f"Waiting {seconds} seconds..."
    )

    for remaining in range(
        seconds,
        0,
        -1,
    ):

        print(
            f"\rRemaining: {remaining:3d}s",
            end="",
            flush=True,
        )

        time.sleep(1)

    print()


def get_model_name(model_path):

    return Path(
        model_path
    ).stem


def save_metadata(
    path,
    metadata,
):

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )


# ============================================================
# SINGLE MEASURED RUN
# ============================================================

def run_single(
    model,
    quantization,
    prompt_size,
    prompt_file,
    run_number,
    output_tokens,
    raw_root,
):

    model_name = get_model_name(
        model
    )

    run_dir = (
        raw_root
        / model_name
        / quantization
        / prompt_size
        / f"run_{run_number:02d}"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    energy_csv = (
        run_dir / "energy.csv"
    )

    output_txt = (
        run_dir / "output.txt"
    )

    stderr_txt = (
        run_dir / "stderr.txt"
    )

    metadata_json = (
        run_dir / "metadata.json"
    )

    prompt = prompt_file.read_text(
        encoding="utf-8"
    ).strip()

    print()
    print("#" * 80)
    print(
        f"MODEL        : {model_name}"
    )
    print(
        f"QUANTIZATION : {quantization}"
    )
    print(
        f"PROMPT SIZE  : {prompt_size}"
    )
    print(
        f"RUN          : {run_number}"
    )
    print(
        f"OUTPUT TOKENS: {output_tokens}"
    )
    print("#" * 80)

    # --------------------------------------------------------
    # LLAMA COMMAND
    # --------------------------------------------------------

    llama_command = [

        "llama-cli",

        "-m",
        str(model),

        "-f",
        str(prompt_file),

        # GPU
        "-ngl",
        str(GPU_LAYERS),

        # Memory configuration
        "-c",
        str(CONTEXT_SIZE),

        "-b",
        str(BATCH_SIZE),

        "-ub",
        str(UBATCH_SIZE),

        # Fixed output limit
        "-n",
        str(output_tokens),

        # Deterministic
        "--temp",
        "0",

        "--seed",
        "42",

        # Disable Qwen reasoning
        "--reasoning-budget",
        "0",

        # Clean output
        "--no-display-prompt",
        "--single-turn",
        "--simple-io",

        # Do our own warm-up
        "--no-warmup",

        # llama.cpp performance information
        "--perf",
    ]

    # --------------------------------------------------------
    # ENERGIBRIDGE
    # --------------------------------------------------------

    energy_command = [

        "energibridge",

        "-o",
        str(energy_csv),

        "-c",
        str(output_txt),

        "-i",
        str(
            ENERGYBRIDGE_INTERVAL_MS
        ),

        "-g",

        "--summary",

        "--",
    ]

    energy_command.extend(
        llama_command
    )

    print()
    print(
        "Starting measurement..."
    )

    monitor = SystemMonitor()

    start_datetime = (
        datetime.now()
        .isoformat()
    )

    wall_start = (
        time.perf_counter()
    )

    monitor.start()

    try:

        stdout, stderr = run_command(
            energy_command
        )

    except Exception as error:

        monitor.stop()

        stderr_txt.write_text(
            str(error),
            encoding="utf-8",
        )

        raise

    finally:

        monitor.stop()

    wall_end = (
        time.perf_counter()
    )

    wall_clock_time = (
        wall_end - wall_start
    )

    # --------------------------------------------------------
    # llama.cpp metrics
    #
    # `stdout`/`stderr` here are energibridge's OWN streams, not
    # llama-cli's. llama-cli's transcript (including the timing
    # footer) was captured by energibridge into output_txt via
    # the `-c` flag, so that's what we must parse.
    # --------------------------------------------------------

    try:
        output_txt_content = output_txt.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except Exception as error:
        print(f"Warning: could not read {output_txt}: {error}")
        output_txt_content = ""

    llama_metrics = extract_llama_timing(
        output_txt_content
    )

    if not llama_metrics:
        # Fallback in case a future/older build writes timing info
        # to energibridge's own stdout/stderr instead.
        combined_output = stdout + "\n" + stderr
        llama_metrics = extract_llama_timing(combined_output)

    # Exact token counts, independent of whatever llama-cli prints.
    prompt_tokens = count_tokens(
        model,
        prompt,
        add_bos=True,
    )

    response_text = extract_response_text(
        output_txt_content
    )

    output_tokens_actual = count_tokens(
        model,
        response_text,
        add_bos=False,
    ) if response_text else None

    if prompt_tokens is not None:
        llama_metrics["prompt_tokens"] = prompt_tokens

    if output_tokens_actual is not None:
        llama_metrics["output_tokens"] = output_tokens_actual

    # Derive millisecond timings from throughput + exact counts,
    # when both are available and we don't already have them from
    # the (legacy) verbose format.
    if (
        "prompt_eval_time_ms" not in llama_metrics
        and "prompt_tokens_per_second" in llama_metrics
        and llama_metrics.get("prompt_tokens_per_second", 0) > 0
        and prompt_tokens
    ):
        llama_metrics["prompt_eval_time_ms"] = (
            prompt_tokens
            / llama_metrics["prompt_tokens_per_second"]
            * 1000.0
        )

    if (
        "generation_time_ms" not in llama_metrics
        and "generation_tokens_per_second" in llama_metrics
        and llama_metrics.get("generation_tokens_per_second", 0) > 0
        and output_tokens_actual
    ):
        llama_metrics["generation_time_ms"] = (
            output_tokens_actual
            / llama_metrics["generation_tokens_per_second"]
            * 1000.0
        )

    monitor_metrics = (
        monitor.results()
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = {

        "model":
            model_name,

        "model_path":
            str(model),

        "quantization":
            quantization,

        "prompt_size":
            prompt_size,

        "prompt_file":
            str(prompt_file),

        "run":
            run_number,

        "requested_output_tokens":
            output_tokens,

        "actual_output_tokens":
            llama_metrics.get(
                "output_tokens"
            ),

        "measurement_start":
            start_datetime,

        "measurement_end":
            datetime.now().isoformat(),

        "wall_clock_time_s":
            wall_clock_time,

        "energibridge_interval_ms":
            ENERGYBRIDGE_INTERVAL_MS,

        # llama.cpp
        "llama_metrics":
            llama_metrics,

        # independent system monitoring
        "system_metrics":
            monitor_metrics,

        # experiment configuration
        "configuration": {

            "gpu_layers":
                GPU_LAYERS,

            "context_size":
                CONTEXT_SIZE,

            "batch_size":
                BATCH_SIZE,

            "ubatch_size":
                UBATCH_SIZE,

            "output_tokens":
                output_tokens,

            "temperature":
                0,

            "seed":
                42,

            "reasoning_budget":
                0,
        },
    }

    save_metadata(
        metadata_json,
        metadata,
    )

    print()
    print(
        "Run complete."
    )

    print(
        f"Wall time: "
        f"{wall_clock_time:.3f}s"
    )

    if llama_metrics:

        print(
            "\nllama.cpp metrics:"
        )

        for key, value in (
            llama_metrics.items()
        ):

            print(
                f"  {key}: {value}"
            )

    print(
        "\nSystem metrics:"
    )

    for key, value in (
        monitor_metrics.items()
    ):

        print(
            f"  {key}: {value}"
        )

    return run_dir


# ============================================================
# WARM-UP
# ============================================================

def warmup(
    model,
    prompt_file,
    output_tokens,
):

    print()
    print("=" * 70)
    print(
        f"WARM-UP: {prompt_file.name}"
    )
    print("=" * 70)

    prompt = prompt_file.read_text(
        encoding="utf-8"
    ).strip()

    command = [

        "llama-cli",

        "-m",
        str(model),

        "-f",
        str(prompt_file),

        "-ngl",
        str(GPU_LAYERS),

        "-c",
        str(CONTEXT_SIZE),

        "-b",
        str(BATCH_SIZE),

        "-ub",
        str(UBATCH_SIZE),

        "-n",
        str(output_tokens),

        "--temp",
        "0",

        "--seed",
        "42",

        "--reasoning-budget",
        "0",

        "--no-display-prompt",

        "--single-turn",

        "--simple-io",
    ]

    run_command(
        command
    )

    print(
        "Warm-up complete."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=
        "Run local LLM energy benchmark."
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to GGUF model",
    )

    parser.add_argument(
        "--quantization",
        required=True,
        help="Quantization label",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
    )

    parser.add_argument(
        "--output-tokens",
        type=int,
        default=DEFAULT_OUTPUT_TOKENS,
    )

    parser.add_argument(
        "--prompt-dir",
        default="prompts/",
    )

    parser.add_argument(
        "--results-dir",
        default="results/raw",
    )

    parser.add_argument(
        "--skip-cooldown",
        action="store_true",
    )

    args = parser.parse_args()

    model = Path(
        args.model
    ).expanduser().resolve()

    if not model.exists():

        print(
            f"ERROR: model does not exist: "
            f"{model}"
        )

        sys.exit(1)

    prompt_dir = Path(
        args.prompt_dir
    ).expanduser().resolve()

    raw_root = Path(
        args.results_dir
    ).expanduser().resolve()

    # Check prompts

    for size, filename in (
        PROMPT_SIZES.items()
    ):

        prompt_file = (
            prompt_dir / filename
        )

        if not prompt_file.exists():

            print(
                f"ERROR: missing prompt: "
                f"{prompt_file}"
            )

            sys.exit(1)

    print()
    print("=" * 70)
    print(
        "LOCAL LLM ENERGY EXPERIMENT"
    )
    print("=" * 70)

    print(
        f"Model:          {model.name}"
    )

    print(
        f"Quantization:   "
        f"{args.quantization}"
    )

    print(
        f"Runs/size:      {args.runs}"
    )

    print(
        f"Output tokens:  "
        f"{args.output_tokens}"
    )

    print(
        f"GPU layers:     "
        f"{GPU_LAYERS}"
    )

    print(
        f"Context:        "
        f"{CONTEXT_SIZE}"
    )

    print(
        f"Batch:          "
        f"{BATCH_SIZE}"
    )

    print(
        f"Micro-batch:    "
        f"{UBATCH_SIZE}"
    )

    print(
        "Prompt sizes:    "
        "small / medium / large"
    )

    print("=" * 70)

    input(
        "\nPress ENTER to start..."
    )

    sizes = list(
        PROMPT_SIZES.items()
    )

    for size_index, (
        prompt_size,
        filename,
    ) in enumerate(sizes):

        prompt_file = (
            prompt_dir / filename
        )

        # One warm-up per prompt size
        warmup(
            model,
            prompt_file,
            args.output_tokens,
        )

        if not args.skip_cooldown:

            sleep_with_message(
                30,
                "Stabilizing after warm-up..."
            )

        for run_number in range(
            1,
            args.runs + 1,
        ):

            run_single(
                model=model,
                quantization=
                    args.quantization,
                prompt_size=
                    prompt_size,
                prompt_file=
                    prompt_file,
                run_number=
                    run_number,
                output_tokens=
                    args.output_tokens,
                raw_root=
                    raw_root,
            )

            if (
                run_number < args.runs
                and not args.skip_cooldown
            ):

                sleep_with_message(
                    RUN_COOLDOWN_SECONDS,
                    "Cooldown between runs..."
                )

        if (
            size_index
            < len(sizes) - 1
            and not args.skip_cooldown
        ):

            sleep_with_message(
                BLOCK_COOLDOWN_SECONDS,
                "Cooldown before next prompt size..."
            )

    print()
    print("=" * 70)
    print(
        "BENCHMARK COMPLETE"
    )
    print("=" * 70)

    print(
        f"Raw results saved to:\n"
        f"{raw_root}"
    )


if __name__ == "__main__":
    main()