import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

PROMPTS = {
    "small": "prompt_small.txt",
    "medium": "prompt_medium.txt",
    "large": "prompt_large.txt",
}

DEFAULT_RUNS = 10
DEFAULT_OUTPUT_TOKENS = 512

# Cooldown between measured runs
RUN_COOLDOWN_SECONDS = 0

# Cooldown between prompt sizes
BLOCK_COOLDOWN_SECONDS = 120

# Time between EnergiBridge samples
ENERGYBRIDGE_INTERVAL_MS = 200

# llama.cpp memory configuration
CONTEXT_SIZE = 2048

BATCH_SIZE = 512

UBATCH_SIZE = 256

# GPU offload
GPU_LAYERS = 99


# ============================================================
# UTILITY
# ============================================================

def wait_with_countdown(seconds, message):

    if seconds <= 0:
        return

    print()
    print(message)

    for remaining in range(seconds, 0, -1):

        print(
            f"\rRemaining: {remaining:3d} seconds",
            end="",
            flush=True,
        )

        time.sleep(1)

    print()


# ============================================================
# RUN COMMAND
# ============================================================

def run_command(command):

    print()
    print("=" * 80)
    print("COMMAND")
    print("=" * 80)

    print(" ".join(command))

    print()
    print("=" * 80)

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    print("Return code:", result.returncode)

    if result.stdout:
        print()
        print("--- STDOUT ---")
        print(result.stdout)

    if result.stderr:
        print()
        print("--- STDERR ---")
        print(result.stderr)

    if result.returncode != 0:

        raise RuntimeError(
            f"Command failed with return code "
            f"{result.returncode}"
        )

    return result.stdout, result.stderr


# ============================================================
# WARM-UP
# ============================================================

def warmup(model, prompt_file):

    print()
    print("=" * 80)
    print("WARM-UP")
    print("=" * 80)

    prompt = prompt_file.read_text(
        encoding="utf-8"
    ).strip()

    command = [

        "llama-cli",

        "-m",
        str(model),

        # Prompt
        "-p",
        prompt,

        # GPU offload
        "-ngl",
        str(GPU_LAYERS),

        # Controlled memory usage
        "-c",
        str(CONTEXT_SIZE),

        "-b",
        str(BATCH_SIZE),

        "-ub",
        str(UBATCH_SIZE),

        # Warm-up does not need full generation
        "-n",
        "128",

        # Deterministic generation
        "--temp",
        "0",

        "--seed",
        "42",

        # Do not print prompt
        "--no-display-prompt",

        # One generation then exit
        "--single-turn",

        # Clean output
        "--simple-io",
        "--reasoning-budget" ,
        "0"

    ]

    stdout, stderr = run_command(command)

    print()
    print("Warm-up completed successfully.")

    return stdout, stderr


# ============================================================
# MEASURED RUN
# ============================================================

def measured_run(
    model,
    prompt_file,
    model_name,
    quantization,
    prompt_size,
    run_number,
    output_tokens,
    results_dir,
):

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    run_dir = (
        results_dir
        / model_name
        / quantization
        / prompt_size
        / f"run_{run_number:02d}"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    energy_csv = run_dir / "energy.csv"
    output_txt = run_dir / "output.txt"
    stderr_txt = run_dir / "stderr.txt"
    metadata_json = run_dir / "metadata.json"

    # --------------------------------------------------------
    # Read prompt
    # --------------------------------------------------------

    prompt = prompt_file.read_text(
        encoding="utf-8"
    ).strip()

    # --------------------------------------------------------
    # Display experiment information
    # --------------------------------------------------------

    print()
    print("#" * 80)
    print(f"MODEL        : {model_name}")
    print(f"QUANTIZATION : {quantization}")
    print(f"PROMPT SIZE  : {prompt_size}")
    print(f"RUN          : {run_number}")
    print(f"OUTPUT TOKENS: {output_tokens}")
    print("#" * 80)

    # --------------------------------------------------------
    # llama.cpp command
    # --------------------------------------------------------

    llama_command = [

        "llama-cli",

        "-m",
        str(model),

        # ----------------------------------------------------
        # IMPORTANT:
        # Pass prompt directly instead of using -f
        # ----------------------------------------------------

        "-p",
        prompt,

        # ----------------------------------------------------
        # GPU
        # ----------------------------------------------------

        "-ngl",
        str(GPU_LAYERS),

        # ----------------------------------------------------
        # MEMORY CONTROL
        # ----------------------------------------------------

        "-c",
        str(CONTEXT_SIZE),

        "-b",
        str(BATCH_SIZE),

        "-ub",
        str(UBATCH_SIZE),

        # ----------------------------------------------------
        # OUTPUT LENGTH
        # ----------------------------------------------------

        "-n",
        str(output_tokens),

        # ----------------------------------------------------
        # DETERMINISTIC GENERATION
        # ----------------------------------------------------

        "--temp",
        "0",

        "--seed",
        "42",

        # ----------------------------------------------------
        # OUTPUT OPTIONS
        # ----------------------------------------------------

        "--no-display-prompt",

        "--single-turn",

        "--simple-io",
        "--reasoning-budget" ,
        "0"

        # ----------------------------------------------------
        # We perform our own warm-up
        # ----------------------------------------------------

        "--no-warmup",

        # ----------------------------------------------------
        # llama.cpp performance information
        # ----------------------------------------------------

        "--perf",
    ]

    # --------------------------------------------------------
    # EnergiBridge command
    # --------------------------------------------------------

    energy_command = [

        "energibridge",

        "-o",
        str(energy_csv),

        "-c",
        str(output_txt),

        "-i",
        str(ENERGYBRIDGE_INTERVAL_MS),

        # GPU monitoring
        "-g",

        # Energy summary
        "--summary",

        "--",
    ]

    # Append llama.cpp command
    energy_command.extend(
        llama_command
    )

    # --------------------------------------------------------
    # Start measurement
    # --------------------------------------------------------

    start_time = datetime.now()

    wall_start = time.perf_counter()

    try:

        stdout, stderr = run_command(
            energy_command
        )

    except Exception as error:

        wall_end = time.perf_counter()

        # Save error information
        stderr_txt.write_text(
            str(error),
            encoding="utf-8",
        )

        metadata = {

            "model": model_name,

            "model_path": str(model),

            "quantization": quantization,

            "prompt_size": prompt_size,

            "run": run_number,

            "requested_output_tokens":
                output_tokens,

            "context_size":
                CONTEXT_SIZE,

            "batch_size":
                BATCH_SIZE,

            "ubatch_size":
                UBATCH_SIZE,

            "gpu_layers":
                GPU_LAYERS,

            "energibridge_interval_ms":
                ENERGYBRIDGE_INTERVAL_MS,

            "status":
                "FAILED",

            "error":
                str(error),

            "start_time":
                start_time.isoformat(),

            "wall_clock_time_s":
                wall_end - wall_start,
        }

        metadata_json.write_text(
            json.dumps(
                metadata,
                indent=2,
            ),
            encoding="utf-8",
        )

        print()
        print("!" * 80)
        print("MEASURED RUN FAILED")
        print("!" * 80)
        print(error)

        raise

    wall_end = time.perf_counter()

    end_time = datetime.now()

    # --------------------------------------------------------
    # Save stdout
    # --------------------------------------------------------

    # EnergiBridge -c normally writes the application output.
    # We additionally save llama stdout if available.

    if stdout:

        llama_stdout_file = (
            run_dir / "llama_stdout.txt"
        )

        llama_stdout_file.write_text(
            stdout,
            encoding="utf-8",
        )

    # --------------------------------------------------------
    # Save stderr
    # --------------------------------------------------------

    stderr_txt.write_text(
        stderr,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Check generated output
    # --------------------------------------------------------

    output_exists = output_txt.exists()

    output_size = (
        output_txt.stat().st_size
        if output_exists
        else 0
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {

        "model": model_name,

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

        "context_size":
            CONTEXT_SIZE,

        "batch_size":
            BATCH_SIZE,

        "ubatch_size":
            UBATCH_SIZE,

        "gpu_layers":
            GPU_LAYERS,

        "temperature":
            0,

        "seed":
            42,

        "energibridge_interval_ms":
            ENERGYBRIDGE_INTERVAL_MS,

        "start_time":
            start_time.isoformat(),

        "end_time":
            end_time.isoformat(),

        "wall_clock_time_s":
            wall_end - wall_start,

        "output_file_exists":
            output_exists,

        "output_file_size_bytes":
            output_size,

        "energy_file_exists":
            energy_csv.exists(),

        "status":
            "SUCCESS",
    }

    metadata_json.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Validate results
    # --------------------------------------------------------

    if not output_exists:

        raise RuntimeError(
            "Run finished but output.txt "
            "was not created."
        )

    if output_size == 0:

        raise RuntimeError(
            "Run finished but output.txt "
            "is empty."
        )

    if not energy_csv.exists():

        raise RuntimeError(
            "Run finished but energy.csv "
            "was not created."
        )

    # --------------------------------------------------------
    # Success
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("MEASURED RUN SUCCESSFUL")
    print("=" * 80)

    print(
        "Output:",
        output_txt,
    )

    print(
        "Energy:",
        energy_csv,
    )

    print(
        "Metadata:",
        metadata_json,
    )

    print(
        "Wall-clock time:",
        f"{wall_end - wall_start:.3f} s",
    )

    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run local LLM energy benchmark "
            "using llama.cpp and EnergiBridge."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to GGUF model",
    )

    parser.add_argument(
        "--quantization",
        required=True,
        help="Quantization label, e.g. Q4_K_M",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help="Number of measured runs per prompt size",
    )

    parser.add_argument(
        "--output-tokens",
        type=int,
        default=DEFAULT_OUTPUT_TOKENS,
        help="Maximum generated tokens",
    )

    parser.add_argument(
        "--prompt-dir",
        default="prompts",
        help="Directory containing prompt files",
    )

    parser.add_argument(
        "--results-dir",
        default="results/raw",
        help="Output directory",
    )

    args = parser.parse_args()

    # ========================================================
    # PATHS
    # ========================================================

    model = Path(
        args.model
    ).expanduser().resolve()

    prompt_dir = Path(
        args.prompt_dir
    ).expanduser().resolve()

    results_dir = Path(
        args.results_dir
    ).expanduser().resolve()

    # ========================================================
    # VALIDATION
    # ========================================================

    if not model.exists():

        print(
            f"ERROR: model does not exist:\n{model}"
        )

        sys.exit(1)

    for prompt_filename in PROMPTS.values():

        prompt_path = (
            prompt_dir / prompt_filename
        )

        if not prompt_path.exists():

            print(
                "ERROR: prompt file does not exist:"
            )

            print(prompt_path)

            sys.exit(1)

    # ========================================================
    # MODEL NAME
    # ========================================================

    model_name = model.stem

    # ========================================================
    # EXPERIMENT INFORMATION
    # ========================================================

    print()
    print("=" * 80)
    print("LOCAL LLM ENERGY EXPERIMENT")
    print("=" * 80)

    print(
        f"Model           : {model_name}"
    )

    print(
        f"Quantization    : {args.quantization}"
    )

    print(
        f"Runs/size       : {args.runs}"
    )

    print(
        f"Output tokens   : {args.output_tokens}"
    )

    print(
        "Prompt sizes    : Small / Medium / Large"
    )

    print(
        f"GPU layers      : {GPU_LAYERS}"
    )

    print(
        f"Context         : {CONTEXT_SIZE}"
    )

    print(
        f"Batch           : {BATCH_SIZE}"
    )

    print(
        f"Micro-batch     : {UBATCH_SIZE}"
    )

    print(
        f"EB interval     : {ENERGYBRIDGE_INTERVAL_MS} ms"
    )

    print(
        f"Run cooldown    : {RUN_COOLDOWN_SECONDS} s"
    )

    print(
        f"Block cooldown  : {BLOCK_COOLDOWN_SECONDS} s"
    )

    print("=" * 80)

    input(
        "\nPress ENTER to begin the experiment..."
    )

    # ========================================================
    # PROMPT SIZES
    # ========================================================

    prompt_sizes = [
        "small",
        "medium",
        "large",
    ]

    for block_index, prompt_size in enumerate(
        prompt_sizes
    ):

        prompt_file = (
            prompt_dir
            / PROMPTS[prompt_size]
        )

        print()
        print()
        print("#" * 80)
        print(
            f"PROMPT SIZE: {prompt_size.upper()}"
        )
        print("#" * 80)

        # ====================================================
        # WARM-UP ONCE
        # ====================================================

        warmup(
            model,
            prompt_file,
        )

        # ====================================================
        # STABILIZATION
        # ====================================================

        wait_with_countdown(
            30,
            "Stabilizing system after warm-up..."
        )

        # ====================================================
        # MEASURED RUNS
        # ====================================================

        for run_number in range(
            1,
            args.runs + 1,
        ):

            measured_run(
                model=model,

                prompt_file=prompt_file,

                model_name=model_name,

                quantization=args.quantization,

                prompt_size=prompt_size,

                run_number=run_number,

                output_tokens=args.output_tokens,

                results_dir=results_dir,
            )

            # -----------------------------------------------
            # Cooldown between measured runs
            # -----------------------------------------------

            if run_number < args.runs:

                wait_with_countdown(
                    RUN_COOLDOWN_SECONDS,
                    "Cooldown before next measured run..."
                )

        # ====================================================
        # COOLDOWN BETWEEN PROMPT SIZES
        # ====================================================

        if block_index < len(prompt_sizes) - 1:

            wait_with_countdown(
                BLOCK_COOLDOWN_SECONDS,
                "Cooldown before next prompt size..."
            )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print()
    print("=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)

    print(
        f"Results stored in:\n{results_dir}"
    )

    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()