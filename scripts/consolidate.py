
import argparse
import csv
import json
import re
from pathlib import Path

import pandas as pd


# ============================================================
# HELPERS
# ============================================================

def read_metadata(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def find_column(df, candidates):

    columns_lower = {
        c.lower(): c
        for c in df.columns
    }

    for candidate in candidates:

        candidate_lower = candidate.lower()

        if candidate_lower in columns_lower:
            return columns_lower[candidate_lower]

    # Partial matching
    for column in df.columns:

        column_lower = column.lower()

        for candidate in candidates:

            if candidate.lower() in column_lower:
                return column

    return None


def mean_column(df, candidates):

    column = find_column(
        df,
        candidates,
    )

    if column is None:
        return None

    values = pd.to_numeric(
        df[column],
        errors="coerce",
    )

    if values.dropna().empty:
        return None

    return float(values.mean())


def max_column(df, candidates):

    column = find_column(
        df,
        candidates,
    )

    if column is None:
        return None

    values = pd.to_numeric(
        df[column],
        errors="coerce",
    )

    if values.dropna().empty:
        return None

    return float(values.max())


def calculate_energy_from_power(df):

    power_column = find_column(
        df,
        [
            "SYSTEM_POWER (Watts)",
            "GPU_POWER (Watts)",
            "CPU_POWER (Watts)",
        ],
    )

    delta_column = find_column(
        df,
        [
            "Delta",
        ],
    )

    if (
        power_column is None
        or delta_column is None
    ):
        return None

    power = pd.to_numeric(
        df[power_column],
        errors="coerce",
    )

    delta_ms = pd.to_numeric(
        df[delta_column],
        errors="coerce",
    )

    valid = (
        power.notna()
        & delta_ms.notna()
    )

    if not valid.any():
        return None

    # J = W × seconds
    energy_j = (
        power[valid]
        * delta_ms[valid]
        / 1000.0
    ).sum()

    return float(energy_j)


def extract_summary_energy(metadata_path):

    """
    EnergiBridge --summary is printed to the terminal.

    The benchmark currently keeps the raw CSV as the
    authoritative measurement source.

    This function exists for future summary parsing.
    """

    return None


# ============================================================
# PROCESS ONE RUN
# ============================================================

def process_run(run_dir):

    energy_file = run_dir / "energy.csv"
    metadata_file = run_dir / "metadata.json"

    if not energy_file.exists():
        return None

    metadata = read_metadata(
        metadata_file
    )

    try:
        df = pd.read_csv(
            energy_file
        )
    except Exception as e:

        print(
            f"Could not read {energy_file}: {e}"
        )

        return None

    # --------------------------------------------------------
    # Basic metadata
    # --------------------------------------------------------

    llama_metrics = metadata.get(
        "llama_metrics",
        {},
    )

    model = metadata.get(
        "model",
        "",
    )

    quantization = metadata.get(
        "quantization",
        "",
    )

    prompt_size = metadata.get(
        "prompt_size",
        "",
    )

    run_number = metadata.get(
        "run",
        "",
    )

    # --------------------------------------------------------
    # Energy
    # --------------------------------------------------------

    energy_j = calculate_energy_from_power(
        df
    )

    # --------------------------------------------------------
    # Execution time
    # --------------------------------------------------------

    execution_time_s = None

    if "Delta" in df.columns:

        delta = pd.to_numeric(
            df["Delta"],
            errors="coerce",
        )

        execution_time_s = (
            delta.sum() / 1000.0
        )

    elif metadata.get(
        "wall_clock_time_s"
    ):

        execution_time_s = metadata[
            "wall_clock_time_s"
        ]

    # --------------------------------------------------------
    # CPU
    # --------------------------------------------------------

    cpu_columns = [
        column
        for column in df.columns
        if re.match(
            r"CPU_USAGE_\d+",
            column,
            re.IGNORECASE,
        )
    ]

    cpu_usage = None

    if cpu_columns:

        cpu_values = df[
            cpu_columns
        ].apply(
            pd.to_numeric,
            errors="coerce",
        )

        cpu_usage = float(
            cpu_values.mean(
                axis=1
            ).mean()
        )

    else:

        cpu_usage = mean_column(
            df,
            [
                "CPU_USAGE",
                "CPU usage",
            ],
        )

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    gpu_usage = mean_column(
        df,
        [
            "GPU_UTILIZATION (%)",
            "GPU_USAGE (%)",
            "GPU_UTILIZATION",
            "GPU_USAGE",
        ],
    )

    gpu_power = mean_column(
        df,
        [
            "GPU_POWER (Watts)",
            "GPU_POWER",
        ],
    )

    gpu_memory = mean_column(
        df,
        [
            "GPU_MEMORY (Bytes)",
            "GPU_MEMORY",
        ],
    )

    # --------------------------------------------------------
    # System memory
    # --------------------------------------------------------

    used_memory = mean_column(
        df,
        [
            "USED_MEMORY",
            "USED_MEMORY (Bytes)",
        ],
    )

    max_memory = max_column(
        df,
        [
            "USED_MEMORY",
            "USED_MEMORY (Bytes)",
        ],
    )

    # Convert bytes → MB
    if used_memory is not None:
        used_memory_mb = (
            used_memory / 1024 / 1024
        )
    else:
        used_memory_mb = None

    if max_memory is not None:
        max_memory_mb = (
            max_memory / 1024 / 1024
        )
    else:
        max_memory_mb = None

    # --------------------------------------------------------
    # GPU memory
    # --------------------------------------------------------

    if gpu_memory is not None:
        gpu_memory_mb = (
            gpu_memory / 1024 / 1024
        )
    else:
        gpu_memory_mb = None

    # --------------------------------------------------------
    # Output tokens
    # --------------------------------------------------------

    output_tokens = llama_metrics.get(
        "output_tokens"
    )

    prompt_tokens = llama_metrics.get(
        "prompt_tokens"
    )

    # If output count isn't available,
    # keep it empty rather than inventing it.
    # This is important for scientific analysis.

    energy_per_token = None

    if (
        energy_j is not None
        and output_tokens
        and output_tokens > 0
    ):
        energy_per_token = (
            energy_j
            / output_tokens
        )

    # --------------------------------------------------------
    # GPU energy
    # --------------------------------------------------------

    gpu_energy_j = None

    if (
        gpu_power is not None
        and execution_time_s is not None
    ):
        gpu_energy_j = (
            gpu_power
            * execution_time_s
        )

    # --------------------------------------------------------
    # Final row
    # --------------------------------------------------------

    return {
        "model": model,
        "quantization": quantization,
        "prompt_size": prompt_size,

        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,

        "execution_time_s":
            execution_time_s,

        "energy_j":
            energy_j,

        "energy_j_per_output_token":
            energy_per_token,

        "cpu_usage_pct":
            cpu_usage,

        "gpu_usage_pct":
            gpu_usage,

        "gpu_power_w":
            gpu_power,

        "gpu_energy_j":
            gpu_energy_j,

        "memory_mb":
            used_memory_mb,

        "max_memory_mb":
            max_memory_mb,

        "gpu_memory_mb":
            gpu_memory_mb,

        "run":
            run_number,

        "run_directory":
            str(run_dir),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="results/raw",
    )

    parser.add_argument(
        "--output",
        default="results/consolidated/local_results.csv",
    )

    args = parser.parse_args()

    input_dir = Path(
        args.input
    ).expanduser().resolve()

    output_file = Path(
        args.output
    ).expanduser().resolve()

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Searching:\n{input_dir}"
    )

    rows = []

    for metadata_file in sorted(
        input_dir.rglob("metadata.json")
    ):

        run_dir = metadata_file.parent

        row = process_run(
            run_dir
        )

        if row is not None:
            rows.append(row)

    if not rows:

        print(
            "No valid benchmark runs found."
        )

        return

    df = pd.DataFrame(
        rows
    )

    # Sort scientifically:
    # model → quantization → prompt size → run
    prompt_order = {
        "small": 0,
        "medium": 1,
        "large": 2,
    }

    df["_prompt_order"] = (
        df["prompt_size"]
        .map(prompt_order)
        .fillna(99)
    )

    df = df.sort_values(
        [
            "model",
            "quantization",
            "_prompt_order",
            "run",
        ]
    )

    df = df.drop(
        columns=["_prompt_order"]
    )

    df.to_csv(
        output_file,
        index=False,
    )

    print("\n")
    print("=" * 70)
    print("CONSOLIDATION COMPLETE")
    print("=" * 70)

    print(
        f"Runs: {len(df)}"
    )

    print(
        f"Output:\n{output_file}"
    )

    print("\nColumns:")
    for column in df.columns:
        print(f"  {column}")


if __name__ == "__main__":
    main()
